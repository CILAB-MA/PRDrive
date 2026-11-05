
import numpy as np


class LevelSampler:
    def __init__(
        self,
        num_maps,
        strategy="prioritized",  # "prioritized" or "uniform"
        score_transform="rank",  # "rank" or "power". Rank is required for negative MaxMC scores.
        temperature=2.0,  # beta: rank/power temperature
        staleness_coef=0.3,  # mix weight on staleness vs score
        staleness_transform="power",
        staleness_temperature=1.0,
        rho=0.2,  # fraction of maps that must be seen before replay can start
        nu=0.5,  # once past rho, probability of replaying vs. sampling unseen
        alpha=1.0,  # score EMA rate: new = (1 - alpha) * old + alpha * latest
        eps=1e-3,  # power-transform smoothing so a never-updated score isn't exactly 0
        seed=None,
        score_aggregate="mean",  # how per-agent scores collapse to one map score
        gae_gamma=0.99,  # discount used when PVL builds GAE from a stored rollout
        gae_lambda=0.95,  # GAE lambda used when PVL builds advantages from a stored rollout
        max_controlled_agents=32,  # per-vehicle MaxMC R_max slots, matching Drive max_controlled_agents
        invert_scores=False,  # if True, rank low stored scores first (episode return)
        outcome_alpha=0.2,  # EMA for per-vehicle fail/collision rates (score alpha is often 1)
    ):
        if strategy not in ("prioritized", "uniform", "sequential"):
            raise ValueError(
                f"LevelSampler strategy must be 'prioritized', 'uniform', or 'sequential', got {strategy!r}"
            )
        if score_transform not in ("power", "rank"):
            raise ValueError(f"score_transform must be 'power' or 'rank', got {score_transform!r}")
        if staleness_transform not in ("power", "rank"):
            raise ValueError(f"staleness_transform must be 'power' or 'rank', got {staleness_transform!r}")

        self.num_maps = int(num_maps)
        if self.num_maps < 1:
            raise ValueError("num_maps must be >= 1")

        self.strategy = strategy
        self.score_transform = score_transform
        self.temperature = float(temperature)
        self.staleness_coef = float(staleness_coef)
        self.staleness_transform = staleness_transform
        self.staleness_temperature = float(staleness_temperature)
        self.rho = float(rho)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.score_aggregate = score_aggregate
        if self.score_aggregate not in ("mean", "max", "min"):
            raise ValueError(f"score_aggregate must be 'mean', 'max', or 'min', got {score_aggregate!r}")
        self.gae_gamma = float(gae_gamma)
        self.gae_lambda = float(gae_lambda)
        self.max_controlled_agents = int(max_controlled_agents)
        if self.max_controlled_agents < 1:
            raise ValueError("max_controlled_agents must be >= 1")
        self.invert_scores = bool(invert_scores)
        self.outcome_alpha = float(outcome_alpha)
        if not (0.0 < self.outcome_alpha <= 1.0):
            raise ValueError(f"outcome_alpha must be in (0, 1], got {outcome_alpha!r}")
        self._rng = np.random.default_rng(seed)

        # Per-map state.
        self.scores = np.zeros(self.num_maps, dtype=np.float64)
        self.staleness = np.zeros(self.num_maps, dtype=np.float64)
        self.seen = np.zeros(self.num_maps, dtype=bool)

        # C caps controllable agents at MAX_AGENTS=32; training npy max is also 32.
        self.max_returns = np.zeros((self.num_maps, self.max_controlled_agents), dtype=np.float64)
        self.max_return_valid = np.zeros((self.num_maps, self.max_controlled_agents), dtype=bool)
        # Per-vehicle fail / collision EMA. Slot index is the map's controllable order.
        self.fail_ema = np.zeros((self.num_maps, self.max_controlled_agents), dtype=np.float64)
        self.coll_ema = np.zeros((self.num_maps, self.max_controlled_agents), dtype=np.float64)
        self.hit_ema = np.zeros((self.num_maps, self.max_controlled_agents), dtype=np.float64)
        self.outcome_n = np.zeros((self.num_maps, self.max_controlled_agents), dtype=np.float64)
        self._next_index = 0

    # ---- weighting -------------------------------------------------

    def _transform(self, transform, temperature, values):
        values = np.asarray(values, dtype=np.float64)
        if transform == "power":
            e = 0.0 if self.staleness_coef > 0 else self.eps
            return (values + e) ** (1.0 / temperature)
        order = np.flip(values.argsort())
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order)) + 1
        return 1.0 / (ranks ** (1.0 / temperature))

    def _sample_weights(self):
        score_values = np.array(self.scores, copy=True)
        if self.invert_scores:
            score_values = -score_values
        if self.score_transform == "rank":
            score_values[~self.seen] = -np.inf
        weights = self._transform(self.score_transform, self.temperature, score_values)
        weights = weights * self.seen  # never-scored maps get zero weight here
        total = weights.sum()
        if total > 0:
            weights = weights / total

        if self.staleness_coef > 0:
            staleness_weights = self._transform(self.staleness_transform, self.staleness_temperature, self.staleness)
            staleness_weights = staleness_weights * self.seen
            s_total = staleness_weights.sum()
            if s_total > 0:
                staleness_weights = staleness_weights / s_total
            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights
        return weights

    # ---- picking one map id -----------------------------------------

    def _touch(self, map_id):
        """Every time a map is handed out, its staleness resets and everyone else's grows."""
        self.staleness += 1
        self.staleness[map_id] = 0

    def _sample_replay(self):
        weights = self._sample_weights()
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total > 0 and np.isfinite(total):
            map_id = int(self._rng.choice(self.num_maps, p=weights / total))
        else:
            map_id = int(self._rng.integers(self.num_maps))
        self._touch(map_id)
        return map_id

    def _sample_unseen(self):
        unseen = np.flatnonzero(~self.seen)
        map_id = int(self._rng.choice(unseen))
        self._touch(map_id)
        return map_id

    def sample_one(self):

        if self.strategy == "sequential":
            map_id = int(self._next_index % self.num_maps)
            self._next_index += 1
            self._touch(map_id)
            return map_id

        if self.strategy == "uniform":
            map_id = int(self._rng.integers(self.num_maps))
            self._touch(map_id)
            return map_id

        num_unseen = int((~self.seen).sum())
        proportion_seen = (self.num_maps - num_unseen) / self.num_maps

        if proportion_seen >= self.rho:
            if self._rng.random() > self.nu or proportion_seen >= 1.0:
                return self._sample_replay()
        return self._sample_unseen()

    def sample_batch(self, batch_size):
        out = np.empty(int(batch_size), dtype=np.int32)
        for i in range(out.shape[0]):
            out[i] = self.sample_one()

        num_unseen = int((~self.seen).sum())
        metrics = {
            "level_sampler/proportion_seen": float((self.num_maps - num_unseen) / self.num_maps),
            "level_sampler/mean_score": float(self.scores[self.seen].mean()) if self.seen.any() else 0.0,
            "level_sampler/max_score": float(self.scores[self.seen].max()) if self.seen.any() else 0.0,
            "level_sampler/mean_staleness": float(self.staleness.mean()),
        }
        return out, metrics

    def sample_until(self, agent_counts, agent_budget):
        agent_counts = np.asarray(agent_counts)
        if agent_counts.shape[0] < self.num_maps:
            raise ValueError(
                f"agent_counts has {agent_counts.shape[0]} entries, need at least num_maps={self.num_maps}"
            )
        budget = int(agent_budget)
        if budget < 1:
            raise ValueError(f"agent_budget must be >= 1, got {budget}")

        chosen = []
        total = 0
        for _ in range(self.num_maps):
            map_id = self.sample_one()
            chosen.append(map_id)
            total += int(agent_counts[map_id])
            if total >= budget:
                break
        else:
            raise RuntimeError(
                f"one pass over {self.num_maps} maps has {total} controllable agents, below budget {budget}"
            )

        out = np.asarray(chosen, dtype=np.int32)
        num_unseen = int((~self.seen).sum())
        metrics = {
            "level_sampler/proportion_seen": float((self.num_maps - num_unseen) / self.num_maps),
            "level_sampler/mean_score": float(self.scores[self.seen].mean()) if self.seen.any() else 0.0,
            "level_sampler/max_score": float(self.scores[self.seen].max()) if self.seen.any() else 0.0,
            "level_sampler/mean_staleness": float(self.staleness.mean()),
            "level_sampler/queue_maps": float(out.shape[0]),
            "level_sampler/queue_agents": float(total),
        }
        return out, metrics

    # ---- feeding results back in --------------------------------------

    def update(self, map_id, score):
        """Record the outcome for one map after it was actually played."""
        map_id = int(map_id)
        if not (0 <= map_id < self.num_maps):
            raise ValueError(f"map_id {map_id} out of range [0, {self.num_maps})")
        self.seen[map_id] = True
        old = self.scores[map_id]
        self.scores[map_id] = (1 - self.alpha) * old + self.alpha * float(score)

    def update_batch(self, map_ids, scores):
        """Vectorized `update()` for multiple (map_id, score) pairs from one rollout."""
        map_ids = np.asarray(map_ids, dtype=np.int64).reshape(-1)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if map_ids.shape[0] != scores.shape[0]:
            raise ValueError("map_ids and scores must have the same length")
        for map_id, score in zip(map_ids, scores):
            self.update(map_id, score)

    def score_max_mc(self, map_id, rewards, values, dones):
        """Score one map and store it in ``scores[map_id]``.

        map_id: int. One map. rewards, values, and dones are already sliced to this map.
        rewards: (T, A) float. Reward at each step for each controllable agent on this map.
        values: (T, A) float. V(s) of the state those rewards leave, same layout as rewards.
        dones: (T, A) float. 1 where that agent ends an episode at that step, else 0.
        T is the steps recorded since the last resample. A is the controllable agents on this map.

        self.max_returns[map_id]: (max_controlled_agents,) best episodic return per vehicle so far.
        self.max_return_valid[map_id]: True only for vehicles that have a stored R_max.
            Unused padding stays False so a 0 slot is not treated as R_max=0.
        agent_scores: (A,) float. Each agent's mean of (that agent's R_max - V(s)).
        new_max: (A,) updated per-vehicle max returns for this visit.
        reduced: scalar. agent_scores collapsed by score_aggregate ("mean", "max", or "min").
        scores[map_id]: scalar buffer. EMA of reduced. This is the value Drive reads.
        """
        map_id = int(map_id)
        rewards = np.asarray(rewards)
        n_agents = 1 if rewards.ndim < 2 else int(rewards.shape[1])
        n_store = min(n_agents, self.max_returns.shape[1])
        prev = np.full(n_agents, -np.inf, dtype=np.float64)
        valid = self.max_return_valid[map_id, :n_store]
        prev[:n_store] = np.where(valid, self.max_returns[map_id, :n_store], -np.inf)
        agent_scores, new_max = compute_agent_max_mc(
            rewards, values, dones, prev, gamma=self.gae_gamma
        )
        new_max = np.asarray(new_max, dtype=np.float64).reshape(-1)
        self.max_returns[map_id, :n_store] = new_max[:n_store]
        self.max_return_valid[map_id, :n_store] = True
        reduced = aggregate_agent_scores(agent_scores[:n_store], self.score_aggregate)
        self.update(map_id, reduced)
        return float(self.scores[map_id])

    def score_pvl(self, map_id, advantages, dones):
        """Score one map with Positive Value Loss and store it in ``scores[map_id]``.

        advantages: (T, A) float. GAE (or any advantage) at each step for each
            controllable agent on this map.
        dones: (T, A) float. 1 where that agent ends an episode at that step.
        Per-agent score is the mean of max(advantage, 0) over each episode,
        then mean over episodes (unfinished tail counts, same as MaxMC).
        """
        map_id = int(map_id)
        reduced = aggregate_agent_scores(compute_agent_pvl(advantages, dones), self.score_aggregate)
        self.update(map_id, reduced)
        return float(self.scores[map_id])

    def score_pvl_rollout(
        self, map_id, rewards, values, dones, gamma=None, gae_lambda=None, bootstrap_values=None
    ):
        """Build GAE from a stored (r, V, done) rollout, then ``score_pvl``.

        Drive records those arrays the same way as MaxMC. PPO advantages are
        not available at resample time, so GAE is computed here with
        ``gae_gamma`` / ``gae_lambda``. ``bootstrap_values`` is V(s_T) for a
        truncated last step; if omitted, that backup is 0.
        """
        advantages = compute_gae(
            rewards,
            values,
            dones,
            gamma=self.gae_gamma if gamma is None else gamma,
            gae_lambda=self.gae_lambda if gae_lambda is None else gae_lambda,
            bootstrap_values=bootstrap_values,
        )
        return self.score_pvl(map_id, advantages, dones)

    def score_learnability(self, map_id, successes, dones=None, unsafe=None):
        """Score one map with goal-success learnability and store it in ``scores[map_id]``.

        successes: (T, A) 1 on steps where that agent reached its goal, else 0.
            An episode succeeds if it hit the goal at least once.
        dones: (T, A) optional. 1 where that agent ends an episode.
        unsafe: (T, A) optional. 1 on collision/off-road steps. When given
            (Learn-hard), an episode succeeds only if it reached the goal and
            never collided or went off-road.
        The map score is the mean over vehicles of p_n(1-p_n). Peaks at 0.25
        when a vehicle succeeds on half of its own episodes.
        """
        map_id = int(map_id)
        reduced = compute_learnability(successes, dones, unsafe=unsafe)
        self.update(map_id, reduced)
        return float(self.scores[map_id])

    def score(
        self,
        map_id,
        score_function,
        rewards=None,
        values=None,
        dones=None,
        advantages=None,
        successes=None,
        unsafe=None,
        bootstrap_values=None,
    ):
        """Dispatch to MaxMC, PVL, or learnability and write ``scores[map_id]``.

        Mirrors Kinetix ``compute_score``. ``advantages`` is optional for PVL;
        if omitted, GAE is built from ``rewards`` / ``values`` / ``dones``.
        Learnability uses ``successes`` (goal reached) and ``dones``.
        Learn-hard also uses ``unsafe`` (collision/off-road).
        """
        name = _normalize_score_function(score_function)
        if name == "max_mc":
            if self.score_transform != "rank":
                raise ValueError(
                    "MaxMC scores can be negative, so score_transform must be 'rank'. "
                    f"Got: {self.score_transform!r}"
                )
            reduced = self.score_max_mc(map_id, rewards, values, dones)
            self.update_agent_outcomes(map_id, rewards=rewards, dones=dones)
            return reduced
        if name == "pvl":
            if advantages is None:
                reduced = self.score_pvl_rollout(
                    map_id, rewards, values, dones, bootstrap_values=bootstrap_values
                )
            else:
                reduced = self.score_pvl(map_id, advantages, dones)
            if rewards is not None and dones is not None:
                self.update_agent_outcomes(map_id, rewards=rewards, dones=dones)
            return reduced
        if name == "learnability":
            reduced = self.score_learnability(map_id, successes, dones, unsafe=None)
            self.update_agent_outcomes(map_id, successes=successes, dones=dones, unsafe=unsafe)
            return reduced
        if name == "learnability_hard":
            reduced = self.score_learnability(map_id, successes, dones, unsafe=unsafe)
            self.update_agent_outcomes(map_id, successes=successes, dones=dones, unsafe=unsafe)
            return reduced
        raise ValueError(f"Unknown score function: {score_function!r}")

    def update_agent_outcomes(self, map_id, successes=None, dones=None, unsafe=None, rewards=None):
        """EMA of per-vehicle fail, collision, and hit-count on completed episodes.

        Open resample tails are ignored. Fail is 1 if the finished episode did
        not score (no goal, or unsafe when given). Collision is 1 if unsafe
        fired. Hits are unsafe steps in that episode. ``rewards > 0`` is goal
        and ``rewards < 0`` is unsafe when only rewards are passed.
        """
        if dones is None:
            return
        if successes is None:
            if rewards is None:
                return
            rewards = np.asarray(rewards)
            successes = (rewards > 0).astype(np.float64)
            if unsafe is None:
                unsafe = (rewards < 0).astype(np.float64)
        fail_rate, coll_rate, hit_mean, n_ep = compute_agent_outcomes(successes, dones, unsafe)
        map_id = int(map_id)
        n_store = min(int(fail_rate.shape[0]), self.fail_ema.shape[1])
        alpha = self.outcome_alpha
        for agent in range(n_store):
            visits = float(n_ep[agent])
            if visits <= 0:
                continue
            if self.outcome_n[map_id, agent] <= 0:
                self.fail_ema[map_id, agent] = float(fail_rate[agent])
                self.coll_ema[map_id, agent] = float(coll_rate[agent])
                self.hit_ema[map_id, agent] = float(hit_mean[agent])
            else:
                self.fail_ema[map_id, agent] = (1.0 - alpha) * self.fail_ema[map_id, agent] + alpha * float(
                    fail_rate[agent]
                )
                self.coll_ema[map_id, agent] = (1.0 - alpha) * self.coll_ema[map_id, agent] + alpha * float(
                    coll_rate[agent]
                )
                self.hit_ema[map_id, agent] = (1.0 - alpha) * self.hit_ema[map_id, agent] + alpha * float(
                    hit_mean[agent]
                )
            self.outcome_n[map_id, agent] += visits


def _split_sampler_state(state):
    if isinstance(state, dict):
        return (
            state["scores"],
            state["seen"],
            state.get("fail_ema"),
            state.get("coll_ema"),
            state.get("hit_ema"),
            state.get("outcome_n"),
        )
    scores, seen = state[0], state[1]
    if len(state) >= 6:
        return scores, seen, state[2], state[3], state[4], state[5]
    if len(state) >= 5:
        return scores, seen, state[2], state[3], None, state[4]
    return scores, seen, None, None, None, None


def merge_sampler_states(states):
    states = list(states)
    if not states:
        return None
    first = _split_sampler_state(states[0])
    num_maps = int(np.asarray(first[0]).shape[0])
    score_sum = np.zeros(num_maps, dtype=np.float64)
    seen_count = np.zeros(num_maps, dtype=np.float64)
    fail_sum = coll_sum = hit_sum = n_sum = None
    n_agents = None
    for state in states:
        scores, seen, fail_ema, coll_ema, hit_ema, outcome_n = _split_sampler_state(state)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        seen = np.asarray(seen, dtype=bool).reshape(-1)
        if scores.shape[0] != num_maps or seen.shape[0] != num_maps:
            raise ValueError("all sampler states must have the same num_maps")
        score_sum += np.where(seen, scores, 0.0)
        seen_count += seen.astype(np.float64)
        if fail_ema is None or coll_ema is None or outcome_n is None:
            continue
        fail_ema = np.asarray(fail_ema, dtype=np.float64)
        coll_ema = np.asarray(coll_ema, dtype=np.float64)
        outcome_n = np.asarray(outcome_n, dtype=np.float64)
        if hit_ema is None:
            hit_ema = np.zeros_like(fail_ema)
        else:
            hit_ema = np.asarray(hit_ema, dtype=np.float64)
        if fail_ema.shape != coll_ema.shape or fail_ema.shape != outcome_n.shape or fail_ema.shape != hit_ema.shape:
            raise ValueError("fail_ema, coll_ema, hit_ema, and outcome_n must match")
        if fail_sum is None:
            n_agents = fail_ema.shape[1] if fail_ema.ndim == 2 else 1
            fail_sum = np.zeros((num_maps, n_agents), dtype=np.float64)
            coll_sum = np.zeros_like(fail_sum)
            hit_sum = np.zeros_like(fail_sum)
            n_sum = np.zeros_like(fail_sum)
        fail_sum += fail_ema * outcome_n
        coll_sum += coll_ema * outcome_n
        hit_sum += hit_ema * outcome_n
        n_sum += outcome_n
    merged_seen = seen_count > 0
    merged_scores = np.zeros(num_maps, dtype=np.float64)
    np.divide(score_sum, seen_count, out=merged_scores, where=merged_seen)
    if fail_sum is None:
        return merged_scores, merged_seen
    merged_fail = np.zeros_like(fail_sum)
    merged_coll = np.zeros_like(coll_sum)
    merged_hit = np.zeros_like(hit_sum)
    np.divide(fail_sum, n_sum, out=merged_fail, where=n_sum > 0)
    np.divide(coll_sum, n_sum, out=merged_coll, where=n_sum > 0)
    np.divide(hit_sum, n_sum, out=merged_hit, where=n_sum > 0)
    return {
        "scores": merged_scores,
        "seen": merged_seen,
        "fail_ema": merged_fail,
        "coll_ema": merged_coll,
        "hit_ema": merged_hit,
        "outcome_n": n_sum,
    }


def score_replay_weights(
    scores, seen, score_transform="rank", temperature=2.0, eps=1e-3, invert_scores=False
):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    seen = np.asarray(seen, dtype=bool).reshape(-1)
    sampler = LevelSampler(
        num_maps=int(scores.shape[0]),
        score_transform=score_transform,
        temperature=temperature,
        staleness_coef=0.0,
        eps=eps,
        invert_scores=invert_scores,
    )
    sampler.scores = np.array(scores, copy=True)
    sampler.seen = np.array(seen, copy=True)
    return sampler._sample_weights()


# CL4AD 10K: β is rank temperature, ρ is staleness coefficient.
# Appendix E.3: MaxMC 10K uses β=4; 1K/80K use β=2.
PLR_SCORE_HYPERS = {
    "max_mc": (4.0, 0.3),
    "pvl": (4.0, 0.3),
    "learnability": (4.0, 0.1),
    "learnability_hard": (2.0, 0.1),
    "return": (2.0, 0.3),
}


def _score_key(score_function):
    name = str(score_function).strip().lower().replace("-", "_")
    if name in ("maxmc", "max_mc"):
        return "max_mc"
    if name in ("pvl", "positive_value_loss"):
        return "pvl"
    if name in ("learn_hard", "learnability_hard"):
        return "learnability_hard"
    if name in ("learn", "learnability"):
        return "learnability"
    if name in ("return", "episode_return"):
        return "return"
    return name


def plr_hypers_for_score(map_score, temperature=None, staleness_coef=None):
    """CL4AD 10K (β, ρ) for a map score. Explicit values win over the preset."""
    beta, rho = PLR_SCORE_HYPERS.get(_score_key(map_score), (2.0, 0.3))
    if temperature is not None:
        beta = float(temperature)
    if staleness_coef is not None:
        rho = float(staleness_coef)
    return beta, rho


def _normalize_score_function(score_function):
    return _score_key(score_function)


def aggregate_agent_scores(agent_scores, score_aggregate="mean"):
    """Collapse one score per agent into the score for their map."""
    agent_scores = np.asarray(agent_scores, dtype=np.float64).reshape(-1)
    if agent_scores.size == 0:
        return 0.0
    if score_aggregate == "mean":
        return float(agent_scores.mean())
    if score_aggregate == "max":
        return float(agent_scores.max())
    if score_aggregate == "min":
        return float(agent_scores.min())
    raise ValueError(f"agent aggregate must be 'mean', 'max', or 'min', got {score_aggregate!r}")


def _broadcast_agent_max_return(max_return, n_agents):
    prev = np.asarray(max_return, dtype=np.float64)
    if prev.ndim == 0:
        return np.full(n_agents, float(prev), dtype=np.float64)
    prev = prev.reshape(-1)
    if prev.shape[0] == n_agents:
        return np.array(prev, copy=True)
    out = np.zeros(n_agents, dtype=np.float64)
    n = min(n_agents, prev.shape[0])
    out[:n] = prev[:n]
    return out


def episode_live_mask(dones):
    """Skip Drive re-asserting ``terminals=1`` after a car already finished.

    Returns ``(live, finish)`` bool arrays shaped like 2d ``dones``. ``live``
    is an in-progress episode, including its first done. ``finish`` is that
    first done. Open tails are live with no finish. Consecutive terminals
    after a finish are neither.
    """
    done = np.asarray(dones) > 0
    timesteps, agents = done.shape
    live = np.empty((timesteps, agents), dtype=bool)
    finish = np.empty((timesteps, agents), dtype=bool)
    prev_done = np.zeros(agents, dtype=bool)
    for t in range(timesteps):
        is_done = done[t]
        step_live = ~(prev_done & is_done)
        live[t] = step_live
        finish[t] = is_done & step_live
        prev_done = is_done
    return live, finish


def compute_agent_max_mc(rewards, values, dones, max_return, gamma=0.99):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    if rewards.ndim == 1:
        rewards = rewards[:, None]
        values = values[:, None]
        dones = dones[:, None]
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have the same shape")
    if rewards.size == 0:
        prev = np.asarray(max_return, dtype=np.float64).reshape(-1)
        return np.zeros(0, dtype=np.float64), prev

    timesteps, agents = rewards.shape
    live, finish = episode_live_mask(dones)
    prev = _broadcast_agent_max_return(max_return, agents)
    gamma = float(gamma)
    partial_return = np.zeros(agents, dtype=np.float64)
    discount = np.ones(agents, dtype=np.float64)
    max_ep_return = np.full(agents, -np.inf, dtype=np.float64)
    for t in range(timesteps):
        partial_return = np.where(live[t], partial_return + discount * rewards[t], partial_return)
        discount = np.where(live[t], discount * gamma, discount)
        max_ep_return = np.where(finish[t], np.maximum(max_ep_return, partial_return), max_ep_return)
        partial_return = np.where(finish[t], 0.0, partial_return)
        discount = np.where(finish[t], 1.0, discount)
    # Unfinished tail still counts as an episode return.
    max_ep_return = np.where(np.isfinite(max_ep_return), max_ep_return, partial_return)
    agent_max = np.maximum(prev, max_ep_return)

    partial_gap = np.zeros(agents, dtype=np.float64)
    partial_steps = np.zeros(agents, dtype=np.float64)
    sum_ep_gap = np.zeros(agents, dtype=np.float64)
    n_episodes = np.zeros(agents, dtype=np.float64)
    for t in range(timesteps):
        partial_gap = np.where(live[t], partial_gap + (agent_max - values[t]), partial_gap)
        partial_steps = np.where(live[t], partial_steps + 1.0, partial_steps)
        sum_ep_gap = np.where(finish[t], sum_ep_gap + partial_gap / np.maximum(partial_steps, 1.0), sum_ep_gap)
        n_episodes = np.where(finish[t], n_episodes + 1.0, n_episodes)
        partial_gap = np.where(finish[t], 0.0, partial_gap)
        partial_steps = np.where(finish[t], 0.0, partial_steps)
    open_tail = partial_steps > 0
    tail_gap = np.divide(partial_gap, partial_steps, out=np.zeros_like(partial_gap), where=open_tail)
    sum_ep_gap = np.where(open_tail, sum_ep_gap + tail_gap, sum_ep_gap)
    n_episodes = np.where(open_tail, n_episodes + 1.0, n_episodes)
    n_episodes = np.maximum(n_episodes, 1.0)
    return sum_ep_gap / n_episodes, agent_max


def compute_max_mc(rewards, values, dones, max_return, score_aggregate="mean", gamma=0.99):
    """Agent MaxMC scores reduced to one map score.

    Returns (map_score, updated_max_return) where ``updated_max_return`` is
    (A,) per-vehicle discounted \(R_{\max}\). ``score_aggregate`` selects how
    the per-agent scores are combined.
    """
    agent_scores, agent_max = compute_agent_max_mc(
        rewards, values, dones, max_return, gamma=gamma
    )
    return aggregate_agent_scores(agent_scores, score_aggregate), agent_max


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95, bootstrap_values=None):
    """GAE(lambda) over a stored rollout. Same layout as MaxMC: (T, A) or (T,).

    values[t] is V(s_t) and rewards[t] is r_t, matching Drive's MaxMC buffers.
    Interior steps use V(s_{t+1})=values[t+1]. The last step has no V(s_T) in
    the buffer: pass it as ``bootstrap_values`` (A,). If omitted and the last
    step is not done, the backup is 0 rather than V(s_{T-1}). Done steps cut
    the backup.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    squeeze = rewards.ndim == 1
    if squeeze:
        rewards = rewards[:, None]
        values = values[:, None]
        dones = dones[:, None]
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values, and dones must have the same shape")
    if rewards.size == 0:
        return np.zeros((0, 0), dtype=np.float64) if not squeeze else np.zeros(0, dtype=np.float64)

    timesteps, agents = rewards.shape
    live, finish = episode_live_mask(dones)
    advantages = np.zeros((timesteps, agents), dtype=np.float64)
    lastgaelam = np.zeros(agents, dtype=np.float64)
    gamma = float(gamma)
    lam = float(gae_lambda)
    if bootstrap_values is None:
        bootstrap = np.zeros(agents, dtype=np.float64)
    else:
        bootstrap = np.asarray(bootstrap_values, dtype=np.float64).reshape(-1)
        if bootstrap.shape[0] != agents:
            raise ValueError("bootstrap_values must have one value per agent")
    for t in range(timesteps - 1, -1, -1):
        nonterminal = (~finish[t]).astype(np.float64)
        next_v = values[t + 1] if t + 1 < timesteps else bootstrap
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        new_lam = delta + gamma * lam * nonterminal * lastgaelam
        lastgaelam = np.where(live[t], new_lam, lastgaelam)
        advantages[t] = np.where(live[t], lastgaelam, 0.0)
    if squeeze:
        return advantages[:, 0]
    return advantages


def compute_agent_pvl(advantages, dones):
    """Per-agent Positive Value Loss. Returns (A,) mean of max(A_t, 0) per episode."""
    advantages = np.asarray(advantages, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    if advantages.ndim == 1:
        advantages = advantages[:, None]
        dones = dones[:, None]
    if advantages.shape != dones.shape:
        raise ValueError("advantages and dones must have the same shape")
    if advantages.size == 0:
        return np.zeros(0, dtype=np.float64)

    pos = np.maximum(advantages, 0.0)
    timesteps, agents = pos.shape
    live, finish = episode_live_mask(dones)
    partial_sum = np.zeros(agents, dtype=np.float64)
    partial_steps = np.zeros(agents, dtype=np.float64)
    sum_ep = np.zeros(agents, dtype=np.float64)
    n_episodes = np.zeros(agents, dtype=np.float64)
    for t in range(timesteps):
        partial_sum = np.where(live[t], partial_sum + pos[t], partial_sum)
        partial_steps = np.where(live[t], partial_steps + 1.0, partial_steps)
        sum_ep = np.where(finish[t], sum_ep + partial_sum / np.maximum(partial_steps, 1.0), sum_ep)
        n_episodes = np.where(finish[t], n_episodes + 1.0, n_episodes)
        partial_sum = np.where(finish[t], 0.0, partial_sum)
        partial_steps = np.where(finish[t], 0.0, partial_steps)
    open_tail = partial_steps > 0
    tail = np.divide(partial_sum, partial_steps, out=np.zeros_like(partial_sum), where=open_tail)
    sum_ep = np.where(open_tail, sum_ep + tail, sum_ep)
    n_episodes = np.where(open_tail, n_episodes + 1.0, n_episodes)
    n_episodes = np.maximum(n_episodes, 1.0)
    return sum_ep / n_episodes


def compute_pvl(advantages, dones, score_aggregate="mean"):
    """Agent PVL scores reduced to one map score."""
    return aggregate_agent_scores(compute_agent_pvl(advantages, dones), score_aggregate)


def compute_agent_outcomes(successes, dones, unsafe=None):
    """Per-agent fail, collision, and hit counts over **completed** episodes.

    A life starts on a non-done step (or the first done after a live stretch).
    Consecutive terminals after a car already finished — Drive re-asserts
    ``terminals=1`` until the map resets — are skipped. Open resample tails
    never get a done, so they are ignored.

    Fail is 1 if a finished episode did not score (no goal, or ``unsafe``
    fired). Collision is 1 if ``unsafe`` fired. Hits are unsafe steps in that
    episode. Returns ``(fail_rate, coll_rate, hit_mean, n_episodes)``.
    """
    successes = np.asarray(successes, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    if successes.ndim == 1:
        successes = successes[:, None]
        dones = dones[:, None]
        if unsafe is not None:
            unsafe = np.asarray(unsafe, dtype=np.float64)[:, None]
    if successes.shape != dones.shape:
        raise ValueError("successes and dones must have the same shape")
    if unsafe is not None:
        unsafe = np.asarray(unsafe, dtype=np.float64)
        if unsafe.shape != successes.shape:
            raise ValueError("unsafe must have the same shape as successes")

    live, finish = episode_live_mask(dones)
    _, agents = successes.shape
    partial_goal = np.zeros(agents, dtype=np.float64)
    partial_unsafe = np.zeros(agents, dtype=np.float64)
    partial_hits = np.zeros(agents, dtype=np.float64)
    n_fail = np.zeros(agents, dtype=np.float64)
    n_coll = np.zeros(agents, dtype=np.float64)
    n_hits = np.zeros(agents, dtype=np.float64)
    n_episodes = np.zeros(agents, dtype=np.float64)
    for t in range(successes.shape[0]):
        if unsafe is None:
            step_unsafe = np.zeros(agents, dtype=np.float64)
        else:
            step_unsafe = (unsafe[t] > 0).astype(np.float64)
        step_goal = (successes[t] > 0).astype(np.float64)
        partial_goal = np.where(live[t], np.maximum(partial_goal, step_goal), partial_goal)
        partial_unsafe = np.where(live[t], np.maximum(partial_unsafe, step_unsafe), partial_unsafe)
        partial_hits = np.where(live[t], partial_hits + step_unsafe, partial_hits)
        scored = partial_goal if unsafe is None else partial_goal * (1.0 - partial_unsafe)
        n_fail += np.where(finish[t], 1.0 - scored, 0.0)
        n_coll += np.where(finish[t], partial_unsafe, 0.0)
        n_hits += np.where(finish[t], partial_hits, 0.0)
        n_episodes += finish[t].astype(np.float64)
        partial_goal = np.where(finish[t], 0.0, partial_goal)
        partial_unsafe = np.where(finish[t], 0.0, partial_unsafe)
        partial_hits = np.where(finish[t], 0.0, partial_hits)
    fail_rate = np.divide(n_fail, n_episodes, out=np.zeros_like(n_fail), where=n_episodes > 0)
    coll_rate = np.divide(n_coll, n_episodes, out=np.zeros_like(n_coll), where=n_episodes > 0)
    hit_mean = np.divide(n_hits, n_episodes, out=np.zeros_like(n_hits), where=n_episodes > 0)
    return fail_rate, coll_rate, hit_mean, n_episodes


def fail_top1_share(fail_ema, outcome_n, min_episodes=2):
    """Fraction of fail-mass on the worst vehicle. None if no slot has enough visits."""
    fail = np.asarray(fail_ema, dtype=np.float64).reshape(-1)
    visits = np.asarray(outcome_n, dtype=np.float64).reshape(-1)
    if fail.shape[0] != visits.shape[0]:
        raise ValueError("fail_ema and outcome_n must have the same length")
    valid = visits >= float(min_episodes)
    if not valid.any():
        return None
    mass = fail[valid]
    total = float(mass.sum())
    if total <= 0:
        return 0.0
    return float(mass.max() / total)


def fail_worst_stat(fail_ema, stat_ema, outcome_n, min_episodes=2):
    """``stat_ema`` of the vehicle that owns the most fail mass. None if unused."""
    fail = np.asarray(fail_ema, dtype=np.float64).reshape(-1)
    stat = np.asarray(stat_ema, dtype=np.float64).reshape(-1)
    visits = np.asarray(outcome_n, dtype=np.float64).reshape(-1)
    if fail.shape != stat.shape or fail.shape != visits.shape:
        raise ValueError("fail_ema, stat_ema, and outcome_n must match")
    valid = visits >= float(min_episodes)
    if not valid.any():
        return None
    masked = np.where(valid, fail, -np.inf)
    if not np.isfinite(masked).any() or float(fail[valid].sum()) <= 0:
        return 0.0
    return float(stat[int(np.argmax(masked))])


def fail_worst_collision(fail_ema, coll_ema, outcome_n, min_episodes=2):
    """Collision EMA of the vehicle that owns the most fail mass."""
    return fail_worst_stat(fail_ema, coll_ema, outcome_n, min_episodes=min_episodes)


def compute_learnability(successes, dones=None, unsafe=None):
    """Map learnability as mean over vehicles of p_n(1-p_n).

    successes: (T, A) flag that the agent reached its goal at that step, or a
        1d array of already-finished episode outcomes for one agent.
    dones: (T, A) optional episode ends. An episode succeeds if it saw a goal
        hit at least once. Repeated terminals after a finish are skipped.
        Unfinished tails count, same as MaxMC.
    unsafe: (T, A) optional collision/off-road flags. When given, an episode
        succeeds only if it reached the goal and never set unsafe (Learn-hard).
    Each vehicle gets its own success rate p_n. Always-succeed and
        always-fail vehicles both contribute 0, so a 50/50 split across cars
        is not treated as high learnability.
    """
    successes = np.asarray(successes, dtype=np.float64)
    if successes.size == 0:
        return 0.0
    if dones is None:
        outcomes = (successes.reshape(-1) > 0).astype(np.float64)
        p = float(outcomes.mean())
        return p * (1.0 - p)

    dones = np.asarray(dones, dtype=np.float64)
    if successes.ndim == 1:
        successes = successes[:, None]
        dones = dones[:, None]
        if unsafe is not None:
            unsafe = np.asarray(unsafe, dtype=np.float64)[:, None]
    if successes.shape != dones.shape:
        raise ValueError("successes and dones must have the same shape")
    if unsafe is not None:
        unsafe = np.asarray(unsafe, dtype=np.float64)
        if unsafe.shape != successes.shape:
            raise ValueError("unsafe must have the same shape as successes")

    timesteps, agents = successes.shape
    live, finish = episode_live_mask(dones)
    partial_hit = np.zeros(agents, dtype=np.float64)
    partial_unsafe = np.zeros(agents, dtype=np.float64)
    partial_steps = np.zeros(agents, dtype=np.float64)
    n_success = np.zeros(agents, dtype=np.float64)
    n_episodes = np.zeros(agents, dtype=np.float64)
    for t in range(timesteps):
        step_goal = (successes[t] > 0).astype(np.float64)
        partial_hit = np.where(live[t], np.maximum(partial_hit, step_goal), partial_hit)
        if unsafe is not None:
            step_unsafe = (unsafe[t] > 0).astype(np.float64)
            partial_unsafe = np.where(live[t], np.maximum(partial_unsafe, step_unsafe), partial_unsafe)
        partial_steps = np.where(live[t], partial_steps + 1.0, partial_steps)
        ep_ok = partial_hit if unsafe is None else partial_hit * (1.0 - partial_unsafe)
        n_success = np.where(finish[t], n_success + ep_ok, n_success)
        n_episodes = np.where(finish[t], n_episodes + 1.0, n_episodes)
        partial_hit = np.where(finish[t], 0.0, partial_hit)
        partial_unsafe = np.where(finish[t], 0.0, partial_unsafe)
        partial_steps = np.where(finish[t], 0.0, partial_steps)
    open_tail = partial_steps > 0
    ep_ok = partial_hit if unsafe is None else partial_hit * (1.0 - partial_unsafe)
    n_success += np.where(open_tail, ep_ok, 0.0)
    n_episodes += open_tail.astype(np.float64)
    n_episodes = np.maximum(n_episodes, 1.0)
    p = n_success / n_episodes
    return float(np.mean(p * (1.0 - p)))


def compute_score(
    score_function,
    *,
    dones,
    values=None,
    max_return=0.0,
    rewards=None,
    advantages=None,
    successes=None,
    unsafe=None,
    score_aggregate="mean",
    gamma=0.99,
    gae_lambda=0.95,
    bootstrap_values=None,
):
    """Pick MaxMC, PVL, or learnability. Returns (map_score, extra) where extra
    is the per-vehicle updated max return for MaxMC and NaN otherwise.

    Mirrors Kinetix ``compute_score``. PVL uses ``advantages`` when given,
    otherwise GAE from ``rewards`` / ``values`` / ``dones``. Learnability uses
    ``successes`` (goal reached) and ``dones``. Learn-hard also uses ``unsafe``.
    """
    name = _normalize_score_function(score_function)
    if name == "max_mc":
        return compute_max_mc(
            rewards, values, dones, max_return, score_aggregate, gamma=gamma
        )
    if name == "pvl":
        if advantages is None:
            advantages = compute_gae(
                rewards, values, dones, gamma, gae_lambda, bootstrap_values=bootstrap_values
            )
        return compute_pvl(advantages, dones, score_aggregate), float("nan")
    if name == "learnability":
        return compute_learnability(successes, dones), float("nan")
    if name == "learnability_hard":
        return compute_learnability(successes, dones, unsafe=unsafe), float("nan")
    raise ValueError(f"Unknown score function: {score_function!r}")
