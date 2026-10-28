
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
        agent_aggregate="mean",  # how per-agent scores collapse to one map score
        gae_gamma=0.99,  # discount used when PVL builds GAE from a stored rollout
        gae_lambda=0.95,  # GAE lambda used when PVL builds advantages from a stored rollout
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
        self.agent_aggregate = agent_aggregate
        if self.agent_aggregate not in ("mean", "max", "min"):
            raise ValueError(f"agent_aggregate must be 'mean', 'max', or 'min', got {agent_aggregate!r}")
        self.gae_gamma = float(gae_gamma)
        self.gae_lambda = float(gae_lambda)
        self._rng = np.random.default_rng(seed)

        # Per-map state.
        self.scores = np.zeros(self.num_maps, dtype=np.float64)
        self.staleness = np.zeros(self.num_maps, dtype=np.float64)
        self.seen = np.zeros(self.num_maps, dtype=bool)

        # Best episodic return seen on each map. Used by MaxMC (minimax compute_max_mc).
        self.max_returns = np.zeros(self.num_maps, dtype=np.float64)
        self._next_index = 0

    # ---- weighting -------------------------------------------------

    def _transform(self, transform, temperature, values):
        values = np.asarray(values, dtype=np.float64)
        if transform == "power":
            e = 0.0 if self.staleness_coef > 0 else self.eps
            return (values + e) ** (1.0 / temperature)
        # "rank": highest value gets rank 1, weight 1 / rank^(1/temperature).
        # Matches minimax PLR when use_score_ranks=True. Sign of the score
        # only changes order, so negative MaxMC scores stay valid weights.
        order = np.flip(values.argsort())
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order)) + 1
        return 1.0 / (ranks ** (1.0 / temperature))

    def _sample_weights(self):
        """Sampling distribution over already-seen maps, blending score and staleness."""
        score_values = np.array(self.scores, copy=True)
        if self.score_transform == "rank":
            # Unseen maps are not scored. Keep them below every real score so
            # they do not take the top ranks away from negative MaxMC scores.
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
            # Nothing scored yet despite being "seen" (shouldn't normally happen) -- fall back.
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
        """Return an ordered (int32) array of `batch_size` candidate map ids.

        This is what gets passed to `binding.shared(map_id_queue=...)`. C
        walks the array in order and stops once it has collected enough
        controllable agents, so `batch_size` should comfortably over-estimate
        how many maps a resample could plausibly need -- unused entries at
        the tail are simply never consumed.

        Returns (map_ids, metrics) where metrics is a small dict suitable
        for logging (matches the wandb_metrics convention used by
        CurriculumSampler.sample() / AgentSampler.sample()).
        """
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
        """Sample maps until their controllable-agent counts cover `agent_budget`.

        `agent_counts[map_id]` is the count C's `set_active_agents` reports for
        that map. A map is taken whole, so the sum may exceed the budget; the
        next call continues at the following map. Returns (map_ids, metrics).
        """
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

        self.max_returns[map_id]: scalar. Best episodic return seen on this map so far.
        agent_scores: (A,) float. Each agent's mean of (map best return - V(s)).
        new_max: scalar. max(previous max_returns[map_id], best return in this visit).
        reduced: scalar. agent_scores collapsed by agent_aggregate ("mean", "max", or "min").
        scores[map_id]: scalar buffer. EMA of reduced. This is the value Drive reads.
        """
        map_id = int(map_id)
        agent_scores, new_max = compute_agent_max_mc(rewards, values, dones, self.max_returns[map_id])
        self.max_returns[map_id] = new_max
        reduced = aggregate_agent_scores(agent_scores, self.agent_aggregate)
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
        reduced = aggregate_agent_scores(compute_agent_pvl(advantages, dones), self.agent_aggregate)
        self.update(map_id, reduced)
        return float(self.scores[map_id])

    def score_pvl_rollout(self, map_id, rewards, values, dones, gamma=None, gae_lambda=None):
        """Build GAE from a stored (r, V, done) rollout, then ``score_pvl``.

        Drive records those arrays the same way as MaxMC. PPO advantages are
        not available at resample time, so GAE is computed here with
        ``gae_gamma`` / ``gae_lambda``.
        """
        advantages = compute_gae(
            rewards,
            values,
            dones,
            gamma=self.gae_gamma if gamma is None else gamma,
            gae_lambda=self.gae_lambda if gae_lambda is None else gae_lambda,
        )
        return self.score_pvl(map_id, advantages, dones)

    def score_learnability(self, map_id, successes, dones=None):
        """Score one map with goal-success learnability and store it in ``scores[map_id]``.

        successes: (T, A) 1 on steps where that agent reached its goal, else 0.
            An episode succeeds if it hit the goal at least once.
        dones: (T, A) optional. 1 where that agent ends an episode.
        The map score is p(1-p) where p is the fraction of episodes (pooled
        across agents on this map) that reached the goal. Peaks at 0.25 when
        half the episodes succeed.
        """
        map_id = int(map_id)
        reduced = compute_learnability(successes, dones)
        self.update(map_id, reduced)
        return float(self.scores[map_id])

    def score(self, map_id, score_function, rewards=None, values=None, dones=None, advantages=None, successes=None):
        """Dispatch to MaxMC, PVL, or learnability and write ``scores[map_id]``.

        Mirrors Kinetix ``compute_score``. ``advantages`` is optional for PVL;
        if omitted, GAE is built from ``rewards`` / ``values`` / ``dones``.
        Learnability uses ``successes`` (goal reached) and ``dones``.
        """
        name = _normalize_score_function(score_function)
        if name == "max_mc":
            if self.score_transform != "rank":
                raise ValueError(
                    "MaxMC scores can be negative, so score_transform must be 'rank'. "
                    f"Got: {self.score_transform!r}"
                )
            return self.score_max_mc(map_id, rewards, values, dones)
        if name == "pvl":
            if advantages is None:
                return self.score_pvl_rollout(map_id, rewards, values, dones)
            return self.score_pvl(map_id, advantages, dones)
        if name == "learnability":
            return self.score_learnability(map_id, successes, dones)
        raise ValueError(f"Unknown score function: {score_function!r}")


def merge_sampler_states(states):
    states = list(states)
    if not states:
        return None
    num_maps = int(np.asarray(states[0][0]).shape[0])
    score_sum = np.zeros(num_maps, dtype=np.float64)
    seen_count = np.zeros(num_maps, dtype=np.float64)
    for scores, seen in states:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        seen = np.asarray(seen, dtype=bool).reshape(-1)
        if scores.shape[0] != num_maps or seen.shape[0] != num_maps:
            raise ValueError("all sampler states must have the same num_maps")
        score_sum += np.where(seen, scores, 0.0)
        seen_count += seen.astype(np.float64)
    merged_seen = seen_count > 0
    merged_scores = np.zeros(num_maps, dtype=np.float64)
    np.divide(score_sum, seen_count, out=merged_scores, where=merged_seen)
    return merged_scores, merged_seen


def score_replay_weights(scores, seen, score_transform="rank", temperature=2.0, eps=1e-3):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    seen = np.asarray(seen, dtype=bool).reshape(-1)
    sampler = LevelSampler(
        num_maps=int(scores.shape[0]),
        score_transform=score_transform,
        temperature=temperature,
        staleness_coef=0.0,
        eps=eps,
    )
    sampler.scores = np.array(scores, copy=True)
    sampler.seen = np.array(seen, copy=True)
    return sampler._sample_weights()


def _normalize_score_function(score_function):
    name = str(score_function).strip().lower().replace("-", "_")
    if name in ("maxmc", "max_mc"):
        return "max_mc"
    if name in ("pvl", "positive_value_loss"):
        return "pvl"
    if name in ("return", "learnability"):
        return name
    return name


def aggregate_agent_scores(agent_scores, agent_aggregate="mean"):
    """Collapse one score per agent into the score for their map."""
    agent_scores = np.asarray(agent_scores, dtype=np.float64).reshape(-1)
    if agent_scores.size == 0:
        return 0.0
    if agent_aggregate == "mean":
        return float(agent_scores.mean())
    if agent_aggregate == "max":
        return float(agent_scores.max())
    if agent_aggregate == "min":
        return float(agent_scores.min())
    raise ValueError(f"agent aggregate must be 'mean', 'max', or 'min', got {agent_aggregate!r}")


def compute_agent_max_mc(rewards, values, dones, max_return):
    """Per-agent MaxMC. Returns (agent_scores, updated_max_return)."""
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
        return np.zeros(0, dtype=np.float64), float(max_return)

    timesteps, agents = rewards.shape
    partial_return = np.zeros(agents, dtype=np.float64)
    max_ep_return = np.full(agents, -np.inf, dtype=np.float64)
    for t in range(timesteps):
        partial_return += rewards[t]
        done = dones[t] > 0
        max_ep_return = np.where(done, np.maximum(max_ep_return, partial_return), max_ep_return)
        partial_return = np.where(done, 0.0, partial_return)
    # Unfinished tail still counts as an episode return.
    max_ep_return = np.where(np.isfinite(max_ep_return), max_ep_return, partial_return)
    level_max = max(float(max_return), float(max_ep_return.max()))

    partial_gap = np.zeros(agents, dtype=np.float64)
    partial_steps = np.zeros(agents, dtype=np.float64)
    sum_ep_gap = np.zeros(agents, dtype=np.float64)
    n_episodes = np.zeros(agents, dtype=np.float64)
    for t in range(timesteps):
        partial_gap += level_max - values[t]
        partial_steps += 1.0
        done = dones[t] > 0
        sum_ep_gap += np.where(done, partial_gap / partial_steps, 0.0)
        n_episodes += done
        partial_gap = np.where(done, 0.0, partial_gap)
        partial_steps = np.where(done, 0.0, partial_steps)
    open_tail = partial_steps > 0
    tail_gap = np.divide(partial_gap, partial_steps, out=np.zeros_like(partial_gap), where=open_tail)
    sum_ep_gap = np.where(open_tail, sum_ep_gap + tail_gap, sum_ep_gap)
    n_episodes = np.where(open_tail, n_episodes + 1.0, n_episodes)
    n_episodes = np.maximum(n_episodes, 1.0)
    return sum_ep_gap / n_episodes, level_max


def compute_max_mc(rewards, values, dones, max_return, agent_aggregate="mean"):
    """Agent MaxMC scores reduced to one map score.

    Returns (map_score, updated_max_return). ``agent_aggregate`` selects how
    the per-agent scores are combined.
    """
    agent_scores, level_max = compute_agent_max_mc(rewards, values, dones, max_return)
    return aggregate_agent_scores(agent_scores, agent_aggregate), level_max


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """GAE(lambda) over a stored rollout. Same layout as MaxMC: (T, A) or (T,).

    values[t] is V(s_t) and rewards[t] is r_t, matching Drive's MaxMC buffers.
    The last step has no V(s_{t+1}); if that step is not done, V(s_t) is used
    as the bootstrap (same truncation heuristic as pufferl). Done steps cut
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
    advantages = np.zeros((timesteps, agents), dtype=np.float64)
    lastgaelam = np.zeros(agents, dtype=np.float64)
    gamma = float(gamma)
    lam = float(gae_lambda)
    for t in range(timesteps - 1, -1, -1):
        nonterminal = 1.0 - (dones[t] > 0).astype(np.float64)
        next_v = values[t + 1] if t + 1 < timesteps else values[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        advantages[t] = lastgaelam
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
    partial_sum = np.zeros(agents, dtype=np.float64)
    partial_steps = np.zeros(agents, dtype=np.float64)
    sum_ep = np.zeros(agents, dtype=np.float64)
    n_episodes = np.zeros(agents, dtype=np.float64)
    for t in range(timesteps):
        partial_sum += pos[t]
        partial_steps += 1.0
        done = dones[t] > 0
        sum_ep += np.where(done, partial_sum / partial_steps, 0.0)
        n_episodes += done
        partial_sum = np.where(done, 0.0, partial_sum)
        partial_steps = np.where(done, 0.0, partial_steps)
    open_tail = partial_steps > 0
    tail = np.divide(partial_sum, partial_steps, out=np.zeros_like(partial_sum), where=open_tail)
    sum_ep = np.where(open_tail, sum_ep + tail, sum_ep)
    n_episodes = np.where(open_tail, n_episodes + 1.0, n_episodes)
    n_episodes = np.maximum(n_episodes, 1.0)
    return sum_ep / n_episodes


def compute_pvl(advantages, dones, agent_aggregate="mean"):
    """Agent PVL scores reduced to one map score."""
    return aggregate_agent_scores(compute_agent_pvl(advantages, dones), agent_aggregate)


def compute_learnability(successes, dones=None):
    """Map learnability p(1-p) from binary goal success.

    successes: (T, A) flag that the agent reached its goal at that step, or a
        1d array of already-finished episode outcomes.
    dones: (T, A) optional episode ends. An episode succeeds if it saw a goal
        hit at least once. Unfinished tails count, same as MaxMC.
    Episodes are pooled across agents on the map. Per-agent p_i(1-p_i) would
        collapse to 0 whenever each agent only finishes one episode.
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
    if successes.shape != dones.shape:
        raise ValueError("successes and dones must have the same shape")

    timesteps, agents = successes.shape
    partial_hit = np.zeros(agents, dtype=np.float64)
    partial_steps = np.zeros(agents, dtype=np.float64)
    n_success = 0.0
    n_episodes = 0.0
    for t in range(timesteps):
        partial_hit = np.maximum(partial_hit, (successes[t] > 0).astype(np.float64))
        partial_steps += 1.0
        done = dones[t] > 0
        n_success += float(np.where(done, partial_hit, 0.0).sum())
        n_episodes += float(done.sum())
        partial_hit = np.where(done, 0.0, partial_hit)
        partial_steps = np.where(done, 0.0, partial_steps)
    open_tail = partial_steps > 0
    n_success += float(np.where(open_tail, partial_hit, 0.0).sum())
    n_episodes += float(open_tail.sum())
    n_episodes = max(n_episodes, 1.0)
    p = n_success / n_episodes
    return float(p * (1.0 - p))


def compute_score(
    score_function,
    *,
    dones,
    values=None,
    max_return=0.0,
    rewards=None,
    advantages=None,
    successes=None,
    agent_aggregate="mean",
    gamma=0.99,
    gae_lambda=0.95,
):
    """Pick MaxMC, PVL, or learnability. Returns (map_score, extra) where extra
    is the updated max return for MaxMC and NaN otherwise.

    Mirrors Kinetix ``compute_score``. PVL uses ``advantages`` when given,
    otherwise GAE from ``rewards`` / ``values`` / ``dones``. Learnability uses
    ``successes`` (goal reached) and ``dones``.
    """
    name = _normalize_score_function(score_function)
    if name == "max_mc":
        return compute_max_mc(rewards, values, dones, max_return, agent_aggregate)
    if name == "pvl":
        if advantages is None:
            advantages = compute_gae(rewards, values, dones, gamma, gae_lambda)
        return compute_pvl(advantages, dones, agent_aggregate), float("nan")
    if name == "learnability":
        return compute_learnability(successes, dones), float("nan")
    raise ValueError(f"Unknown score function: {score_function!r}")
