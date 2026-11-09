
import numpy as np

from pufferlib.ocean.benchmark.structure import episode_live_mask


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
        gae_gamma=0.99,  # discount used by MaxMC
        max_controlled_agents=32,  # per-vehicle MaxMC R_max slots, matching Drive max_controlled_agents
        invert_scores=False,  # if True, rank low stored scores first (episode return)
    ):
        self.num_maps = int(num_maps)
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
        self.gae_gamma = float(gae_gamma)
        self.max_controlled_agents = int(max_controlled_agents)
        self.invert_scores = bool(invert_scores)
        self._rng = np.random.default_rng(seed)

        self.scores = np.zeros(self.num_maps, dtype=np.float64)
        self.staleness = np.zeros(self.num_maps, dtype=np.float64)
        self.seen = np.zeros(self.num_maps, dtype=bool)

        # C caps controllable agents at MAX_AGENTS=32; training npy max is also 32.
        self.max_returns = np.zeros((self.num_maps, self.max_controlled_agents), dtype=np.float64)
        self.max_return_valid = np.zeros((self.num_maps, self.max_controlled_agents), dtype=bool)
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

    def sample_weights(self):
        score_values = np.array(self.scores, copy=True)
        if self.invert_scores:
            score_values = -score_values
        if self.score_transform == "rank":
            score_values[~self.seen] = -np.inf
        weights = self._transform(self.score_transform, self.temperature, score_values)
        weights = weights * self.seen
        total = weights.sum()
        if total > 0:
            weights = weights / total
        if self.staleness_coef > 0:
            staleness_weights = self._transform(
                self.staleness_transform, self.staleness_temperature, self.staleness
            )
            staleness_weights = staleness_weights * self.seen
            s_total = staleness_weights.sum()
            if s_total > 0:
                staleness_weights = staleness_weights / s_total
            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * staleness_weights
        return weights

    def sample(self):
        if self.strategy == "sequential":
            map_id = int(self._next_index % self.num_maps)
            self._next_index += 1
        elif self.strategy == "uniform":
            map_id = int(self._rng.integers(self.num_maps))
        else:
            num_unseen = int((~self.seen).sum())
            proportion_seen = (self.num_maps - num_unseen) / self.num_maps
            if proportion_seen >= self.rho and (
                self._rng.random() > self.nu or proportion_seen >= 1.0
            ):
                weights = np.maximum(self.sample_weights(), 0.0)
                total = float(weights.sum())
                if total > 0 and np.isfinite(total):
                    map_id = int(self._rng.choice(self.num_maps, p=weights / total))
                else:
                    map_id = int(self._rng.integers(self.num_maps))
            else:
                map_id = int(self._rng.choice(np.flatnonzero(~self.seen)))
        self.staleness += 1
        self.staleness[map_id] = 0
        return map_id

    def sample_queue(self, agent_counts, agent_budget):
        # Sample whole maps
        agent_counts = np.asarray(agent_counts)
        budget = int(agent_budget)
        chosen = []
        total = 0
        for _ in range(self.num_maps):
            map_id = self.sample()
            chosen.append(map_id)
            total += int(agent_counts[map_id])
            if total >= budget:
                return np.asarray(chosen, dtype=np.int32)

    # ---- feeding results back in --------------------------------------

    def update(self, map_id, score):
        """Record one map or a batch of (map_id, score) pairs after they were played."""
        map_ids = np.asarray(map_id, dtype=np.int64).reshape(-1)
        scores = np.asarray(score, dtype=np.float64).reshape(-1)
        if map_ids.shape[0] != scores.shape[0]:
            raise ValueError("map_id and score must have the same length")
        for mid, sc in zip(map_ids, scores):
            mid = int(mid)
            if not (0 <= mid < self.num_maps):
                raise ValueError(f"map_id {mid} out of range [0, {self.num_maps})")
            self.seen[mid] = True
            self.scores[mid] = (1 - self.alpha) * self.scores[mid] + self.alpha * float(sc)

    def score(
        self,
        map_id,
        score_function,
        rewards=None,
        values=None,
        dones=None,
        successes=None,
        unsafe=None,
    ):
        name = _score_key(score_function)
        map_id = int(map_id)
        if name == "max_mc": # should be rank score transform
            rewards = np.asarray(rewards)
            n_agents = 1 if rewards.ndim < 2 else int(rewards.shape[1])
            n_store = min(n_agents, self.max_returns.shape[1])
            prev = np.full(n_agents, -np.inf, dtype=np.float64)
            valid = self.max_return_valid[map_id, :n_store]
            prev[:n_store] = np.where(valid, self.max_returns[map_id, :n_store], -np.inf)
            agent_scores, new_max = compute_max_mc(
                rewards, values, dones, prev, gamma=self.gae_gamma
            )
            new_max = np.asarray(new_max, dtype=np.float64).reshape(-1)
            self.max_returns[map_id, :n_store] = new_max[:n_store]
            self.max_return_valid[map_id, :n_store] = True
            stored = np.asarray(agent_scores, dtype=np.float64).reshape(-1)[:n_store]
            if stored.size == 0:
                reduced = 0.0
            elif self.score_aggregate == "top":
                if self.invert_scores:
                    reduced = float(stored.min())
                else:
                    reduced = float(stored.max())
            elif self.score_aggregate == "mean":
                reduced = float(stored.mean())
            elif self.score_aggregate == "cvar":
                alpha = 0.1
                k = max(1, int(np.ceil(alpha * stored.size)))
                ordered = np.sort(stored)
                # Hard tail: MaxMC high=hard → upper; return low=hard → lower.
                if self.invert_scores:
                    reduced = float(ordered[:k].mean())
                else:
                    reduced = float(ordered[-k:].mean())

            self.update(map_id, reduced)
            return float(self.scores[map_id])
        elif name == "learnability":
            self.update(map_id, compute_learnability(successes, dones))
            return float(self.scores[map_id])
        elif name == "learnability_hard":
            self.update(map_id, compute_learnability(successes, dones, unsafe=unsafe))
            return float(self.scores[map_id])
        raise ValueError(f"Unknown score function: {score_function!r}")


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
    return sampler.sample_weights()


# CL4AD 10K: β is rank temperature, ρ is staleness coefficient.
# Appendix E.3: MaxMC 10K uses β=4; 1K/80K use β=2.
PLR_SCORE_HYPERS = {
    "max_mc": (4.0, 0.3),
    "learnability": (4.0, 0.1),
    "learnability_hard": (2.0, 0.1),
    "return": (2.0, 0.3),
}


def _score_key(score_function):
    name = str(score_function).strip().lower().replace("-", "_")
    if name in ("maxmc", "max_mc"):
        return "max_mc"
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


def compute_max_mc(rewards, values, dones, max_return, gamma=0.99):
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
    prev = np.asarray(max_return, dtype=np.float64).reshape(-1)
    if prev.shape[0] != agents:
        padded = np.full(agents, -np.inf, dtype=np.float64)
        n = min(agents, prev.shape[0])
        padded[:n] = prev[:n]
        prev = padded
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
    max_ep_return = np.where(np.isfinite(max_ep_return), max_ep_return, partial_return)
    agent_max = np.maximum(prev, max_ep_return)

    partial_gap = np.zeros(agents, dtype=np.float64)
    partial_steps = np.zeros(agents, dtype=np.float64)
    sum_ep_gap = np.zeros(agents, dtype=np.float64)
    n_episodes = np.zeros(agents, dtype=np.float64)
    for t in range(timesteps):
        partial_gap = np.where(live[t], partial_gap + (agent_max - values[t]), partial_gap)
        partial_steps = np.where(live[t], partial_steps + 1.0, partial_steps)
        sum_ep_gap = np.where(
            finish[t], sum_ep_gap + partial_gap / np.maximum(partial_steps, 1.0), sum_ep_gap
        )
        n_episodes = np.where(finish[t], n_episodes + 1.0, n_episodes)
        partial_gap = np.where(finish[t], 0.0, partial_gap)
        partial_steps = np.where(finish[t], 0.0, partial_steps)
    open_tail = partial_steps > 0
    tail_gap = np.divide(partial_gap, partial_steps, out=np.zeros_like(partial_gap), where=open_tail)
    sum_ep_gap = np.where(open_tail, sum_ep_gap + tail_gap, sum_ep_gap)
    n_episodes = np.where(open_tail, n_episodes + 1.0, n_episodes)
    n_episodes = np.maximum(n_episodes, 1.0)
    return sum_ep_gap / n_episodes, agent_max


def compute_learnability(successes, dones=None, unsafe=None):
    successes = np.asarray(successes, dtype=np.float64)
    if successes.size == 0:
        return 0.0
    if dones is None:
        p = float((successes.reshape(-1) > 0).astype(np.float64).mean())
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
