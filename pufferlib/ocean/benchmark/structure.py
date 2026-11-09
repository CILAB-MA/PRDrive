from collections import Counter

import numpy as np

MANY_FAIL_RETURN = 0.0
STRUCTURE_CATEGORIES = ("solved", "localized_failure", "globally_suboptimal", "many_fail")
MAX_LOCALIZED_CARS = 1
# A car still fails if its training fail EMA is above this after enough episodes.
FAIL_CAR_EMA = 0.05
# Training-time (map, agent) fail EMA: one car owns this much of the fail mass.
FAIL_STRUCTURAL_TOP1 = 0.8
# Worst-car collision EMA at or above this → the leftover fail is a crash, else DNF.
FAIL_COLLISION_EMA = 0.5
# Mean collision/off-road steps on completed episodes for the worst car.
FAIL_MULTI_HIT = 2.0
MIN_OUTCOME_EPISODES = 2
# Ties when pooling seeds: catastrophe > spread > one-car > solved.
_STRUCTURE_SEVERITY = {
    "many_fail": 0,
    "globally_suboptimal": 1,
    "localized_failure": 2,
    "solved": 3,
}


def failed_cars(log, score_key="score"):
    n = float(log.get("n") or 0)
    if n <= 0 or score_key not in log:
        return 0.0
    return max((1.0 - float(log[score_key])) * n, 0.0)


def classify_structure(
    log,
    max_localized_cars=MAX_LOCALIZED_CARS,
    score_key="score",
    return_key="episode_return",
    many_fail_return=MANY_FAIL_RETURN,
    fail_ema=None,
    outcome_n=None,
):
    """Label a map. Prefer training fail EMA over a single eval rollout."""
    if fail_ema is not None and outcome_n is not None:
        n_valid = 0
        failed_n = 0
        for fail, visits in zip(fail_ema, outcome_n):
            if float(visits) < float(MIN_OUTCOME_EPISODES):
                continue
            n_valid += 1
            if float(fail) > float(FAIL_CAR_EMA):
                failed_n += 1
        if n_valid:
            if failed_n <= 0:
                return "solved"
            if failed_n <= int(max_localized_cars):
                return "localized_failure"
            return "globally_suboptimal"
    n = float(log.get("n") or 0)
    if n <= 0 or score_key not in log:
        return "solved"
    failed_n = int(round(failed_cars(log, score_key=score_key)))
    if failed_n <= 0:
        return "solved"
    ep_ret = log.get(return_key)
    try:
        if ep_ret is not None and float(ep_ret) < float(many_fail_return):
            return "many_fail"
    except (TypeError, ValueError):
        pass
    if failed_n <= int(max_localized_cars):
        return "localized_failure"
    return "globally_suboptimal"


def episode_live_mask(dones):
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


def fail_top1_share(fail_ema, outcome_n, min_episodes=MIN_OUTCOME_EPISODES):
    """Fraction of fail-mass on the worst vehicle. None if no slot has enough visits."""
    fail = np.asarray(fail_ema, dtype=np.float64).reshape(-1)
    visits = np.asarray(outcome_n, dtype=np.float64).reshape(-1)
    valid = visits >= float(min_episodes)
    if not valid.any():
        return None
    mass = fail[valid]
    total = float(mass.sum())
    if total <= 0:
        return 0.0
    return float(mass.max() / total)


def fail_worst_stat(fail_ema, stat_ema, outcome_n, min_episodes=MIN_OUTCOME_EPISODES):
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


class FailTracker:
    """Per-(map, vehicle) fail / collision EMA used to label solved / LF / GS."""

    def __init__(self, num_maps, max_controlled_agents=32, alpha=0.2):
        self.num_maps = int(num_maps)
        self.alpha = float(alpha)
        n_agents = int(max_controlled_agents)
        self.fail_ema = np.zeros((self.num_maps, n_agents), dtype=np.float64)
        self.coll_ema = np.zeros((self.num_maps, n_agents), dtype=np.float64)
        self.hit_ema = np.zeros((self.num_maps, n_agents), dtype=np.float64)
        self.outcome_n = np.zeros((self.num_maps, n_agents), dtype=np.float64)

    def state_dict(self):
        return {
            "fail_ema": self.fail_ema,
            "coll_ema": self.coll_ema,
            "hit_ema": self.hit_ema,
            "outcome_n": self.outcome_n,
        }

    def update(self, map_id, successes=None, dones=None, unsafe=None, rewards=None):
        if dones is None:
            return
        if successes is None:
            if rewards is None:
                return
            rewards = np.asarray(rewards)
            successes = (rewards > 0).astype(np.float64)
            if unsafe is None:
                unsafe = (rewards < 0).astype(np.float64)
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
            step_unsafe = (
                np.zeros(agents, dtype=np.float64)
                if unsafe is None
                else (unsafe[t] > 0).astype(np.float64)
            )
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

        map_id = int(map_id)
        n_store = min(int(fail_rate.shape[0]), self.fail_ema.shape[1])
        alpha = self.alpha
        for agent in range(n_store):
            visits = float(n_episodes[agent])
            if visits <= 0:
                continue
            if self.outcome_n[map_id, agent] <= 0:
                self.fail_ema[map_id, agent] = float(fail_rate[agent])
                self.coll_ema[map_id, agent] = float(coll_rate[agent])
                self.hit_ema[map_id, agent] = float(hit_mean[agent])
            else:
                self.fail_ema[map_id, agent] = (
                    (1.0 - alpha) * self.fail_ema[map_id, agent] + alpha * float(fail_rate[agent])
                )
                self.coll_ema[map_id, agent] = (
                    (1.0 - alpha) * self.coll_ema[map_id, agent] + alpha * float(coll_rate[agent])
                )
                self.hit_ema[map_id, agent] = (
                    (1.0 - alpha) * self.hit_ema[map_id, agent] + alpha * float(hit_mean[agent])
                )
            self.outcome_n[map_id, agent] += visits


def majority_structure(categories):
    cats = [c for c in categories if c]
    if not cats:
        return "solved"
    counts = Counter(cats)
    top = max(counts.values())
    tied = [c for c, n in counts.items() if n == top]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda c: _STRUCTURE_SEVERITY.get(c, 99))
