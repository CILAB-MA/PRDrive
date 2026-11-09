"""Standalone sanity checks for LevelSampler (no C extension required)."""

import numpy as np

from pufferlib.ocean.drive.level_sampler import (
    LevelSampler,
    merge_sampler_states,
    plr_hypers_for_score,
    score_replay_weights,
)
from pufferlib.ocean.benchmark.structure import FailTracker, fail_top1_share, fail_worst_stat


def test_batch_shape_and_range():
    ls = LevelSampler(num_maps=10, seed=0)
    batch = np.array([ls.sample() for _ in range(37)], dtype=np.int32)
    assert batch.shape == (37,)
    assert batch.dtype == np.int32
    assert batch.min() >= 0 and batch.max() < 10


def test_explores_before_replaying():
    # rho=1.0 forces "see everything at least once" before any replay is possible.
    ls = LevelSampler(num_maps=20, rho=1.0, seed=1)
    seen_order = []
    for _ in range(20):
        map_id = ls.sample()
        ls.update(map_id, score=0.0)
        seen_order.append(map_id)
    assert ls.seen.all(), "expected every map to be seen after num_maps unseen-forced draws"
    assert len(set(seen_order)) == 20, "expected no repeats while still exploring unseen maps"


def test_high_score_maps_get_sampled_more():
    ls = LevelSampler(
        num_maps=5, rho=0.0, nu=1.0, staleness_coef=0.0, alpha=1.0, seed=2,
        score_transform="power", temperature=1.0,
    )
    # Seed every map once so all are "seen" and eligible for replay weighting.
    for map_id in range(5):
        ls.update(map_id, score=0.0)
    # Now make map 3 look much more valuable to train on than the rest.
    ls.update(3, score=100.0)

    counts = np.zeros(5, dtype=int)
    for _ in range(2000):
        map_id = ls.sample()
        counts[map_id] += 1
        # Immediately re-report the same low/high scores so the distribution stays stable.
        ls.update(map_id, score=100.0 if map_id == 3 else 0.0)

    assert counts[3] > counts.sum() * 0.5, f"expected map 3 to dominate sampling, got counts={counts}"


def test_negative_max_mc_scores_stay_valid_probabilities():
    ls = LevelSampler(
        num_maps=4, rho=0.0, nu=1.0, staleness_coef=0.1, alpha=1.0, seed=5, score_transform="rank"
    )
    for map_id, score in enumerate([-30.0, -10.0, -2.0, -1.0]):
        ls.update(map_id, score=score)
    counts = np.zeros(4, dtype=int)
    for _ in range(400):
        counts[ls.sample()] += 1
    assert counts.sum() == 400
    assert counts[3] > counts[0]


def test_staleness_favors_untouched_maps():
    ls = LevelSampler(num_maps=4, rho=0.0, nu=1.0, staleness_coef=1.0, alpha=1.0, seed=3)
    for map_id in range(4):
        ls.update(map_id, score=1.0)  # identical scores -> pure staleness should drive selection

    # Manually make map 2 "fresh" (staleness 0) and everything else very stale.
    ls.staleness[:] = 50.0
    ls.staleness[2] = 0.0

    counts = np.zeros(4, dtype=int)
    for _ in range(200):
        map_id = ls.sample()
        counts[map_id] += 1
        ls.staleness[map_id] = 0.0
        # keep the others artificially stale so the test isolates the staleness effect
        for other in range(4):
            if other != map_id:
                ls.staleness[other] += 1

    assert counts[2] < counts.sum() * 0.4, f"expected the already-fresh map to be picked least often, got counts={counts}"


def test_uniform_strategy_covers_all_maps():
    ls = LevelSampler(num_maps=6, strategy="uniform", seed=4)
    seen = set()
    for _ in range(500):
        seen.add(ls.sample())
    assert seen == set(range(6))


def test_sequential_strategy_is_in_order():
    ls = LevelSampler(num_maps=4, strategy="sequential", seed=0)
    ids = [ls.sample() for _ in range(6)]
    assert ids == [0, 1, 2, 3, 0, 1]


def test_sequential_queue_stops_when_agent_budget_is_met():
    # Counts from the validation bins: map 5 is 12, not the budget-clamped 5.
    counts = np.array([7, 2, 5, 18, 3, 12, 1, 13], dtype=np.int32)
    ls = LevelSampler(num_maps=8, strategy="sequential", seed=0)
    first = ls.sample_queue(counts, 40)
    # 7+2+5+18+3+12 = 47 >= 40, so maps 0..5 are opened and 6 is next.
    assert first.tolist() == [0, 1, 2, 3, 4, 5]
    assert int(counts[first].sum()) == 47
    second = ls.sample_queue(counts, 40)
    # 1+13+7+2+5+18 = 46 >= 40, continuing at map 6 and wrapping to 0.
    assert second.tolist() == [6, 7, 0, 1, 2, 3]


def test_max_mc_uses_best_return_minus_value():
    rewards = np.array([[1.0], [1.0]])
    values = np.array([[0.5], [0.5]])
    dones = np.array([[0.0], [1.0]])
    ls = LevelSampler(num_maps=3, strategy="sequential", alpha=1.0, gae_gamma=1.0)
    stored = ls.score(0, "max_mc", rewards=rewards, values=values, dones=dones)
    assert np.isclose(stored, 1.5)
    assert np.allclose(ls.max_returns[0, :1], [2.0])

    ls.max_returns[1, 0] = 10.0
    ls.max_return_valid[1, 0] = True
    stored = ls.score(1, "max_mc", rewards=rewards, values=values, dones=dones)
    assert np.isclose(stored, 9.5)
    assert np.allclose(ls.max_returns[1, :1], [10.0])

    disc = LevelSampler(num_maps=1, strategy="sequential", alpha=1.0, gae_gamma=0.99)
    stored = disc.score(0, "max_mc", rewards=rewards, values=values, dones=dones)
    assert np.isclose(stored, 1.49)
    assert np.allclose(disc.max_returns[0, :1], [1.99])

    # Two agents on one map. Each vehicle keeps its own R_max (2 and 0),
    # not a shared scenario max of 2.
    rewards = np.array([[1.0, 0.0], [1.0, 0.0]])
    values = np.array([[0.0, 1.0], [0.0, 1.0]])
    dones = np.array([[0.0, 0.0], [1.0, 1.0]])
    two = LevelSampler(
        num_maps=1, strategy="sequential", score_aggregate="mean", alpha=1.0, gae_gamma=1.0
    )
    stored = two.score(0, "max_mc", rewards=rewards, values=values, dones=dones)
    assert stored == 0.5
    assert np.allclose(two.max_returns[0, :2], [2.0, 0.0])
    top = LevelSampler(
        num_maps=1, strategy="sequential", score_aggregate="top", alpha=1.0, gae_gamma=1.0
    )
    stored = top.score(0, "max_mc", rewards=rewards, values=values, dones=dones)
    assert stored == 2.0
    low = LevelSampler(
        num_maps=1,
        strategy="sequential",
        score_aggregate="top",
        invert_scores=True,
        alpha=1.0,
        gae_gamma=1.0,
    )
    stored = low.score(0, "max_mc", rewards=rewards, values=values, dones=dones)
    assert stored == -1.0


def test_max_mc_keeps_per_vehicle_max_return_across_visits():
    ls = LevelSampler(
        num_maps=2, strategy="sequential", score_aggregate="mean", alpha=1.0, gae_gamma=1.0
    )
    rewards = np.array([[1.0, 0.0], [1.0, 0.0]])
    values = np.array([[0.0, 1.0], [0.0, 1.0]])
    dones = np.array([[0.0, 0.0], [1.0, 1.0]])
    ls.score(0, "max_mc", rewards=rewards, values=values, dones=dones)

    later_rewards = np.zeros((2, 2))
    later_values = np.zeros((2, 2))
    later_dones = np.array([[0.0, 0.0], [1.0, 1.0]])
    stored = ls.score(0, "max_mc", rewards=later_rewards, values=later_values, dones=later_dones)
    assert np.allclose(ls.max_returns[0, :2], [2.0, 0.0])
    assert np.isclose(stored, 1.0)


def test_max_mc_mask_ignores_unseen_padding():
    ls = LevelSampler(
        num_maps=1,
        strategy="sequential",
        score_aggregate="mean",
        alpha=1.0,
        max_controlled_agents=4,
        gae_gamma=1.0,
    )
    rewards = np.array([[-1.0], [-1.0]])
    values = np.zeros((2, 1))
    dones = np.array([[0.0], [1.0]])
    ls.score(0, "max_mc", rewards=rewards, values=values, dones=dones)
    assert np.isclose(ls.max_returns[0, 0], -2.0)
    assert ls.max_return_valid[0, 0]
    assert not ls.max_return_valid[0, 1:].any()
    assert np.allclose(ls.max_returns[0, 1:], 0.0)

    wider = np.array([[-1.0, -3.0], [-1.0, -3.0]])
    wider_values = np.zeros((2, 2))
    wider_dones = np.array([[0.0, 0.0], [1.0, 1.0]])
    ls.score(0, "max_mc", rewards=wider, values=wider_values, dones=wider_dones)
    assert np.allclose(ls.max_returns[0, :2], [-2.0, -6.0])
    assert ls.max_return_valid[0, :2].all()
    assert not ls.max_return_valid[0, 2:].any()


def test_repeated_terminals_after_goal_are_not_new_episodes():
    # Goal at t=1, then Drive keeps terminals=1 until the map ends.
    rewards = np.array([[1.0], [1.0], [0.0], [0.0]])
    values = np.array([[0.0], [0.0], [0.0], [0.0]])
    dones = np.array([[0.0], [1.0], [1.0], [1.0]])
    ls = LevelSampler(num_maps=2, strategy="sequential", alpha=1.0, gae_gamma=1.0)
    stored = ls.score(0, "max_mc", rewards=rewards, values=values, dones=dones)
    assert np.isclose(stored, 2.0)
    assert np.allclose(ls.max_returns[0, :1], [2.0])

    successes = np.array([[1.0], [0.0], [0.0], [0.0]])
    stored = ls.score(1, "learnability", successes=successes, dones=dones)
    assert np.isclose(stored, 0.0)


def test_score_rejects_unknown():
    ls = LevelSampler(num_maps=1, strategy="sequential", alpha=1.0, gae_gamma=1.0)
    rewards = np.array([[1.0], [1.0]])
    values = np.array([[0.0], [0.0]])
    dones = np.array([[0.0], [1.0]])
    stored = ls.score(0, "MaxMC", rewards=rewards, values=values, dones=dones)
    assert np.isclose(stored, 2.0)
    assert np.allclose(ls.max_returns[0, :1], [2.0])

    try:
        ls.score(0, "not_a_score")
    except ValueError as err:
        assert "Unknown score function" in str(err)
    else:
        raise AssertionError("expected unknown score function to be rejected")


def test_learnability_is_p_times_one_minus_p_from_goal_success():
    ls = LevelSampler(num_maps=8, strategy="sequential", alpha=1.0)
    # 1d array is one agent's episode outcomes. Mix of success and fail -> 0.25.
    assert np.isclose(ls.score(0, "learnability", successes=np.array([0.0, 0.0])), 0.0)
    assert np.isclose(ls.score(1, "learnability", successes=np.array([1.0, 1.0])), 0.0)
    assert np.isclose(ls.score(2, "learnability", successes=np.array([1.0, 0.0])), 0.25)

    # Two agents, one episode each: agent 0 always succeeds, agent 1 always fails.
    # Pooled p(1-p) would be 0.25; per-vehicle mean is 0.
    successes = np.array([[1.0, 0.0], [0.0, 0.0]])
    dones = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert np.isclose(ls.score(3, "learnability", successes=successes, dones=dones), 0.0)

    # Unfinished tail that already hit the goal is still a successful episode (p=1).
    successes = np.array([[1.0], [0.0]])
    dones = np.array([[0.0], [0.0]])
    assert np.isclose(ls.score(4, "learnability", successes=successes, dones=dones), 0.0)

    # One unfinished hit and one unfinished miss: each vehicle is certain, mean 0.
    successes = np.array([[1.0, 0.0], [0.0, 0.0]])
    dones = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert np.isclose(ls.score(5, "learnability", successes=successes, dones=dones), 0.0)

    # Same vehicle succeeding on one episode and failing on the next -> 0.25.
    successes = np.array([[1.0], [0.0], [0.0], [0.0]])
    dones = np.array([[0.0], [1.0], [0.0], [1.0]])
    stored = ls.score(6, "learnability", successes=successes, dones=dones)
    assert np.isclose(stored, 0.25)
    assert np.isclose(ls.scores[6], 0.25)


def test_learnability_hard_requires_goal_without_collision():
    # Two episodes: safe goal, then goal after a collision. Learn p=1 (U=0); hard p=0.5.
    successes = np.array([[1.0], [0.0], [1.0], [0.0]])
    dones = np.array([[0.0], [1.0], [0.0], [1.0]])
    unsafe = np.array([[0.0], [0.0], [1.0], [0.0]])
    ls = LevelSampler(num_maps=2, strategy="sequential", alpha=1.0)
    assert np.isclose(ls.score(0, "learnability", successes=successes, dones=dones), 0.0)
    stored = ls.score(1, "learnability_hard", successes=successes, dones=dones, unsafe=unsafe)
    assert np.isclose(stored, 0.25)


def _map_bin_name(map_id):
    return f"map_{int(map_id):03d}.bin"


def test_c_simulator_reads_plr_queue_in_order():
    """Print how binding.shared walks a PLR queue and which bin files it opens.

    C keeps loading maps from the front of the queue until num_agents controllable
    agents are collected, so the returned map ids are a prefix of the queue
    (empty maps would be skipped). Requires the compiled drive binding and a
    local bin directory.
    """
    import os

    candidates = [
        "/data/prdrive/binaries/validation",
        "/data/prdrive/binaries/training",
        "/data/puffer/resources/drive/binaries/training",
    ]
    map_dir = next((path for path in candidates if os.path.isfile(os.path.join(path, "map_000.bin"))), None)
    if map_dir is None:
        print("SKIP test_c_simulator_reads_plr_queue_in_order (no map_000.bin)")
        return

    from pufferlib.ocean.drive import binding

    num_maps = 8
    num_agents = 40
    sampler = LevelSampler(
        num_maps=num_maps,
        strategy="prioritized",
        rho=0.0,
        nu=1.0,
        staleness_coef=0.0,
        alpha=1.0,
        seed=0,
    )
    for map_id in range(num_maps):
        sampler.update(map_id, score=0.0)
    sampler.update(2, score=100.0)
    queue = np.array([sampler.sample() for _ in range(num_agents * 4)], dtype=np.int32)

    agent_offsets, map_ids, env_count = binding.shared(
        map_dir=map_dir,
        num_agents=num_agents,
        num_maps=num_maps,
        init_mode=0,  # create_all_valid
        control_mode=1,  # control_agents
        sdc_controller=1,  # policy
        non_sdc_controller=1,
        non_vehicle_controller=1,
        init_steps=0,
        goal_behavior=0,
        goal_target_distance=30.0,
        max_controlled_agents=32,
        seed=0,
        map_id_queue=queue,
    )
    loaded = [int(map_id) for map_id in map_ids]
    print("\n--- C simulator map load (PLR queue) ---")
    print(f"map_dir={map_dir}")
    print(f"num_agents budget={num_agents}  envs opened={int(env_count)}")
    print(f"sampler proportion_seen={float(sampler.seen.mean()):.2f}")
    print(f"queue head ({len(queue)} candidates): {queue[:12].tolist()}")
    print("loaded, in order:")
    for env_idx, map_id in enumerate(loaded):
        agents = int(agent_offsets[env_idx + 1]) - int(agent_offsets[env_idx])
        path = os.path.join(map_dir, _map_bin_name(map_id))
        print(f"  env {env_idx}: id={map_id}  file={_map_bin_name(map_id)}  agents={agents}  exists={os.path.isfile(path)}")

    assert loaded, "C returned no maps"
    assert loaded == queue[: len(loaded)].tolist(), (
        f"C did not consume the PLR queue in order.\nqueue={queue[: len(loaded)].tolist()}\nloaded={loaded}"
    )

    sequential = np.arange(num_maps, dtype=np.int32)
    seq_offsets, seq_ids, seq_count = binding.shared(
        map_dir=map_dir,
        num_agents=num_agents,
        num_maps=num_maps,
        init_mode=0,
        control_mode=1,
        sdc_controller=1,
        non_sdc_controller=1,
        non_vehicle_controller=1,
        init_steps=0,
        goal_behavior=0,
        goal_target_distance=30.0,
        max_controlled_agents=32,
        seed=0,
        map_id_queue=sequential,
    )
    seq_loaded = [int(map_id) for map_id in seq_ids]
    print("\n--- C simulator map load (sequential queue) ---")
    print(f"queue: {sequential.tolist()}  envs opened={int(seq_count)}")
    print("loaded, in order:")
    for env_idx, map_id in enumerate(seq_loaded):
        agents = int(seq_offsets[env_idx + 1]) - int(seq_offsets[env_idx])
        print(f"  env {env_idx}: id={map_id}  file={_map_bin_name(map_id)}  agents={agents}")
    print(f"stopped after {len(seq_loaded)} maps; unused tail={sequential[len(seq_loaded):].tolist()}")
    assert seq_loaded == sequential[: len(seq_loaded)].tolist()


def test_merge_sampler_states_ignores_unseen():
    scores_a = np.array([10.0, 0.0, 3.0])
    seen_a = np.array([True, False, True])
    scores_b = np.array([0.0, 4.0, 9.0])
    seen_b = np.array([False, True, True])
    scores, seen = merge_sampler_states([(scores_a, seen_a), (scores_b, seen_b)])
    assert seen.tolist() == [True, True, True]
    assert np.allclose(scores, [10.0, 4.0, 6.0])


def test_score_replay_weights_rank_high_score():
    scores = np.array([-20.0, -1.0, -5.0])
    seen = np.array([True, True, True])
    weights = score_replay_weights(scores, seen, score_transform="rank", temperature=1.0)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[1] > weights[2] > weights[0]


def test_return_sampling_ranks_low_return_first():
    ls = LevelSampler(
        num_maps=5,
        rho=0.0,
        nu=1.0,
        staleness_coef=0.0,
        alpha=1.0,
        seed=2,
        score_transform="rank",
        invert_scores=True,
    )
    for map_id, score in enumerate([10.0, 8.0, 0.0, 6.0, 4.0]):
        ls.update(map_id, score=score)
    counts = np.zeros(5, dtype=int)
    for _ in range(2000):
        counts[ls.sample()] += 1
    assert counts[2] > counts[0]


def test_plr_hypers_match_cl4ad_10k():
    assert plr_hypers_for_score("max_mc") == (4.0, 0.3)
    assert plr_hypers_for_score("learnability") == (4.0, 0.1)
    assert plr_hypers_for_score("learnability_hard") == (2.0, 0.1)
    assert plr_hypers_for_score("learn_hard") == (2.0, 0.1)
    assert plr_hypers_for_score("learnability", temperature=9, staleness_coef=0.5) == (9.0, 0.5)


def test_agent_outcomes_fail_and_collision_per_car():
    # Two steps per episode, two episodes. Agent 0 never goals and collides;
    # agent 1 goals each episode without colliding.
    successes = np.array([[0, 0], [0, 1], [0, 0], [0, 1]], dtype=np.float64)
    unsafe = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], dtype=np.float64)
    dones = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=np.float64)
    tracker = FailTracker(num_maps=1, max_controlled_agents=2)
    tracker.update(0, successes=successes, dones=dones, unsafe=unsafe)
    assert np.allclose(tracker.outcome_n[0, :2], [2, 2])
    assert np.allclose(tracker.fail_ema[0, :2], [1.0, 0.0])
    assert np.allclose(tracker.coll_ema[0, :2], [1.0, 0.0])
    assert np.allclose(tracker.hit_ema[0, :2], [1.0, 0.0])
    assert np.isclose(fail_top1_share(tracker.fail_ema[0], tracker.outcome_n[0], min_episodes=2), 1.0)
    assert np.isclose(fail_worst_stat(tracker.fail_ema[0], tracker.coll_ema[0], tracker.outcome_n[0], min_episodes=2), 1.0)
    assert np.isclose(fail_worst_stat(tracker.fail_ema[0], tracker.hit_ema[0], tracker.outcome_n[0], min_episodes=2), 1.0)


def test_agent_outcomes_skip_open_tail_and_repeated_terminals():
    # Goal on t=1 then Drive keeps terminals=1; last two steps are an open tail.
    successes = np.array([[0], [1], [0], [0], [0]], dtype=np.float64)
    unsafe = np.array([[1], [0], [0], [0], [0]], dtype=np.float64)
    dones = np.array([[0], [1], [1], [0], [0]], dtype=np.float64)
    tracker = FailTracker(num_maps=1, max_controlled_agents=1)
    tracker.update(0, successes=successes, dones=dones, unsafe=unsafe)
    assert np.allclose(tracker.outcome_n[0, :1], [1])
    assert np.allclose(tracker.fail_ema[0, :1], [1.0])
    assert np.allclose(tracker.coll_ema[0, :1], [1.0])
    assert np.allclose(tracker.hit_ema[0, :1], [1.0])


def test_agent_outcomes_multi_hit_on_completed_episode():
    successes = np.array([[0], [0], [0], [0]], dtype=np.float64)
    unsafe = np.array([[1], [1], [1], [0]], dtype=np.float64)
    dones = np.array([[0], [0], [0], [1]], dtype=np.float64)
    tracker = FailTracker(num_maps=1, max_controlled_agents=1)
    tracker.update(0, successes=successes, dones=dones, unsafe=unsafe)
    assert np.allclose(tracker.outcome_n[0, :1], [1])
    assert np.allclose(tracker.fail_ema[0, :1], [1.0])
    assert np.allclose(tracker.coll_ema[0, :1], [1.0])
    assert np.allclose(tracker.hit_ema[0, :1], [3.0])


def test_fail_top1_spread_is_not_structural():
    fail = np.array([0.25, 0.25, 0.25, 0.25])
    visits = np.ones(4) * 8
    assert np.isclose(fail_top1_share(fail, visits), 0.25)
    assert fail_top1_share(fail, np.zeros(4)) is None


def test_update_agent_outcomes_ema_keeps_same_car():
    tracker = FailTracker(num_maps=2, max_controlled_agents=3, alpha=0.5)
    successes = np.array([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=np.float64)
    dones = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=np.float64)
    unsafe = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.float64)
    tracker.update(0, successes=successes, dones=dones, unsafe=unsafe)
    tracker.update(0, successes=successes, dones=dones, unsafe=unsafe)
    assert tracker.outcome_n[0, 0] == 4
    assert np.isclose(tracker.fail_ema[0, 0], 1.0)
    assert np.isclose(tracker.fail_ema[0, 1], 0.0)
    assert np.isclose(tracker.coll_ema[0, 0], 1.0)
    assert np.isclose(tracker.hit_ema[0, 0], 2.0)
    assert np.isclose(fail_top1_share(tracker.fail_ema[0], tracker.outcome_n[0]), 1.0)


def test_fail_tracker_from_rewards_marks_negative_as_fail():
    tracker = FailTracker(num_maps=1, max_controlled_agents=2)
    rewards = np.array([[-1.0, 1.0], [0.0, 0.0]])
    dones = np.array([[0.0, 0.0], [1.0, 1.0]])
    tracker.update(0, rewards=rewards, dones=dones)
    assert tracker.outcome_n[0, 0] >= 1
    assert tracker.fail_ema[0, 0] > tracker.fail_ema[0, 1]


def test_merge_sampler_states_outcomes_are_visit_weighted():
    fail_a = np.array([[1.0, 0.0], [0.0, 0.0]])
    coll_a = np.array([[1.0, 0.0], [0.0, 0.0]])
    hit_a = np.array([[3.0, 0.0], [0.0, 0.0]])
    n_a = np.array([[10.0, 10.0], [0.0, 0.0]])
    fail_b = np.array([[0.0, 0.0], [0.5, 0.5]])
    coll_b = np.array([[0.0, 0.0], [0.0, 0.0]])
    hit_b = np.array([[0.0, 0.0], [1.0, 1.0]])
    n_b = np.array([[0.0, 0.0], [4.0, 4.0]])
    merged = merge_sampler_states(
        [
            {
                "scores": np.array([1.0, 2.0]),
                "seen": np.array([True, False]),
                "fail_ema": fail_a,
                "coll_ema": coll_a,
                "hit_ema": hit_a,
                "outcome_n": n_a,
            },
            {
                "scores": np.array([0.0, 4.0]),
                "seen": np.array([False, True]),
                "fail_ema": fail_b,
                "coll_ema": coll_b,
                "hit_ema": hit_b,
                "outcome_n": n_b,
            },
        ]
    )
    assert np.allclose(merged["scores"], [1.0, 4.0])
    assert np.allclose(merged["fail_ema"][0], [1.0, 0.0])
    assert np.allclose(merged["fail_ema"][1], [0.5, 0.5])
    assert np.allclose(merged["hit_ema"][0], [3.0, 0.0])
    assert np.allclose(merged["outcome_n"][0], [10.0, 10.0])


def test_update_accepts_one_or_many():
    ls_a = LevelSampler(num_maps=8, seed=5)
    ls_b = LevelSampler(num_maps=8, seed=5)
    map_ids = [1, 3, 3, 7]
    scores = [1.0, 2.0, 5.0, -1.0]
    for m, s in zip(map_ids, scores):
        ls_a.update(m, s)
    ls_b.update(map_ids, scores)
    assert np.allclose(ls_a.scores, ls_b.scores)
    assert np.array_equal(ls_a.seen, ls_b.seen)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
