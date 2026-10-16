"""Standalone sanity checks for LevelSampler (no C extension required)."""

import numpy as np

from pufferlib.ocean.drive.level_sampler import LevelSampler


def test_batch_shape_and_range():
    ls = LevelSampler(num_maps=10, seed=0)
    batch, metrics = ls.sample_batch(37)
    assert batch.shape == (37,)
    assert batch.dtype == np.int32
    assert batch.min() >= 0 and batch.max() < 10
    assert "level_sampler/proportion_seen" in metrics


def test_explores_before_replaying():
    # rho=1.0 forces "see everything at least once" before any replay is possible.
    ls = LevelSampler(num_maps=20, rho=1.0, seed=1)
    seen_order = []
    for _ in range(20):
        map_id = ls.sample_one()
        ls.update(map_id, score=0.0)
        seen_order.append(map_id)
    assert ls.seen.all(), "expected every map to be seen after num_maps unseen-forced draws"
    assert len(set(seen_order)) == 20, "expected no repeats while still exploring unseen maps"


def test_high_score_maps_get_sampled_more():
    ls = LevelSampler(num_maps=5, rho=0.0, nu=1.0, staleness_coef=0.0, alpha=1.0, seed=2)
    # Seed every map once so all are "seen" and eligible for replay weighting.
    for map_id in range(5):
        ls.update(map_id, score=0.0)
    # Now make map 3 look much more valuable to train on than the rest.
    ls.update(3, score=100.0)

    counts = np.zeros(5, dtype=int)
    for _ in range(2000):
        map_id = ls.sample_one()
        counts[map_id] += 1
        # Immediately re-report the same low/high scores so the distribution stays stable.
        ls.update(map_id, score=100.0 if map_id == 3 else 0.0)

    assert counts[3] > counts.sum() * 0.5, f"expected map 3 to dominate sampling, got counts={counts}"


def test_staleness_favors_untouched_maps():
    ls = LevelSampler(num_maps=4, rho=0.0, nu=1.0, staleness_coef=1.0, alpha=1.0, seed=3)
    for map_id in range(4):
        ls.update(map_id, score=1.0)  # identical scores -> pure staleness should drive selection

    # Manually make map 2 "fresh" (staleness 0) and everything else very stale.
    ls.staleness[:] = 50.0
    ls.staleness[2] = 0.0

    counts = np.zeros(4, dtype=int)
    for _ in range(200):
        map_id = ls.sample_one()
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
        seen.add(ls.sample_one())
    assert seen == set(range(6))


def test_update_batch_matches_update():
    ls_a = LevelSampler(num_maps=8, seed=5)
    ls_b = LevelSampler(num_maps=8, seed=5)
    map_ids = [1, 3, 3, 7]
    scores = [1.0, 2.0, 5.0, -1.0]
    for m, s in zip(map_ids, scores):
        ls_a.update(m, s)
    ls_b.update_batch(map_ids, scores)
    assert np.allclose(ls_a.scores, ls_b.scores)
    assert np.array_equal(ls_a.seen, ls_b.seen)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed")
