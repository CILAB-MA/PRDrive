"""Prioritized Level Replay (PLR) over WOMD map ids.

Ports the sampling algorithm from `facebookresearch/level-replay`
(level_replay/level_sampler.py -- Jiang et al., "Prioritized Level Replay"),
adapted to this repo in two ways:

1. "Levels" here are map ids (0..num_maps-1) rather than environment seeds.
2. The original picks one level per parallel actor. Here, the C simulator
   (see `pufferlib/ocean/drive/binding.c`, `my_shared()`) doesn't consume a
   fixed number of maps -- it keeps loading maps until it has collected
   enough controllable agents, because different maps contain different
   numbers of agents. So instead of `sample()` returning one map id,
   `sample_batch()` returns an *ordered* batch of candidate map ids (with
   replacement, PLR-weighted) that C walks through in order, using only as
   many as it needs. Unused entries at the tail are harmless.

This module does not compute the training score itself (the original
supports several score functions -- policy entropy, GAE, value L1, one-step
TD error -- that need access to rollout internals). The caller is expected
to compute whatever score signal it wants (e.g. episode return, collision
rate, value loss) and pass it to `update()` / `update_batch()`.
"""

import numpy as np


class LevelSampler:
    """Tracks a score and a staleness value per map id, and samples which
    map to try next by blending "replay high-score maps" with "make sure
    stale/unseen maps aren't starved" -- see `sample_one` for the schedule.
    """

    def __init__(
        self,
        num_maps,
        strategy="prioritized",  # "prioritized" or "uniform"
        score_transform="power",  # "power" or "rank"
        temperature=1.0,
        staleness_coef=0.1,
        staleness_transform="power",
        staleness_temperature=1.0,
        rho=0.2,  # fraction of maps that must be seen before replay can start
        nu=0.5,  # once past rho, probability of replaying vs. sampling unseen
        alpha=1.0,  # score EMA rate: new = (1 - alpha) * old + alpha * latest
        eps=1e-3,  # power-transform smoothing so a never-updated score isn't exactly 0
        seed=None,
    ):
        if strategy not in ("prioritized", "uniform"):
            raise ValueError(f"LevelSampler strategy must be 'prioritized' or 'uniform', got {strategy!r}")
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
        self._rng = np.random.default_rng(seed)

        # Per-map state.
        self.scores = np.zeros(self.num_maps, dtype=np.float64)
        self.staleness = np.zeros(self.num_maps, dtype=np.float64)
        self.seen = np.zeros(self.num_maps, dtype=bool)

    # ---- weighting -------------------------------------------------

    def _transform(self, transform, temperature, values):
        values = np.asarray(values, dtype=np.float64)
        if transform == "power":
            e = 0.0 if self.staleness_coef > 0 else self.eps
            return (values + e) ** (1.0 / temperature)
        # "rank": highest value gets rank 1, weight ~ 1 / rank^(1/temperature)
        order = np.flip(values.argsort())
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order)) + 1
        return 1.0 / (ranks ** (1.0 / temperature))

    def _sample_weights(self):
        """Sampling distribution over already-seen maps, blending score and staleness."""
        weights = self._transform(self.score_transform, self.temperature, self.scores)
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
        if not np.isclose(weights.sum(), 0.0):
            map_id = int(self._rng.choice(self.num_maps, p=weights))
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
        """Pick a single next map id.

        "uniform" strategy: plain random choice (a baseline / ablation).
        "prioritized": explore unseen maps until at least `rho` of all maps
        have been seen, then mostly replay by score+staleness (probability
        `nu`), occasionally still sampling something unseen. Once every map
        has been seen at least once, always replay.
        """
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
