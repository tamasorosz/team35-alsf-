"""pymoo offspring repair that avoids verified high-variation tolerance boxes."""
from __future__ import annotations
import numpy as np
from pymoo.core.repair import Repair
from src.find_robust_clusters import decision_columns
from src.offspring_policy import VerifiedViolationFilter


class OffspringRepair(Repair):
    def __init__(self, archive_path, cache, d_repair,
                 min_neighbours=3, decimal_places=8,
                 max_resample_attempts=100, rng=None, tolerances=None):
        super().__init__()
        if tolerances is None:
            raise ValueError('Explicit decision-variable tolerances required')
        self.archive_path = archive_path
        self.cache = cache
        self.d_repair = float(d_repair)
        # Legacy argument retained for API compatibility but not used:
        # a SINGLE verified violation is now sufficient to forbid a candidate.
        self.min_neighbours = min_neighbours
        self.decimal_places = int(decimal_places)
        self.max_resample_attempts = int(max_resample_attempts)
        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.policy = VerifiedViolationFilter(tolerances, radius=d_repair)
        self.n_regenerated = 0
        self.n_attempts = 0

    def update_flags(self, archive_df):
        coords = archive_df[decision_columns(archive_df)].to_numpy(dtype=float)
        if 'verified_violation' not in archive_df:
            raise ValueError('Archive lacks verified_violation; run local classifier first')
        self.policy.update(coords, archive_df['verified_violation'].astype(bool).to_numpy())

    def _forbidden(self, x):
        return self.policy.forbidden(x)

    def _do(self, problem, X, **kwargs):
        data = np.clip(np.round(np.asarray(X, dtype=float).copy(),
                                self.decimal_places), problem.xl, problem.xu)
        for i in range(len(data)):
            if not self._forbidden(data[i]):
                continue
            for _ in range(self.max_resample_attempts):
                self.n_attempts += 1
                candidate = np.round(self.rng.uniform(problem.xl, problem.xu),
                                     self.decimal_places)
                if not self._forbidden(candidate):
                    data[i] = candidate
                    self.n_regenerated += 1
                    break
            else:
                raise RuntimeError('ALSF could not sample outside verified high-sensitivity '
                                   'tolerance boxes. No forbidden offspring was accepted; '
                                   'review tolerance size, thresholds and search bounds.')
        return data

    def pop_counters(self):
        result = self.n_regenerated, self.n_attempts
        self.n_regenerated = 0
        self.n_attempts = 0
        return result
