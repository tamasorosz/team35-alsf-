"""Pure-Python/Python-scientific-core violation-buffer rejection policy.

This module deliberately does NOT import pymoo, so it can be unit tested and
run as a diagnostic even when the evolutionary framework is unavailable.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from src.local_variation import tolerance_vector


class VerifiedViolationFilter:
    def __init__(self, tolerances, radius=1.0):
        self.tolerances = tolerances
        self.radius = float(radius)
        self._tree = None
        self.count = 0

    def update(self, coords: np.ndarray, verified: np.ndarray):
        coords = np.asarray(coords, dtype=float)
        verified = np.asarray(verified, dtype=bool)
        if coords.ndim != 2 or verified.shape != (len(coords),):
            raise ValueError('Expected coords[n,d] and verified[n]')
        delta = tolerance_vector(self.tolerances, coords.shape[1])
        sensitive = coords[verified] / delta
        self._tree = cKDTree(sensitive) if len(sensitive) else None
        self._delta = delta
        self.count = len(sensitive)

    def forbidden(self, x: np.ndarray) -> bool:
        if self._tree is None:
            return False
        distances, _ = self._tree.query(np.asarray(x)/self._delta,
                                        p=np.inf, k=1,
                                        distance_upper_bound=self.radius + 1e-12)
        return bool(np.isfinite(distances))
