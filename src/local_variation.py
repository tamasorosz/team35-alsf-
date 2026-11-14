"""Incremental empirical sensitivity checks in a tolerance-normalized L-infinity box.

A *verified violation* means that TWO previously evaluated designs at a
Chebyshev distance <= epsilon have a relative objective difference exceeding
one of the specified thresholds. It does not establish a global worst case.

This cache assumes an append-only archive. It updates old endpoints when a new
sample arrives, and requires no new objective-function evaluations.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def tolerance_vector(tolerances, dimension: int) -> np.ndarray:
    values = np.asarray(tolerances, dtype=float)
    if values.ndim == 0:
        values = np.full(dimension, float(values))
    if values.shape != (dimension,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"tolerances must be {dimension} finite positive entries")
    return values


def relative_pairwise_variation(a, b, floors):
    """Symmetric relative difference with a FIXED positive denominator floor.

    r_j = |a_j - b_j|/max(|a_j|, |b_j|, floor_j).
    This avoids an undefined zero-medoid relative measure. Floors are a chosen
    benchmark policy and must be reported alongside thresholds.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    floors = np.asarray(floors, dtype=float)
    if np.any(floors <= 0) or not np.isfinite(floors).all():
        raise ValueError("objective floors must be finite and positive")
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floors)


class LocalVariationState:
    """Incrementally mark observed high-variation neighborhoods; no trimming.

    On each update, compare only newly added archive points with all preceding
    points inside the L-infinity tolerance radius. Existing verified violating
    endpoints remain verified for the duration of the run. This behavior is
    intentional: newly added archive points cannot erase an observed violation.
    """

    def __init__(self, tolerances, thresholds=(0.1, 0.1),
                 floors=(1e-3, 1e-3), radius=1.0):
        self.tolerances = tolerances
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.floors = np.asarray(floors, dtype=float)
        self.radius = float(radius)
        if self.thresholds.shape != (2,) or np.any(self.thresholds <= 0):
            raise ValueError("thresholds must contain two positive values")
        if self.floors.shape != (2,) or np.any(self.floors <= 0):
            raise ValueError("floors must contain two positive values")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("radius must be a positive finite number")
        self._old_x = np.empty((0, 0))
        self._old_f = np.empty((0, 2))
        self.verified_violation = np.zeros(0, dtype=bool)
        self.max_observed_variation = np.zeros((0, 2), dtype=float)

    def update(self, coords: np.ndarray, objectives: np.ndarray):
        coords = np.asarray(coords, dtype=float)
        objectives = np.asarray(objectives, dtype=float)
        if coords.ndim != 2 or objectives.shape != (len(coords), 2):
            raise ValueError("expected X[n,d] and objectives[n,2]")
        if not np.isfinite(coords).all() or not np.isfinite(objectives).all():
            raise ValueError("non-finite archive entries")
        n, d = coords.shape
        delta = tolerance_vector(self.tolerances, d)
        previous = len(self._old_f)
        if (n < previous or (previous and
                (self._old_x.shape[1] != d or
                 not np.array_equal(coords[:previous], self._old_x) or
                 not np.array_equal(objectives[:previous], self._old_f)))):
            previous = 0
            self.verified_violation = np.zeros(n, dtype=bool)
            self.max_observed_variation = np.zeros((n, 2), dtype=float)
        elif n > previous:
            self.verified_violation = np.pad(self.verified_violation, (0, n-previous))
            self.max_observed_variation = np.pad(
                self.max_observed_variation, ((0, n-previous), (0, 0)))

        scaled = coords / delta
        tree = cKDTree(scaled) if n else None
        # One comparison for each unordered pair, including pairs where both
        # endpoints were added in this update. No full N-by-N matrix is built.
        for i in range(previous, n):
            near = tree.query_ball_point(scaled[i], r=self.radius + 1e-12, p=np.inf)
            other = np.asarray([j for j in near if j < i], dtype=int)
            if not len(other):
                continue
            variation = relative_pairwise_variation(objectives[other], objectives[i], self.floors)
            self.max_observed_variation[i] = np.maximum(self.max_observed_variation[i], variation.max(axis=0))
            np.maximum.at(self.max_observed_variation, other, variation)
            bad = np.any(variation > self.thresholds, axis=1)
            if bad.any():
                self.verified_violation[i] = True
                self.verified_violation[other[bad]] = True

        counts = (tree.query_ball_point(scaled, r=self.radius + 1e-12,
                                        p=np.inf, return_length=True)
                  if n else np.zeros(0, dtype=int))
        self._old_x = coords.copy()
        self._old_f = objectives.copy()
        return counts
