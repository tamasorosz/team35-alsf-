"""Pavel review variant: tolerance-normalized Chebyshev DBSCAN and LOCAL flags.

DBSCAN is used to visualize density-connected components, *not* to assert
stability across the entire chain. Local objective deviations are checked only
for archived pairs inside the same prescribed tolerance hyperrectangle.
Greedy medoid trimming and its optimistic boundary bias are removed entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from src.local_variation import LocalVariationState, tolerance_vector


def decision_columns(df):
    cols = sorted((c for c in df if c.startswith("X") and c[1:].isdigit()),
                  key=lambda c: int(c[1:]))
    if not cols or cols != [f"X{i}" for i in range(1, len(cols) + 1)]:
        raise ValueError("Archive must contain X1..Xn consecutively")
    return cols


def classify_dataframe(archive: pd.DataFrame, tolerances, t1, t2,
                       min_samples=10, radius=1.0, floors=(1e-3, 1e-3),
                       state: LocalVariationState | None = None):
    """Classify each archive point using observed L-infinity local evidence.

    robust: min_samples observations in its own tolerance neighborhood and NO
            observed pairwise objective-threshold violation at this point.
    non-robust: endpoint of at least one observed violating pair, regardless of
                whether DBSCAN considers the endpoint core, border or noise.
    NEI: not enough direct local observations to claim robustness.

    Importantly, these are archive-supported labels, not worst-case guarantees.
    A single DBSCAN cluster can contain robust and non-robust points.
    """
    archive = archive.reset_index(drop=True).copy()
    cols = decision_columns(archive)
    X = archive[cols].to_numpy(dtype=float)
    F = archive[['Obj1','Obj2']].to_numpy(dtype=float)
    delta = tolerance_vector(tolerances, len(cols))
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if state is None:
        state = LocalVariationState(delta, thresholds=(t1, t2), floors=floors,
                                    radius=radius)
    counts = state.update(X, F)
    normalized = X / delta
    labels = (DBSCAN(eps=radius + 1e-12, min_samples=min_samples,
                     metric='chebyshev', algorithm='auto').fit_predict(normalized)
              if len(X) else np.zeros(0, dtype=int))
    bad = state.verified_violation
    flag = np.full(len(X), 'NEI', dtype=object)
    flag[(counts >= min_samples) & ~bad] = 'robust'
    flag[bad] = 'non-robust'
    archive['flag'] = flag
    archive['cluster_id'] = labels
    archive['is_medoid'] = False
    archive['cluster_pass'] = np.where(bad, 4, np.where(flag == 'robust', 1, 0))
    archive['verified_violation'] = bad
    archive['local_n'] = counts
    archive['R1_local'] = state.max_observed_variation[:, 0]
    archive['R2_local'] = state.max_observed_variation[:, 1]
    archive['cluster_n'] = 0

    # Representative indices support old plotting scripts. Approximate medoid:
    # closest actual member to geometric center of the group, in normalized
    # coordinates. NOT used for sensitivity classification.
    flags_by_cluster = {}
    for cid in np.unique(labels[labels >= 0]):
        indices = np.flatnonzero(labels == cid)
        archive.loc[indices, 'cluster_n'] = len(indices)
        members = normalized[indices]
        center = members.mean(axis=0)
        medoid_candidate = indices[np.argmin(np.max(np.abs(members - center), axis=1))]
        archive.at[medoid_candidate, 'is_medoid'] = True
        statuses = set(flag[indices])
        flags_by_cluster[int(cid)] = ('non-robust' if 'non-robust' in statuses else
                                       'robust' if statuses == {'robust'} else 'NEI')

    stats = {
        'n_clusters': len(flags_by_cluster),
        'n_robust_clusters': sum(v == 'robust' for v in flags_by_cluster.values()),
        'n_non_robust_clusters': sum(v == 'non-robust' for v in flags_by_cluster.values()),
        'n_nei_clusters': sum(v == 'NEI' for v in flags_by_cluster.values()),
        'n_noise': int(np.count_nonzero(labels == -1)),
        'n_verified_violations': int(bad.sum()),
    }
    return archive, stats


def flag_by_clusters(archive_path, d_cluster, t1, t2, min_samples=10,
                     max_trim_fraction=None, tolerances=None, floors=(1e-3, 1e-3),
                     state=None):
    """Compatibility entry point; d_cluster is radius in tolerance units.

    max_trim_fraction is retained in the signature for old callers, but greedy
    trimming is NOT performed: it could create false-safe cluster boundaries.
    The tolerance vector is mandatory to avoid silently reverting to raw L2.
    """
    if tolerances is None:
        raise ValueError("tolerances must be provided explicitly in the L-infinity version")
    archive = pd.read_csv(archive_path)
    archive, stats = classify_dataframe(
        archive, tolerances, t1, t2, min_samples, radius=float(d_cluster),
        floors=floors, state=state,
    )
    archive.to_csv(archive_path, index=False)
    print(f"Local L-inf filter ({len(archive)} evaluations, {stats['n_clusters']} DBSCAN components): "
          f"{stats['n_verified_violations']} verified-violation endpoints, "
          f"{int((archive.flag == 'robust').sum())} locally stable samples, "
          f"{int((archive.flag == 'NEI').sum())} NEI")
    return archive, stats


def robust_cluster_medoids(archive_path):
    archive = pd.read_csv(archive_path)
    if 'flag' not in archive or 'is_medoid' not in archive:
        return pd.DataFrame()
    return archive[(archive.flag == 'robust') & archive.is_medoid][
        decision_columns(archive) + ['Obj1','Obj2']].reset_index(drop=True)
