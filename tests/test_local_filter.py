import numpy as np
import pandas as pd
import pytest
from src.local_variation import LocalVariationState, relative_pairwise_variation
from src.offspring_policy import VerifiedViolationFilter
from src.find_robust_clusters import classify_dataframe


def frame(points, fs):
    x = np.asarray(points, dtype=float)
    f = np.asarray(fs, dtype=float)
    return pd.DataFrame(np.column_stack([x,f]),
                        columns=[f'X{i+1}' for i in range(x.shape[1])]+['Obj1','Obj2'])


def test_chebyshev_box_accepts_diagonal_that_l2_misses():
    # each parameter is within 0.1; L2-normalized distance = sqrt(2) > 1
    data = frame([[0,0],[0.09,0.09],[2,2]], [[1,1],[1.04,1.02],[1.0,1.0]])
    result, stats = classify_dataframe(data, [0.1,0.1], 0.1,0.1, min_samples=2,
                                        floors=(0.01,0.01))
    assert list(result.flag) == ['robust','robust','NEI']
    assert result.loc[0,'cluster_id'] == result.loc[1,'cluster_id']
    assert result.loc[2,'cluster_id'] == -1


def test_violation_detected_at_noise_and_buffer_forbids_offspring():
    data = frame([[0,0], [0.08,-0.08], [1,1]], [[1,1],[1.5,1],[1,1]])
    out, _ = classify_dataframe(data, [0.1,0.1], .1,.1, min_samples=10,
                                floors=(.01,.01))
    assert list(out.flag) == ['non-robust','non-robust','NEI']
    assert list(out.cluster_id) == [-1,-1,-1]   # DBSCAN noise != safe
    filt = VerifiedViolationFilter([0.1,0.1])
    filt.update(data[['X1','X2']].to_numpy(), out.verified_violation.to_numpy())
    assert filt.forbidden([0.02,0.06])
    assert not filt.forbidden([0.5,0.5])


def test_cluster_chaining_does_not_spread_violation_to_distant_points():
    # Density chain of four points, but the final point is not a
    # neighbor of the first, and local flags are computed independently.
    points = [[0],[0.08],[0.16],[0.24]]
    fs = [[1,1],[1,1],[1.02,1],[1.5,1]]
    out, stats = classify_dataframe(frame(points,fs), .1,.1,.1,
                                     min_samples=2, floors=(.01,.01))
    assert stats['n_clusters'] == 1
    assert list(out.flag) == ['robust','robust','non-robust','non-robust']


def test_new_point_updates_previous_violation_incrementally():
    state = LocalVariationState([.1,.1], (.1,.1), (.01,.01))
    first = frame([[0,0],[2,2]],[[1,1],[1,1]])
    out1,_ = classify_dataframe(first, [.1,.1], .1,.1, min_samples=2,state=state)
    assert not out1.verified_violation.any()
    second = frame([[0,0],[2,2],[0.05,0.05]],[[1,1],[1,1],[2,1]])
    out2,_ = classify_dataframe(second, [.1,.1], .1,.1, min_samples=2,state=state)
    assert out2.verified_violation.tolist()==[True,False,True]


def test_symmetric_relative_differences_and_zero_floor():
    a = np.array([0., 1.])
    b = np.array([0.0005,1.2])
    assert np.allclose(relative_pairwise_variation(a,b,[.001,.001]),
                       relative_pairwise_variation(b,a,[.001,.001]))
    assert np.allclose(relative_pairwise_variation(a,b,[.001,.001]),[.5,1/6])


def test_radius_boundary_accepts_exact_tolerance():
    f = VerifiedViolationFilter([.1,.2])
    f.update(np.array([[0,0]]),np.array([True]))
    assert f.forbidden([.1,.2])
    assert not f.forbidden([.10001,.2])


def test_reject_non_positive_tolerances():
    with pytest.raises(ValueError):
        classify_dataframe(frame([[0],[1]],[[1,1],[1,1]]), 0, .1,.1)
