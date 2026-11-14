"""ALSF + NSGA-II for the 10-D Di Barba solenoid benchmark.

Project layout expected:
  project_root/
    di_barba_solenoid_benchmark.py
    run_di_barba_alsf.py
    src/preprocessing.py          # user's n-dimensional OffspringRepair
    src/postprocessing.py         # user's FlagCallback
    src/find_robust_clusters.py   # user's n-dimensional cluster implementation
    results/

This script reuses the USER'S actual ALSF implementation unchanged.
Variable coordinates are normalized J/(1 MA/m2); all distance thresholds
below are in dimensionless 10-D decision space, NOT meters.

NOTE: This is the Pavel-review experimental variant. The pairwise
variation thresholds and fixed denominator floors are test parameters,
not physical worst-case guarantees.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from di_barba_solenoid_benchmark import DiBarbaSolenoid, SolenoidSettings
from src.preprocessing import OffspringRepair
from src.postprocessing import FlagCallback

ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get('ALSF_RESULTS_DIR', ROOT / 'results'))
ARCHIVE = RESULTS / 'archive_di_barba_alsf_10d.csv'
PROCESS = RESULTS / 'process_di_barba_alsf_10d.csv'
PARETO = RESULTS / 'pareto_di_barba_alsf_10d.csv'

POP_SIZE = int(os.environ.get('ALSF_POP_SIZE', 100))
N_OFFSPRINGS = int(os.environ.get('ALSF_OFFSPRINGS', 100))
N_GEN = int(os.environ.get('ALSF_N_GEN', 200))
SEED = 42
SBX_PROB = 0.9
SBX_ETA = 5
PM_PROB = 1.0
PM_ETA = 5
DECIMALS = 8

# Distance is measured in x_i = J_i/(1 MA/m2), dimensionless coordinates.
# For comparison: 0.10 in one coordinate corresponds to 0.10 MA/m2.
D_CLUSTER = 1.0  # 1 = one tolerance hyperrectangle (Chebyshev norm)
D_REPAIR = 1.0   # buffer around every verified violation endpoint
T1 = 0.10
T2 = 0.10
MIN_SAMPLES = 10
MAX_TRIM_FRACTION = 0.0  # obsolete: no greedy trimming in this version
FLAG_START_GEN = int(os.environ.get('ALSF_FLAG_START_GEN', 50))
REPAIR_MIN_NEIGHBOURS = 1  # obsolete: ANY verified violation rejects
MAX_RESAMPLE_ATTEMPTS = int(os.environ.get('ALSF_RESAMPLE_ATTEMPTS', 100))

SETTINGS = SolenoidSettings(
    j_min=-2.0e6,
    j_max=+2.0e6,
    j_scale=1.0e6,
    independent_j_tolerance=0.1e6,
)
model = DiBarbaSolenoid(SETTINGS)
# x_i = J_i/J_scale: tolerance = independent_j_tolerance/j_scale.
TOLERANCES = np.full(model.n_var, SETTINGS.independent_j_tolerance / SETTINGS.j_scale)
# Explicit fixed objective floors prevent zero-denominator relative variations.
OBJECTIVE_FLOORS = (1e-3, 1e-3)
COLUMNS = [f'X{i}' for i in range(1, model.n_var + 1)] + ['Obj1', 'Obj2']


class DiBarbaProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=model.n_var, n_obj=2,
                         n_ieq_constr=0, n_eq_constr=0,
                         xl=model.lower, xu=model.upper, vtype=float)
        self.cache = {}

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(np.asarray(x, dtype=float), DECIMALS)
        key = tuple(x.tolist())
        if key not in self.cache:
            values = model.evaluate(x)
            self.cache[key] = values
            pd.DataFrame([[*x, *values]], columns=COLUMNS).to_csv(
                ARCHIVE, mode='a', header=not ARCHIVE.exists(), index=False)
        out['F'] = self.cache[key]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    existing = [p for p in (ARCHIVE, PROCESS, PARETO) if p.exists()]
    if existing:
        raise FileExistsError('A clean ALSF run requires fresh output files:\n'
                              + '\n'.join(map(str, existing))
                              + '\nRename outputs for another run or move old files.')

    problem = DiBarbaProblem()
    rng = np.random.default_rng(SEED)
    repair = OffspringRepair(str(ARCHIVE), problem.cache, D_REPAIR,
                             REPAIR_MIN_NEIGHBOURS, DECIMALS,
                             MAX_RESAMPLE_ATTEMPTS, rng, tolerances=TOLERANCES)
    callback = FlagCallback(str(ARCHIVE), str(PROCESS), D_CLUSTER,
                            T1, T2, MIN_SAMPLES, MAX_TRIM_FRACTION,
                            FLAG_START_GEN, repair, tolerances=TOLERANCES,
                            objective_floors=OBJECTIVE_FLOORS)
    algorithm = NSGA2(
        pop_size=POP_SIZE, n_offsprings=N_OFFSPRINGS,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=SBX_PROB, eta=SBX_ETA),
        mutation=PM(prob=PM_PROB, eta=PM_ETA),
        repair=repair, eliminate_duplicates=True,
    )
    result = minimize(problem, algorithm, get_termination('n_gen', N_GEN),
                      seed=SEED, callback=callback,
                      save_history=False, verbose=True)

    if result.X is None or result.F is None:
        raise RuntimeError('ALSF did not produce a feasible Pareto front')
    pareto = pd.DataFrame(np.column_stack([result.X, result.F]), columns=COLUMNS)
    if ARCHIVE.exists():
        archive = pd.read_csv(ARCHIVE)
        if 'flag' in archive.columns:
            # Independent of extra archive flag/cluster columns.
            x_cols = COLUMNS[:model.n_var]
            key_to_flag = {
                tuple(np.round(row, DECIMALS)): flag
                for row, flag in zip(archive[x_cols].to_numpy(dtype=float),
                                     archive['flag'].astype(str))
            }
            pareto['flag'] = [key_to_flag.get(tuple(np.round(x, DECIMALS)), 'NEI')
                              for x in result.X]
    pareto.to_csv(PARETO, index=False)
    print('\n=== DI BARBA ALSF RUN ===')
    print('Pymoo function evaluations:', result.algorithm.evaluator.n_eval)
    print('Unique archive evaluations:', len(problem.cache))
    print('Pareto solutions:', len(pareto))
    print('Archive:', ARCHIVE)
    print('Process:', PROCESS)
    print('Pareto:', PARETO)
    print('Metric: tolerance-normalized Chebyshev (L-inf); radius:', D_CLUSTER)
    print('Per-variable tolerance:', TOLERANCES)
    print('Relative variation floors:', OBJECTIVE_FLOORS)


if __name__ == '__main__':
    main()
