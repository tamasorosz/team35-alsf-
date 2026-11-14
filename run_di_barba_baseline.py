"""NSGA-II baseline and independently calculated Pareto reference for the
Di Barba (2016)-inspired ten-current solenoid benchmark.

Usage:
    Place alongside di_barba_solenoid_benchmark.py in the project root and run
    directly in PyCharm. No command-line arguments.

Dependencies: numpy scipy pandas matplotlib, plus pymoo for NSGA-II.
The reference and diagnostic figures work without pymoo; in that case the
program prints a warning and generates only the reference.

Results are distinct from TEAM 35 and ALSF outputs and never share cache.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from di_barba_solenoid_benchmark import DiBarbaSolenoid, SolenoidSettings

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures' / 'di_barba'

REFERENCE_CSV = RESULTS / 'pareto_di_barba_reference.csv'
BASELINE_ARCHIVE_CSV = RESULTS / 'archive_di_barba_baseline.csv'
BASELINE_PARETO_CSV = RESULTS / 'pareto_di_barba_baseline.csv'
REFERENCE_SENSITIVITY_CSV = RESULTS / 'di_barba_reference_sensitivity.csv'

POP_SIZE = 100
N_OFFSPRINGS = 100
N_GEN = 200
SEED = 42
DECIMALS = 8
SBX_PROB = 0.9
SBX_ETA = 5
PM_PROB = 1.0
PM_ETA = 5

# Keep this False if you only want the analytical Pareto reference and figures.
RUN_NSGA2 = True
# Will refuse to overwrite a clean baseline result. Rename old outputs to rerun.
REQUIRE_NEW_BASELINE = True

# Physical model config. Article-verified settings use SolenoidSettings defaults.
# Adjustable assumptions: j_min, j_max, j_scale, copper_resistivity,
# independent_j_tolerance. Do NOT claim identical Figure 1 numbers until the
# exact published objectives, bounds and normalization are verified.
SETTINGS = SolenoidSettings(
    j_min=-2.0e6,                      # A/m2, adjustable assumption
    j_max=+2.0e6,                      # A/m2, adjustable assumption
    j_scale=1.0e6,                     # A/m2: x=J/J_scale
    independent_j_tolerance=0.1e6,    # A/m2: independent 10D perturbations
)


def _columns(n):
    return [f'X{i}' for i in range(1, n+1)] + ['Obj1', 'Obj2']


def run_nsga2(model):
    try:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.repair import Repair
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.termination import get_termination
        from pymoo.optimize import minimize
    except ImportError as exc:
        print('pymoo not installed; skipping NSGA-II:', exc)
        print('Install with: pip install pymoo')
        return None

    if REQUIRE_NEW_BASELINE:
        existing = [p for p in (BASELINE_ARCHIVE_CSV, BASELINE_PARETO_CSV)
                    if p.exists()]
        if existing:
            raise FileExistsError(
                'Clean independent baseline requires a fresh archive.\n'
                + '\n'.join(map(str, existing))
                + '\nMove/rename previous baseline files or change the filenames.'
            )

    class RoundingRepair(Repair):
        def _do(self, problem, X, **kwargs):
            return np.round(np.asarray(X, dtype=float), DECIMALS)

    class SolenoidProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=model.n_var, n_obj=2,
                             n_ieq_constr=0, n_eq_constr=0,
                             xl=model.lower, xu=model.upper, vtype=float)
            self.cache = {}

        def _evaluate(self, x, out, *args, **kwargs):
            x = np.round(np.asarray(x, dtype=float), DECIMALS)
            key = tuple(x.tolist())
            if key not in self.cache:
                objectives = model.evaluate(x)
                self.cache[key] = objectives
                pd.DataFrame([[*x, *objectives]],
                             columns=_columns(model.n_var)).to_csv(
                    BASELINE_ARCHIVE_CSV, mode='a',
                    header=not BASELINE_ARCHIVE_CSV.exists(), index=False,
                )
            out['F'] = self.cache[key]

    problem = SolenoidProblem()
    algorithm = NSGA2(
        pop_size=POP_SIZE, n_offsprings=N_OFFSPRINGS,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=SBX_PROB, eta=SBX_ETA),
        mutation=PM(prob=PM_PROB, eta=PM_ETA),
        repair=RoundingRepair(), eliminate_duplicates=True,
    )
    result = minimize(problem, algorithm, get_termination('n_gen', N_GEN),
                      seed=SEED, save_history=False, verbose=True)
    if result.X is None or result.F is None:
        raise RuntimeError('No feasible NSGA-II Pareto solutions returned.')
    pareto = pd.DataFrame(np.column_stack([result.X, result.F]),
                          columns=_columns(model.n_var))
    pareto.to_csv(BASELINE_PARETO_CSV, index=False)
    print('\n=== DI BARBA NSGA-II BASELINE ===')
    print('Pymoo function evaluations:', result.algorithm.evaluator.n_eval)
    print('Unique archive evaluations:', len(problem.cache))
    print('Final Pareto members:', len(pareto))
    print('Archive:', BASELINE_ARCHIVE_CSV)
    print('Pareto:', BASELINE_PARETO_CSV)
    return pareto


def save_figure(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES / (name + '.' + ext), dpi=220,
                    bbox_inches='tight')
    plt.close(fig)


def plot_front(reference, baseline):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(reference['Obj1'], reference['Obj2'], '-', lw=2,
            label='Analytical/convex reference')
    if baseline is not None:
        ax.scatter(baseline['Obj1'], baseline['Obj2'],
                   s=16, alpha=0.75, label='NSGA-II, final population')
    ax.set(xlabel='Objective 1: normalized field mean squared error',
           ylabel='Objective 2: normalized current-squared loss',
           title='Di Barba solenoid: Pareto front')
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, '01_pareto_reference_vs_nsga2')

    # Log x-axis reveals high-performance low-residual front near f1=0.
    fig, ax = plt.subplots(figsize=(7, 5))
    r = reference[reference['Obj1'] > 0]
    ax.plot(r['Obj1'], r['Obj2'], '-', lw=2,
            label='Analytical/convex reference')
    if baseline is not None:
        b = baseline[baseline['Obj1'] > 0]
        ax.scatter(b['Obj1'], b['Obj2'],
                   s=16, alpha=0.75, label='NSGA-II')
    ax.set_xscale('log')
    ax.set(xlabel='Objective 1: normalized field MSE (log)',
           ylabel='Objective 2: normalized current-squared loss',
           title='Pareto front: accurate-field region')
    ax.legend()
    ax.grid(alpha=0.2)
    save_figure(fig, '01b_pareto_log_obj1')


def plot_field_and_current(reference, model):
    # The minimum-Obj1 reference point corresponds to the left Pareto endpoint.
    best = reference.loc[reference['Obj1'].idxmin()]
    x = best[[f'X{i}' for i in range(1, model.n_var+1)]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    z_mm = model.z_samples * 1e3
    ax.plot(z_mm, model.field(x) * 1e3, lw=2, label='Best-field reference')
    ax.axhline(model.s.target_field*1e3, c='black', ls='--', label='Target')
    ax.set(xlabel='Axial position (mm)', ylabel='Bz (mT)',
           title='Axial magnetic induction: reference solution')
    ax.legend(); ax.grid(alpha=0.2)
    save_figure(fig, '02_optimal_axial_field')

    fig, ax = plt.subplots(figsize=(8, 4))
    centers = 0.5*(model.edges[:-1]+model.edges[1:]) * 1e3
    ax.bar(centers, model.full_winding_j(x)/1e6,
           width=0.85*(2*model.s.half_length/model.s.n_sections)*1e3)
    ax.set(xlabel='Axial winding-section center (mm)',
           ylabel='Current density (MA/m²)',
           title='20-section symmetric current-density distribution')
    ax.grid(alpha=0.2, axis='y')
    save_figure(fig, '03_optimal_current_density')

    f1, f2 = model.evaluate(x)
    print('\n=== MINIMUM FIELD-ERROR REFERENCE SOLUTION ===')
    print('Obj1:', f1, 'Obj2:', f2)
    print('Max absolute field deviation (mT):',
          np.max(abs(model.field(x)-model.s.target_field))*1e3)
    print('Indicative total copper loss (W):', model.ohmic_loss_watt(x))
    print('J in MA/m² (10 independent symmetric values):')
    print(np.array2string(model.current_density(x)/1e6, precision=6))


def plot_sensitivity(reference, model):
    # Exact tolerance box (not ALSF sampling): a *diagnostic*, not objective #3.
    # Evaluate a small uniform subset to keep the expensive corner checks cheap.
    n = min(24, len(reference))
    idx = np.unique(np.linspace(0, len(reference)-1, n, dtype=int))
    rows = []
    var_cols = [f'X{i}' for i in range(1, model.n_var+1)]
    for j in idx:
        r = reference.iloc[j]
        x = r[var_cols].to_numpy(dtype=float)
        s1, s2 = model.exact_box_sensitivity(x)
        rows.append([float(r['Obj1']), float(r['Obj2']), s1, s2])
    sensitivity = pd.DataFrame(rows, columns=['Obj1','Obj2','S1_box','S2_box'])
    sensitivity.to_csv(REFERENCE_SENSITIVITY_CSV, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(reference['Obj1'], reference['Obj2'], '-', c='0.7',
            label='Reference Pareto front')
    scat = ax.scatter(sensitivity['Obj1'], sensitivity['Obj2'],
                      c=sensitivity['S1_box'], s=45, cmap='viridis',
                      edgecolors='black', linewidths=0.3)
    bar=fig.colorbar(scat, ax=ax)
    bar.set_label('Exact worst-case |ΔObj1| under independent 10-D J errors')
    ax.set(xlabel='Normalized field MSE', ylabel='Normalized current-squared loss',
           title='Reference solutions: independently verified sensitivity')
    ax.grid(alpha=0.2)
    save_figure(fig, '04_exact_independent_10d_sensitivity')


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    model = DiBarbaSolenoid(SETTINGS)
    print('Di Barba-inspired symmetric solenoid: 20 sections, 10 currents')
    print('Model matrix shape:', model.A.shape)
    print('Current-density bounds (A/m2):', SETTINGS.j_min, SETTINGS.j_max)
    print('Reference solves are globally optimal weighted-sum subproblems.')
    reference = model.reference_pareto(n_weights=150)
    reference.to_csv(REFERENCE_CSV, index=False)
    print('Analytical reference CSV:', REFERENCE_CSV)
    print('Reference objective ranges:',
          (reference.Obj1.min(), reference.Obj1.max()),
          (reference.Obj2.min(), reference.Obj2.max()))
    pareto = run_nsga2(model) if RUN_NSGA2 else None
    plot_front(reference, pareto)
    plot_field_and_current(reference, model)
    plot_sensitivity(reference, model)
    print('Diagnostic box tolerance (A/m2):', SETTINGS.independent_j_tolerance)
    print('Sensitivity CSV:', REFERENCE_SENSITIVITY_CSV)
    print('Figures:', FIGURES)


if __name__ == '__main__':
    main()
