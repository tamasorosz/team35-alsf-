# ESCO2026-ALSF — ten-dimensional analytical solenoid

This repository contains a reproducible **Di Barba (2016)-inspired** solenoid benchmark and an experimental implementation of Archive-Based Local Sensitivity Filtering (ALSF), including the changes suggested in Pavel Solin's November 2026 review.

> **Research caveat:** The benchmark uses explicitly defined current-density bounds and normalized objective functions; it is **not** claimed to reproduce Di Barba's original Figure 1 exactly. ALSF flags represent empirical local archive evidence, not guaranteed worst-case sensitivity. This project is public; do not upload unpublished confidential material or private data.

## Run locally (Linux/PyCharm)

Use Python 3.11+ and run from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest
python -m pytest -q
python quick_check.py
python run_di_barba_baseline.py
python run_di_barba_alsf.py
python plot_di_barba_comparison.py
```

For Windows, activate with `.venv\\Scripts\\activate`. Full NSGA-II and ALSF runs use a population of 100, 200 generations, and seed 42. Outputs live in `results/` and `figures/di_barba/` (ignored by Git). Rename existing `archive_...` and `pareto_...` CSVs before rerunning: the scripts deliberately refuse to overwrite them.

For a **fast integration check**, before launching ALSF run:

```bash
ALSF_POP_SIZE=24 ALSF_OFFSPRINGS=24 ALSF_N_GEN=8 ALSF_FLAG_START_GEN=3 ALSF_RESULTS_DIR=results/smoke python run_di_barba_alsf.py
```

The `results/smoke` directory will contain the short-run archive, cluster flags, process statistics, and final Pareto-front CSV. For 200-generation runs, omit these environment overrides.

## Run in GitHub Actions without a local Python setup

Go to the **Actions** tab → **ALSF tests and smoke run** → **Run workflow**. The workflow installs `pymoo` and other requirements on GitHub, runs unit tests and a short NSGA-II+ALSF integration smoke run, then uploads a downloadable `alsf-smoke-results` artifact. Branch pushes and pull requests also run it automatically. This smoke run is **not** a 200-generation benchmark.

## Pavel-review modifications (experimental branch)

| Component | Previous | This version |
|---|---|---|
| Distance | Euclidean norm of raw `X1..X10` | `d_inf(x,y) = max_i abs(x_i-y_i)/Delta_i` |
| DBSCAN | `metric='euclidean'`, ad hoc `eps` | `metric='chebyshev'`, `eps=1.0`, `min_samples=10`, on tolerance-normalized design variables |
| Stability assessment | Entire chained cluster vs one medoid | Direct pairs of evaluated designs within `d_inf<=1` |
| Greedy trimming | Three-pass DBSCAN + medoid trimming | **Removed**: cannot hide an observed high-sensitivity boundary |
| Offspring filter | Reject only with 3 nearby non-robust cluster neighbors | Reject if within tolerance box of **any** archived verified violation endpoint |
| Query speed | Repeated pairwise cluster distances | `cKDTree` (L∞) for pair checks and buffered offspring queries; archive pairs checked only once as new points arrive |

For the current model, `x_i = J_i/(1 MA/m²)` and independent `J_i` tolerance is `±0.1 MA/m²`; thus each `Delta_i=0.1` in normalized coordinates. Both DBSCAN and the offspring-buffer radius are **one tolerance box** (`eps=1`). The `D_CLUSTER` and `D_REPAIR` settings are multipliers of this normalized distance, **not** physical meters.

The deviation of a local pair `x,y` in objective `j` is

```
r_j(x,y) = |f_j(x)-f_j(y)| / max(|f_j(x)|, |f_j(y)|, floor_j).
```

Current experimental parameters: `T1=T2=0.1`, fixed `floor_1=floor_2=1e-3`, `MIN_SAMPLES=10`, `FLAG_START_GEN=50`, `MAX_RESAMPLE_ATTEMPTS=100`. These objective floors are explicit numerical choices introduced to avoid unstable relative ratios near zero. They are **not** a mathematically unique definition of physical robustness; sensitivity to their values needs separate experimental validation.

Points classified `non-robust` are observed endpoints of at least one violating pair, **including DBSCAN noise points**. `robust` means no *observed* violating pair in a sufficiently sampled local box. `NEI` means locally under-sampled and without observed violations. A DBSCAN cluster can contain different point flags; full cluster chaining does not determine rejection. Columns `R1_local`, `R2_local`, `verified_violation`, `local_n`, `cluster_id`, `is_medoid` are written to the archive. `is_medoid` is an approximate representative for plotting only; classification no longer relies on it.

`OffspringRepair` performs uniform resampling in the full allowed design box. If it cannot find a non-forbidden sample after the configured attempts, it **raises a descriptive error** instead of silently accepting a forbidden offspring. This can be conservative and may harm nominal HV; compare independent sensitivity and coverage, not HV alone.

## Reproducibility and limitations

- The same objective functions and bounds are used for baseline and ALSF, but their random streams diverge after repairs.
- The convex weighted-sum reference CSV gives globally solved points for the **defined** quadratic benchmark.
- The classifier's `verified_violation` is *archive-based evidence*, not an exact worst-case result over every tolerance box. True independent 10D worst-case values can be computed with `DiBarbaSolenoid.exact_box_sensitivity` as a separate diagnostic.
- The geometry of tolerance boxes can leave many NEI samples in ten dimensions even with `L_inf` DBSCAN; count NEI and runtime explicitly.
- A full 200-generation run has not yet been executed in this chat environment. GitHub Actions provides an integration smoke check.
