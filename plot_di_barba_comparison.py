"""Overlay the independent reference, baseline NSGA-II, and ALSF fronts.

Run from PyCharm directly after run_di_barba_baseline.py and
run_di_barba_alsf.py. Missing optional result CSVs are simply omitted.
Both axes use log scale, with distinct MAJOR and MINOR grids, as requested.

The normalized axes are scaled using REFERENCE maxima ONLY, so comparisons
are invariant to the choice of ALFS/baseline run.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures' / 'di_barba'
REFERENCE = RESULTS / 'pareto_di_barba_reference.csv'
BASELINE = RESULTS / 'pareto_di_barba_baseline.csv'
ALSF = RESULTS / 'pareto_di_barba_alsf_10d.csv'
PROCESS = RESULTS / 'process_di_barba_alsf_10d.csv'


def read(path):
    if not path.exists():
        print('Not found; skipped:', path.name)
        return None
    frame = pd.read_csv(path)
    if not {'Obj1', 'Obj2'}.issubset(frame.columns):
        raise ValueError(f'Missing Obj1/Obj2 columns in {path}')
    print('Loaded:', path.name, 'rows:', len(frame))
    return frame


def positive_xy(frame, scale):
    array = frame[['Obj1', 'Obj2']].to_numpy(dtype=float)
    mask = np.isfinite(array).all(axis=1) & (array > 0).all(axis=1)
    return (array[mask] / scale), mask


def set_log_grid(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, numticks=20))
        axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=200))
        axis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', linewidth=0.78, alpha=0.42, linestyle='-', color='#8492a1')
    ax.grid(True, which='minor', linewidth=0.40, alpha=0.25, linestyle=':', color='#aab5c0')
    ax.tick_params(which='major', length=5.5, width=0.85)
    ax.tick_params(which='minor', length=2.5, width=0.5)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel(r'Normalized magnetic-field residual, $f_1$ (MSE)')
    ax.set_ylabel(r'Normalized winding-loss proxy, $f_2$ ($\propto J^2$)')


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ('png','pdf'):
        path = FIGURES / f'{name}.{ext}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        print('Saved:', path)
    plt.close(fig)


def main():
    reference = read(REFERENCE)
    if reference is None:
        raise FileNotFoundError('Missing reference. Run run_di_barba_baseline.py first.')
    baseline, alsf = read(BASELINE), read(ALSF)
    scale = reference[['Obj1','Obj2']].max().to_numpy(dtype=float)
    if np.any(scale <= 0):
        raise ValueError('Reference must have positive maxima')

    fig, ax = plt.subplots(figsize=(9.0, 6.3))
    xy, mask = positive_xy(reference, scale)
    xy = xy[np.argsort(xy[:,0])]
    ax.plot(xy[:,0], xy[:,1], lw=2.25, color='#1769aa',
            label='Independent convex reference', zorder=2)
    for frame, label, color, marker, size in (
        (baseline, 'NSGA-II baseline', '#d97706', 'o', 19),
        (alsf, 'ALSF + NSGA-II', '#0e8468', 'x', 31),
    ):
        if frame is None:
            continue
        points, valid = positive_xy(frame, scale)
        ax.scatter(points[:,0], points[:,1], marker=marker, s=size,
                   linewidth=1.0, alpha=0.8, color=color, label=label, zorder=3)
        print(f'{label}: {len(points)} positive/finite points; '
              f'{len(frame)-len(points)} zero or invalid points omitted from log plot')
    set_log_grid(ax)
    ax.set_title('Solenoid optimization: reference vs. NSGA-II vs. ALSF', pad=13)
    ax.legend(loc='upper right', frameon=False)
    save(fig, '05_pareto_reference_baseline_alsf_loglog')

    # Optional second view: final ALSF solutions colored by cluster flag.
    if alsf is not None and 'flag' in alsf.columns:
        fig, ax = plt.subplots(figsize=(9.0, 6.3))
        ax.plot(xy[:,0], xy[:,1], lw=2.1, color='#1769aa',
                label='Independent convex reference')
        flag_styles = {
            'robust': ('#1a936f', 'o', 'Stable-labelled'),
            'non-robust': ('#df7c17', 'x', 'Non-robust-labelled'),
            'NEI': ('#8c93a3', '^', 'Not sufficiently explored'),
        }
        for flag, (color, marker, label) in flag_styles.items():
            subset = alsf.loc[alsf['flag'].fillna('NEI') == flag]
            if subset.empty:
                continue
            coords, _ = positive_xy(subset, scale)
            ax.scatter(coords[:,0], coords[:,1], color=color, marker=marker,
                       s=27, alpha=.78, label=f'{label} (n={len(coords)})', zorder=3)
        set_log_grid(ax)
        ax.set_title('ALSF final Pareto set: archive-cluster classification', pad=13)
        ax.legend(loc='upper right', frameon=False)
        save(fig, '06_alsf_pareto_cluster_flags_loglog')

    process = read_process(PROCESS)
    if process is not None:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        g = process['generation'].to_numpy()
        if 'n_regenerated' in process:
            ax.plot(g, process['n_regenerated'], label='Regenerated offspring / generation')
        if 'n_non_robust_clusters' in process:
            ax.plot(g, process['n_non_robust_clusters'], label='Non-robust clusters')
        if 'n_robust_clusters' in process:
            ax.plot(g, process['n_robust_clusters'], label='Robust clusters')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Count')
        ax.set_title('ALSF evolution: classification and offspring regeneration')
        ax.grid(which='major', linewidth=.7, alpha=.3)
        ax.legend(frameon=False)
        save(fig, '07_alsf_process')


def read_process(path):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if 'generation' not in frame:
        print('Process CSV lacks generation; skipping process plot')
        return None
    return frame


if __name__ == '__main__':
    main()
