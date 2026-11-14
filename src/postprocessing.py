"""ALSF callback: Chebyshev DBSCAN plus local archive-pair evidence."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from pymoo.core.callback import Callback
from src.find_robust_clusters import flag_by_clusters
from src.local_variation import LocalVariationState


class FlagCallback(Callback):
    def __init__(self, archive_path, process_path, d_cluster,
                 t1, t2, min_samples, max_trim_fraction,
                 flag_start_gen, repair, tolerances=None,
                 objective_floors=(1e-3, 1e-3)):
        super().__init__()
        if tolerances is None:
            raise ValueError('Explicit tolerance vector required')
        self.archive_path = Path(archive_path)
        self.process_path = Path(process_path)
        self.d_cluster = float(d_cluster)
        self.t1, self.t2 = float(t1), float(t2)
        self.min_samples = int(min_samples)
        self.max_trim_fraction = float(max_trim_fraction) # legacy, disabled
        self.flag_start_gen = int(flag_start_gen)
        self.repair = repair
        self.tolerances = tolerances
        self.objective_floors = objective_floors
        self.state = LocalVariationState(tolerances, (t1,t2), objective_floors,
                                          radius=d_cluster)

    def notify(self, algorithm):
        gen = int(algorithm.n_gen)
        if gen < self.flag_start_gen or not self.archive_path.exists():
            return
        archive, stats = flag_by_clusters(
            str(self.archive_path), self.d_cluster, self.t1, self.t2,
            self.min_samples, self.max_trim_fraction,
            tolerances=self.tolerances, floors=self.objective_floors,
            state=self.state,
        )
        self.repair.update_flags(archive)
        regenerated, attempts = self.repair.pop_counters()
        flags = archive['flag'].value_counts()
        row = {
            'generation': gen,
            'n_archive': len(archive),
            'n_robust': int(flags.get('robust', 0)),
            'n_non_robust': int(flags.get('non-robust', 0)),
            'n_nei': int(flags.get('NEI', 0)),
            'n_robust_clusters': stats['n_robust_clusters'],
            'n_non_robust_clusters': stats['n_non_robust_clusters'],
            'n_verified_violations': stats['n_verified_violations'],
            'n_dbscan_noise': stats['n_noise'],
            'n_regenerated': int(regenerated),
            'n_attempts': int(attempts),
        }
        pd.DataFrame([row]).to_csv(self.process_path, mode='a',
                                   header=not self.process_path.exists(), index=False)
