"""
Sweep eta for FDC across all benchmark datasets to find the best achievable ARI.

This identifies which datasets are genuinely hard for FDC (no eta works)
vs. which just need a different eta than the default.

Usage:
    uv run python benchmarks/sweep_eta.py
"""

import sys
import os
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fdc import FDC
from benchmarks.benchmark_quality import load_datasets

warnings.filterwarnings("ignore")

ETA_VALUES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]


def sweep_eta(suite="all"):
    suites = load_datasets(suite)

    print(f"{'dataset':<20s} {'n':>5s} {'k':>3s} {'best_eta':>8s} "
          f"{'best_ARI':>8s} {'default(0.5)':>12s} {'verdict':<20s}")
    print("-" * 85)

    for suite_name, ds_dict in suites:
        for ds_name, (X, y_true) in ds_dict.items():
            short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
            n = X.shape[0]
            k = len(np.unique(y_true[y_true >= 0]))
            X_scaled = StandardScaler().fit_transform(X)

            best_ari = -1.0
            best_eta = 0.0
            default_ari = -1.0
            eta_aris = []

            for eta in ETA_VALUES:
                _real_stdout = sys.__stdout__
                sys.__stdout__ = open(os.devnull, "w")
                _cur_stdout = sys.stdout
                sys.stdout = sys.__stdout__
                try:
                    model = FDC(eta=eta, verbose=0, random_state=42)
                    model.fit(X_scaled)
                    labels = model.cluster_label
                finally:
                    sys.__stdout__.close()
                    sys.__stdout__ = _real_stdout
                    sys.stdout = _cur_stdout

                ari = adjusted_rand_score(y_true, labels)
                eta_aris.append((eta, ari))

                if ari > best_ari:
                    best_ari = ari
                    best_eta = eta

                if eta == 0.5:
                    default_ari = ari

            if best_ari >= 0.9:
                verdict = "OK (eta-sensitive)"
            elif best_ari >= 0.6:
                verdict = "PARTIAL"
            else:
                verdict = "HARD"

            print(f"{short:<20s} {n:>5d} {k:>3d} {best_eta:>8.2f} "
                  f"{best_ari:>8.3f} {default_ari:>12.3f} {verdict:<20s}")

            # Print the full eta curve for datasets that struggle
            if best_ari < 0.9:
                curve = "  eta curve: " + " ".join(
                    f"{eta:.2f}:{ari:.2f}" for eta, ari in eta_aris
                )
                print(curve)

    print()


if __name__ == "__main__":
    sweep_eta()
