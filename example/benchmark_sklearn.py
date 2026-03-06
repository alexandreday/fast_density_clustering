"""
Benchmark FDC against sklearn clustering algorithms on the standard
sklearn clustering comparison datasets.

Algorithms compared: FDC, DBSCAN, HDBSCAN, OPTICS, MeanShift
Metric: Adjusted Rand Index (ARI), runtime (seconds)
"""

import time
import warnings
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, MeanShift, estimate_bandwidth
from fdc import FDC

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Datasets (same as sklearn clustering comparison page, n=1500)
# ---------------------------------------------------------------------------
n_samples = 1500

noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons   = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs         = datasets.make_blobs(n_samples=n_samples, random_state=8)

X_b, y_b = datasets.make_blobs(n_samples=n_samples, random_state=170)
X_aniso   = np.dot(X_b, [[0.6, -0.6], [-0.4, 0.8]])
aniso     = (X_aniso, y_b)

varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)

no_structure = (np.random.rand(n_samples, 2), None)

dataset_names = ["circles", "moons", "varied", "aniso", "blobs", "no_structure"]
dataset_list  = [noisy_circles, noisy_moons, varied, aniso, blobs, no_structure]

# ---------------------------------------------------------------------------
# Algorithm factory — params tuned per dataset (same approach as sklearn page)
# ---------------------------------------------------------------------------
def make_algorithms(X):
    bw = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    return {
        "FDC":       FDC(eta=0.6, verbose=0, random_state=43),
        "DBSCAN":    DBSCAN(eps=0.3, min_samples=5),
        "HDBSCAN":   HDBSCAN(min_cluster_size=15),
        "OPTICS":    OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05),
        "MeanShift": MeanShift(bandwidth=bw, bin_seeding=True),
    }

# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------
algo_names = ["FDC", "DBSCAN", "HDBSCAN", "OPTICS", "MeanShift"]
algo_param_labels = {
    "FDC":       "η=0.6",
    "DBSCAN":    "ε=0.3, min_s=5",
    "HDBSCAN":   "min_cs=15",
    "OPTICS":    "min_s=10, ξ=0.05",
    "MeanShift": "bw=auto",
}

# results[dataset][algo] = (ari, runtime)
# pred_labels[dataset][algo] = predicted labels (reused for plotting)
results = {d: {} for d in dataset_names}
pred_labels = {d: {} for d in dataset_names}

for ds_name, (X, y_true) in zip(dataset_names, dataset_list):
    X = StandardScaler().fit_transform(X)
    algos = make_algorithms(X)

    for algo_name in algo_names:
        algo = algos[algo_name]
        t0 = time.time()
        algo.fit(X)
        dt = time.time() - t0

        labels = algo.cluster_label if hasattr(algo, "cluster_label") else algo.labels_
        pred_labels[ds_name][algo_name] = labels

        if y_true is None:
            ari = float("nan")
        else:
            ari = adjusted_rand_score(y_true, labels)

        results[ds_name][algo_name] = (ari, dt)

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
col_w = 14

header = f"{'dataset':<14}" + "".join(f"{a:^{col_w}}" for a in algo_names)
sep    = "-" * len(header)

print("\n=== Adjusted Rand Index (ARI) — higher is better ===")
print(sep)
print(header)
print(sep)
for ds_name in dataset_names:
    row = f"{ds_name:<14}"
    for algo_name in algo_names:
        ari, _ = results[ds_name][algo_name]
        cell = f"{ari:.3f}" if not np.isnan(ari) else "  n/a "
        row += f"{cell:^{col_w}}"
    print(row)
print(sep)

print("\n=== Runtime (seconds) ===")
print(sep)
print(header)
print(sep)
for ds_name in dataset_names:
    row = f"{ds_name:<14}"
    for algo_name in algo_names:
        _, dt = results[ds_name][algo_name]
        row += f"{dt:^{col_w}.3f}"
    print(row)
print(sep)
print()

# ---------------------------------------------------------------------------
# Figure: scatter plots (rows=datasets, cols=algorithms), ARI + time in title
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = plt.cm.tab10
n_ds  = len(dataset_names)
n_alg = len(algo_names)

fig, axes = plt.subplots(n_ds, n_alg, figsize=(n_alg * 2.8, n_ds * 2.2))
fig.subplots_adjust(hspace=0.45, wspace=0.15)

# Reuse stored labels from the first run (no re-fitting needed)
for row, (ds_name, (X_raw, y_true)) in enumerate(zip(dataset_names, dataset_list)):
    X = StandardScaler().fit_transform(X_raw)

    for col, algo_name in enumerate(algo_names):
        ax = axes[row][col]
        labels = pred_labels[ds_name][algo_name]

        ari, dt = results[ds_name][algo_name]

        # color by predicted label; noise (-1) in gray
        unique = np.unique(labels)
        colors = [cmap(i / max(len(unique), 1)) if lbl != -1 else (0.7, 0.7, 0.7, 1)
                  for i, lbl in enumerate(labels)]
        color_map = {lbl: cmap(i / max(len(unique) - 1, 1)) if lbl != -1 else (0.7, 0.7, 0.7, 1)
                     for i, lbl in enumerate(unique)}
        point_colors = [color_map[lbl] for lbl in labels]

        ax.scatter(X[:, 0], X[:, 1], c=point_colors, s=4, linewidths=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ari_str = f"ARI={ari:.2f}" if not np.isnan(ari) else "ARI=n/a"
        ax.set_title(f"{ari_str}\n{dt:.2f}s", fontsize=8)

        if row == 0:
            param_str = algo_param_labels[algo_name]
            ax.set_xlabel(f"{algo_name}\n{param_str}", fontsize=9, fontweight="bold")
            ax.xaxis.set_label_position("top")
        if col == 0:
            ax.set_ylabel(ds_name, fontsize=9, rotation=90, labelpad=4)

outfile = os.path.join(os.path.dirname(__file__), "benchmark_sklearn.png")
fig.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Plot saved to {outfile}")
