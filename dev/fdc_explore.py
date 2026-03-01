"""
Explore FDC on individual sklearn benchmark datasets.

Usage
-----
    uv run python example/fdc_explore.py --dataset blobs
    uv run python example/fdc_explore.py --dataset circles --eta 0.05 --n-samples 2000
    uv run python example/fdc_explore.py --dataset moons --bandwidth 0.1

Datasets: circles, moons, varied, aniso, blobs, no_structure
"""

import argparse
import os
import sys

import numpy as np
from sklearn import datasets
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fdc import FDC, plotting

# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

DATASET_NAMES = ["circles", "moons", "varied", "aniso", "blobs", "no_structure"]


def load_dataset(name: str, n_samples: int, random_state: int):
    rng = np.random.default_rng(random_state)
    np_seed = int(rng.integers(0, 2**31))
    np.random.seed(np_seed)

    if name == "circles":
        return datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    elif name == "moons":
        return datasets.make_moons(n_samples=n_samples, noise=0.05)
    elif name == "blobs":
        return datasets.make_blobs(n_samples=n_samples, random_state=np_seed)
    elif name == "varied":
        return datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=np_seed)
    elif name == "aniso":
        X_b, y_b = datasets.make_blobs(n_samples=n_samples, random_state=np_seed)
        X_aniso = np.dot(X_b, [[0.6, -0.6], [-0.4, 0.8]])
        return X_aniso, y_b
    elif name == "no_structure":
        return np.random.rand(n_samples, 2), None
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {DATASET_NAMES}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Explore FDC on a benchmark dataset.")
    p.add_argument(
        "--dataset", "-d",
        default="blobs",
        choices=DATASET_NAMES,
        help="Which benchmark dataset to use (default: blobs)",
    )
    p.add_argument(
        "--n-samples", "-n",
        type=int,
        default=1500,
        help="Number of data points (default: 1500)",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="FDC noise threshold for merging (default: 0.1)",
    )
    p.add_argument(
        "--bandwidth",
        type=float,
        default=None,
        help="KDE bandwidth — omit for auto-selection",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--save", "-s",
        default=None,
        help="Save figure to this path (e.g. out.png). Defaults to <dataset>.png",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the interactive plot window",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\n=== FDC on '{args.dataset}' (n={args.n_samples}) ===")

    X_raw, y_true = load_dataset(args.dataset, args.n_samples, args.random_state)
    X = StandardScaler().fit_transform(X_raw)

    model = FDC(eta=args.eta, bandwidth=args.bandwidth, verbose=1)
    model.fit(X)

    n_clusters = len(np.unique(model.cluster_label))
    print(f"\nClusters found : {n_clusters}")

    if y_true is not None:
        ari = adjusted_rand_score(y_true, model.cluster_label)
        print(f"ARI            : {ari:.4f}")
    else:
        print("ARI            : n/a (no ground truth)")

    savefile = args.save or os.path.join(
        os.path.dirname(__file__), f"fdc_{args.dataset}.png"
    )

    plotting.set_nice_font()
    plotting.summary_model(
        model,
        ytrue=y_true,
        show=not args.no_show,
        savefile=savefile,
    )
    print(f"Plot saved to  : {savefile}\n")


if __name__ == "__main__":
    main()
