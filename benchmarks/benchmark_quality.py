"""
Clustering quality benchmark: FDC vs density-based competitors.

Evaluates ARI and AMI across three dataset suites:
  - sklearn toy datasets (circles, moons, blobs, aniso, varied)
  - SIPU shape datasets (Jain, Compound, Aggregation, Pathbased, Spiral, Flame, D31, R15, Unbalance)
  - FCPS datasets (Atom, Chainlink, Lsun, Target, TwoDiamonds, WingNut, Hepta)

Algorithms: FDC, DBSCAN, HDBSCAN, OPTICS, MeanShift

Usage:
    uv run python benchmarks/benchmark_quality.py [--no-plot] [--suite sklearn|sipu|fcps|all]
"""

import argparse
import sys
import os
import time
import warnings

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, MeanShift, estimate_bandwidth
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fdc import FDC

warnings.filterwarnings("ignore")

# ── GitHub URL for clustering-benchmarks data ────────────────────────────────
CLUSTBENCH_URL = "https://github.com/gagolews/clustering-data-v1/raw/v1.1.0"

# ── Dataset definitions ──────────────────────────────────────────────────────

def _load_sklearn_datasets(n_samples=1500):
    """Sklearn toy datasets (same as the sklearn clustering comparison page)."""
    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=42
    )
    noisy_moons = datasets.make_moons(
        n_samples=n_samples, noise=0.05, random_state=42
    )
    blobs = datasets.make_blobs(
        n_samples=n_samples, random_state=8
    )
    X_b, y_b = datasets.make_blobs(
        n_samples=n_samples, random_state=170
    )
    X_aniso = np.dot(X_b, [[0.6, -0.6], [-0.4, 0.8]])
    aniso = (X_aniso, y_b)
    varied = datasets.make_blobs(
        n_samples=n_samples,
        cluster_std=[1.0, 2.5, 0.5],
        random_state=170,
    )
    return {
        "circles": noisy_circles,
        "moons": noisy_moons,
        "blobs": blobs,
        "aniso": aniso,
        "varied": varied,
    }


def _load_clustbench(battery, dataset_names):
    """Load datasets from the Gagolewski clustering-benchmarks suite."""
    import clustbench

    loaded = {}
    for name in dataset_names:
        try:
            d = clustbench.load_dataset(battery, name, url=CLUSTBENCH_URL)
            loaded[f"{battery}/{name}"] = (d.data, d.labels[0])
        except Exception as e:
            print(f"  WARNING: could not load {battery}/{name}: {e}")
    return loaded


def _load_sipu_datasets():
    return _load_clustbench(
        "sipu",
        [
            "jain", "compound", "aggregation", "pathbased", "spiral",
            "flame", "d31", "r15", "unbalance",
        ],
    )


def _load_fcps_datasets():
    return _load_clustbench(
        "fcps",
        [
            "atom", "chainlink", "lsun", "target",
            "twodiamonds", "wingnut", "hepta",
        ],
    )


# Returns list of (suite_name, datasets_dict) to preserve suite grouping.
def load_datasets(suite="all"):
    suites = []
    if suite in ("all", "sklearn"):
        suites.append(("sklearn", _load_sklearn_datasets()))
    if suite in ("all", "sipu"):
        suites.append(("sipu", _load_sipu_datasets()))
    if suite in ("all", "fcps"):
        suites.append(("fcps", _load_fcps_datasets()))
    return suites


# ── Algorithm factory ────────────────────────────────────────────────────────

ALGO_NAMES = ["FDC", "DBSCAN", "HDBSCAN", "OPTICS", "MeanShift"]


def make_algorithms(X):
    bw = estimate_bandwidth(X, quantile=0.2, n_samples=min(500, len(X)))
    if bw == 0:
        bw = 0.5
    return {
        "FDC": FDC(eta=0.6, verbose=0, random_state=42),
        "DBSCAN": DBSCAN(eps=0.3, min_samples=5),
        "HDBSCAN": HDBSCAN(min_cluster_size=15),
        "OPTICS": OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05),
        "MeanShift": MeanShift(bandwidth=bw, bin_seeding=True),
    }


# ── Run benchmark ────────────────────────────────────────────────────────────

def run_benchmark(suite="all"):
    suites = load_datasets(suite)
    if not suites:
        print("No datasets loaded.")
        return [], {}, {}

    # results[ds_name][algo] = {"ari", "ami", "time", "n", "d", "k"}
    results = {}
    pred_labels = {}
    scaled_data = {}

    for suite_name, ds_dict in suites:
        for ds_name, (X, y_true) in ds_dict.items():
            n, d = X.shape
            k = len(np.unique(y_true[y_true >= 0]))  # exclude noise label -1

            X_scaled = StandardScaler().fit_transform(X)
            scaled_data[ds_name] = (X_scaled, y_true)
            algos = make_algorithms(X_scaled)
            results[ds_name] = {}
            pred_labels[ds_name] = {}

            for algo_name in ALGO_NAMES:
                algo = algos[algo_name]
                t0 = time.perf_counter()
                try:
                    # FDC's enablePrint() restores sys.__stdout__ directly,
                    # bypassing contextlib.redirect_stdout. Swap __stdout__
                    # to fully suppress output.
                    _real_stdout = sys.__stdout__
                    sys.__stdout__ = open(os.devnull, "w")
                    _cur_stdout = sys.stdout
                    sys.stdout = sys.__stdout__
                    try:
                        algo.fit(X_scaled)
                    finally:
                        sys.__stdout__.close()
                        sys.__stdout__ = _real_stdout
                        sys.stdout = _cur_stdout
                    dt = time.perf_counter() - t0
                    labels = (
                        algo.cluster_label
                        if hasattr(algo, "cluster_label")
                        else algo.labels_
                    )
                except Exception as e:
                    print(f"  {algo_name} failed on {ds_name}: {e}")
                    dt = float("nan")
                    labels = np.full(len(X_scaled), -1)

                pred_labels[ds_name][algo_name] = labels
                results[ds_name][algo_name] = {
                    "ari": adjusted_rand_score(y_true, labels),
                    "ami": adjusted_mutual_info_score(y_true, labels),
                    "time": dt,
                    "n": n,
                    "d": d,
                    "k": k,
                }

    return suites, results, pred_labels, scaled_data


# ── Formatting helpers ───────────────────────────────────────────────────────

def _fmt_score(val, is_best):
    """Format a score value, marking the best in each row with *."""
    s = f"{val:.3f}"
    return f"{s}*" if is_best else f"{s} "


def _fmt_time(val):
    if np.isnan(val):
        return "FAIL"
    return f"{val:.3f}"


# ── Print results ────────────────────────────────────────────────────────────

def print_results(suites, results):
    if not results:
        return

    # ── Combined ARI + AMI table ─────────────────────────────────────────
    print()
    print("=" * 80)
    print("  CLUSTERING QUALITY BENCHMARK")
    print("  Metrics: ARI (Adjusted Rand Index), AMI (Adjusted Mutual Information)")
    print("  Scoring: higher = better, 1.0 = perfect, 0.0 = random  |  * = best in row")
    print("=" * 80)

    for suite_name, ds_dict in suites:
        ds_names = [n for n in ds_dict if n in results]
        if not ds_names:
            continue

        # Suite header
        print(f"\n── {suite_name.upper()} ", end="")
        print("─" * (76 - len(suite_name)))

        # Build rows
        headers = ["dataset", "n", "d", "k"]
        for algo in ALGO_NAMES:
            headers.extend([f"{algo}", ""])
        # Sub-headers row
        sub_headers = ["", "", "", ""]
        for algo in ALGO_NAMES:
            sub_headers.extend(["ARI", "AMI"])

        rows = []
        for ds_name in ds_names:
            r = results[ds_name]
            meta = r[ALGO_NAMES[0]]  # all algos have same n/d/k
            short_name = ds_name.split("/")[-1] if "/" in ds_name else ds_name

            # Find best ARI and AMI for this dataset
            aris = {a: r[a]["ari"] for a in ALGO_NAMES}
            amis = {a: r[a]["ami"] for a in ALGO_NAMES}
            best_ari = max(aris.values())
            best_ami = max(amis.values())

            row = [short_name, meta["n"], meta["d"], meta["k"]]
            for algo in ALGO_NAMES:
                row.append(_fmt_score(aris[algo], aris[algo] == best_ari))
                row.append(_fmt_score(amis[algo], amis[algo] == best_ami))
            rows.append(row)

        # Per-suite mean row
        sep_row = ["", "", "", ""] + ["─────"] * (len(ALGO_NAMES) * 2)
        rows.append(sep_row)
        mean_row = ["MEAN", "", "", ""]
        for algo in ALGO_NAMES:
            mean_ari = np.mean([results[ds][algo]["ari"] for ds in ds_names])
            mean_ami = np.mean([results[ds][algo]["ami"] for ds in ds_names])
            mean_row.append(f"{mean_ari:.3f} ")
            mean_row.append(f"{mean_ami:.3f} ")
        rows.append(mean_row)

        print(tabulate(
            [sub_headers] + rows,
            headers=headers,
            tablefmt="simple",
            stralign="right",
            numalign="right",
        ))

    # ── Overall summary across all suites ────────────────────────────────
    all_ds = list(results.keys())
    if len(all_ds) > 5:  # only show if multiple suites
        print(f"\n── OVERALL ", end="")
        print("─" * 68)
        headers = [""] + ALGO_NAMES
        ari_row = ["Mean ARI"]
        ami_row = ["Mean AMI"]
        for algo in ALGO_NAMES:
            ari_row.append(f"{np.mean([results[ds][algo]['ari'] for ds in all_ds]):.3f}")
            ami_row.append(f"{np.mean([results[ds][algo]['ami'] for ds in all_ds]):.3f}")
        print(tabulate([ari_row, ami_row], headers=headers, tablefmt="simple"))

    # ── Runtime table ────────────────────────────────────────────────────
    print(f"\n── RUNTIME (seconds) ", end="")
    print("─" * 59)

    headers = ["dataset", "n"] + ALGO_NAMES
    rows = []
    for suite_name, ds_dict in suites:
        ds_names = [n for n in ds_dict if n in results]
        for ds_name in ds_names:
            r = results[ds_name]
            short_name = ds_name.split("/")[-1] if "/" in ds_name else ds_name
            n = r[ALGO_NAMES[0]]["n"]
            row = [short_name, n]
            for algo in ALGO_NAMES:
                row.append(_fmt_time(r[algo]["time"]))
            rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="simple", numalign="right"))
    print()


# ── Plot results ─────────────────────────────────────────────────────────────

def plot_results(suites, results, pred_labels, scaled_data,
                 outfile="benchmark_quality.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds_names = list(results.keys())
    n_ds = len(ds_names)
    n_alg = len(ALGO_NAMES)

    fig, axes = plt.subplots(
        n_ds, n_alg, figsize=(n_alg * 2.6, n_ds * 2.2),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.50, wspace=0.15)
    cmap = plt.cm.tab10

    for row, ds_name in enumerate(ds_names):
        X, y_true = scaled_data[ds_name]
        for col, algo_name in enumerate(ALGO_NAMES):
            ax = axes[row][col]
            labels = pred_labels[ds_name][algo_name]
            ari = results[ds_name][algo_name]["ari"]

            unique = np.unique(labels)
            n_unique = max(len(unique) - 1, 1)
            color_map = {}
            ci = 0
            for lbl in unique:
                if lbl == -1:
                    color_map[lbl] = (0.7, 0.7, 0.7, 1)
                else:
                    color_map[lbl] = cmap(ci / n_unique)
                    ci += 1
            colors = [color_map[lbl] for lbl in labels]

            ax.scatter(X[:, 0], X[:, 1], c=colors, s=3, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"ARI={ari:.2f}", fontsize=7)

            if row == 0:
                ax.set_xlabel(algo_name, fontsize=9, fontweight="bold")
                ax.xaxis.set_label_position("top")
            if col == 0:
                short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
                ax.set_ylabel(short, fontsize=7, rotation=90, labelpad=4)

    outpath = os.path.join(os.path.dirname(__file__), outfile)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {outpath}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clustering quality benchmark")
    parser.add_argument(
        "--suite",
        choices=["sklearn", "sipu", "fcps", "all"],
        default="all",
        help="Which dataset suite to run (default: all)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating the scatter-plot figure",
    )
    args = parser.parse_args()

    print(f"Running quality benchmark (suite={args.suite})...")
    suites, results, pred_labels, scaled_data = run_benchmark(args.suite)
    print_results(suites, results)

    if not args.no_plot and results:
        plot_results(suites, results, pred_labels, scaled_data)


if __name__ == "__main__":
    main()
