"""
Scalability benchmark: runtime and peak memory vs dataset size.

Measures how FDC and competitors scale with increasing n on synthetic
Gaussian blobs in 2D.  Generates a log-log plot with O(n), O(n log n),
and O(n^2) reference lines.

Usage:
    uv run python benchmarks/benchmark_scaling.py [--no-plot] [--trials 3]
"""

import argparse
import gc
import sys
import os
import time
import tracemalloc
import warnings

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, MeanShift, estimate_bandwidth
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fdc import FDC

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_CLUSTERS = 10
N_FEATURES = 2
RANDOM_STATE = 42

ALGO_NAMES = ["FDC", "DBSCAN", "HDBSCAN", "OPTICS", "MeanShift"]

# OPTICS is O(n^2) and very slow past 20k; skip it for large n to keep
# the benchmark tractable.
OPTICS_MAX_N = 20_000


def make_algorithm(name, X):
    if name == "FDC":
        return FDC(eta=0.6, verbose=0, random_state=42)
    elif name == "DBSCAN":
        return DBSCAN(eps=0.3, min_samples=5)
    elif name == "HDBSCAN":
        return HDBSCAN(min_cluster_size=15)
    elif name == "OPTICS":
        return OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
    elif name == "MeanShift":
        bw = estimate_bandwidth(X, quantile=0.2, n_samples=min(500, len(X)))
        if bw == 0:
            bw = 0.5
        return MeanShift(bandwidth=bw, bin_seeding=True)


# ── Measurement helpers ──────────────────────────────────────────────────────

def measure_once(algo_name, X):
    """Return (wall_seconds, peak_memory_MiB) for fitting algo on X."""
    algo = make_algorithm(algo_name, X)

    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()
    try:
        # Suppress all stdout (FDC leaks prints even with verbose=0)
        _real_stdout = sys.__stdout__
        sys.__stdout__ = open(os.devnull, "w")
        _cur_stdout = sys.stdout
        sys.stdout = sys.__stdout__
        try:
            algo.fit(X)
        finally:
            sys.__stdout__.close()
            sys.__stdout__ = _real_stdout
            sys.stdout = _cur_stdout
        elapsed = time.perf_counter() - t0
    except Exception as e:
        elapsed = float("nan")
        print(f"    {algo_name} failed: {e}")

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mib = peak / (1024 * 1024)
    return elapsed, peak_mib


# ── Run benchmark ────────────────────────────────────────────────────────────

def run_benchmark(n_trials=3):
    """
    Returns:
        times[algo][size]  = median wall-clock seconds
        memory[algo][size] = median peak MiB
    """
    times = {a: {} for a in ALGO_NAMES}
    memory = {a: {} for a in ALGO_NAMES}

    for n in SIZES:
        print(f"\nn = {n:,}")
        X, _ = make_blobs(
            n_samples=n,
            n_features=N_FEATURES,
            centers=N_CLUSTERS,
            random_state=RANDOM_STATE,
        )
        X = StandardScaler().fit_transform(X)

        for algo_name in ALGO_NAMES:
            if algo_name == "OPTICS" and n > OPTICS_MAX_N:
                print(f"  {algo_name:>10s}: skipped (n > {OPTICS_MAX_N:,})")
                times[algo_name][n] = float("nan")
                memory[algo_name][n] = float("nan")
                continue

            trial_times = []
            trial_mem = []
            for t in range(n_trials):
                elapsed, peak_mib = measure_once(algo_name, X)
                trial_times.append(elapsed)
                trial_mem.append(peak_mib)

            med_time = float(np.nanmedian(trial_times))
            med_mem = float(np.nanmedian(trial_mem))
            times[algo_name][n] = med_time
            memory[algo_name][n] = med_mem
            print(f"  {algo_name:>10s}: {med_time:8.3f}s  {med_mem:8.1f} MiB")

    return times, memory


# ── Print tables ─────────────────────────────────────────────────────────────

def print_tables(times, memory):
    print("\n=== Runtime (seconds, median) ===\n")
    rows = []
    for n in SIZES:
        row = [f"{n:,}"]
        for a in ALGO_NAMES:
            v = times[a].get(n, float("nan"))
            row.append(f"{v:.3f}" if not np.isnan(v) else "—")
        rows.append(row)
    print(tabulate(rows, headers=["n"] + ALGO_NAMES, tablefmt="simple"))

    print("\n=== Peak Memory (MiB, median) ===\n")
    rows = []
    for n in SIZES:
        row = [f"{n:,}"]
        for a in ALGO_NAMES:
            v = memory[a].get(n, float("nan"))
            row.append(f"{v:.1f}" if not np.isnan(v) else "—")
        rows.append(row)
    print(tabulate(rows, headers=["n"] + ALGO_NAMES, tablefmt="simple"))
    print()


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_scaling(times, outfile="benchmark_scaling.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    markers = {"FDC": "o", "DBSCAN": "s", "HDBSCAN": "D", "OPTICS": "^", "MeanShift": "v"}
    colors = {"FDC": "#e41a1c", "DBSCAN": "#377eb8", "HDBSCAN": "#4daf4a",
              "OPTICS": "#984ea3", "MeanShift": "#ff7f00"}

    for algo_name in ALGO_NAMES:
        ns = sorted(times[algo_name].keys())
        ts = [times[algo_name][n] for n in ns]
        valid = [(n, t) for n, t in zip(ns, ts) if not np.isnan(t)]
        if not valid:
            continue
        vn, vt = zip(*valid)
        ax.plot(
            vn, vt,
            marker=markers[algo_name],
            color=colors[algo_name],
            label=algo_name,
            linewidth=2,
            markersize=6,
        )

    # Reference lines
    n_ref = np.array([SIZES[0], SIZES[-1]], dtype=float)
    t_base = 0.01  # anchor at ~10ms for n=1000

    ax.plot(n_ref, t_base * (n_ref / n_ref[0]),
            "--", color="gray", alpha=0.4, linewidth=1, label="O(n)")
    ax.plot(n_ref, t_base * (n_ref / n_ref[0]) * np.log2(n_ref) / np.log2(n_ref[0]),
            "-.", color="gray", alpha=0.4, linewidth=1, label="O(n log n)")
    ax.plot(n_ref, t_base * (n_ref / n_ref[0]) ** 2,
            ":", color="gray", alpha=0.4, linewidth=1, label=r"O(n$^2$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset size (n)", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Scaling Benchmark — 2D Gaussian Blobs (k=10)", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    outpath = os.path.join(os.path.dirname(__file__), outfile)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {outpath}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Clustering scalability benchmark")
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of trials per (algo, size) pair; reports median (default: 3)",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip generating the scaling plot",
    )
    args = parser.parse_args()

    print(f"Running scaling benchmark (trials={args.trials})...")
    times, memory = run_benchmark(n_trials=args.trials)
    print_tables(times, memory)

    if not args.no_plot:
        plot_scaling(times)


if __name__ == "__main__":
    main()
