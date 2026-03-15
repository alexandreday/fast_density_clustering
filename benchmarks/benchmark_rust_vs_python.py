"""
Benchmark: FDC (Rust & Python) vs C-backed DBSCAN & HDBSCAN.

Compares the Rust-accelerated FDC against sklearn's Cython/C implementations
of DBSCAN and HDBSCAN, plus the pure-Python FDC fallback.

sklearn's DBSCAN uses a Cython KD-tree + C distance computations.
sklearn's HDBSCAN uses Cython MST construction + C linkage extraction.

Usage:
    uv run python benchmarks/benchmark_rust_vs_python.py [--trials 3] [--no-plot]
"""

import argparse
import gc
import os
import sys
import time
import warnings

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, estimate_bandwidth, MeanShift
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

SIZES = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_CLUSTERS = 10
N_FEATURES = 2
RANDOM_STATE = 42

# Algorithms to benchmark (order matters for display)
ALGO_NAMES = ["FDC (Rust)", "FDC (Python)", "DBSCAN (C)", "HDBSCAN (C)"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _suppress_stdout():
    old_stdout = sys.stdout
    old_real = sys.__stdout__
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.__stdout__ = devnull
    return old_stdout, old_real, devnull


def _restore_stdout(old_stdout, old_real, devnull):
    devnull.close()
    sys.__stdout__ = old_real
    sys.stdout = old_stdout


def _time_fdc(use_rust, X, n_trials):
    """Time FDC.fit() with or without Rust backend."""
    import fdc.fdc as fdc_mod
    import fdc.density_estimation as kde_mod

    original_fdc = fdc_mod._HAS_RUST
    original_kde = kde_mod._HAS_RUST
    fdc_mod._HAS_RUST = use_rust
    kde_mod._HAS_RUST = use_rust

    times = []
    for _ in range(n_trials):
        from fdc import FDC
        model = FDC(eta=0.5, verbose=0, random_state=42)
        gc.collect()
        old_stdout, old_real, devnull = _suppress_stdout()
        try:
            t0 = time.perf_counter()
            model.fit(X)
            elapsed = time.perf_counter() - t0
        finally:
            _restore_stdout(old_stdout, old_real, devnull)
        times.append(elapsed)

    fdc_mod._HAS_RUST = original_fdc
    kde_mod._HAS_RUST = original_kde
    return float(np.median(times))


def _time_sklearn(algo_cls, X, n_trials, **kwargs):
    """Time a sklearn algorithm's fit()."""
    times = []
    for _ in range(n_trials):
        model = algo_cls(**kwargs)
        gc.collect()
        t0 = time.perf_counter()
        model.fit(X)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return float(np.median(times))


# ── Run benchmark ────────────────────────────────────────────────────────────

def run_benchmark(n_trials=3):
    try:
        import fdc_rs
        has_rust = True
    except ImportError:
        has_rust = False
        print("WARNING: fdc_rs not installed. FDC (Rust) will be skipped.")

    # Results: {algo_name: {n: time}}
    times = {a: {} for a in ALGO_NAMES}

    print(f"\nBenchmark: FDC (Rust/Python) vs C-backed DBSCAN/HDBSCAN (trials={n_trials})")
    print("=" * 80)
    print(f"  sklearn DBSCAN:  Cython KD-tree + C distance kernel")
    print(f"  sklearn HDBSCAN: Cython MST + C single-linkage extraction")
    print(f"  FDC (Rust):      kiddo KD-tree + rayon parallel + Rust core")
    print(f"  FDC (Python):    sklearn KD-tree + numpy vectorized")
    print("=" * 80)

    for n in SIZES:
        X, _ = make_blobs(
            n_samples=n, n_features=N_FEATURES,
            centers=N_CLUSTERS, random_state=RANDOM_STATE,
        )
        X = StandardScaler().fit_transform(X)

        print(f"\n  n = {n:>7,}")

        # FDC (Rust)
        if has_rust:
            t = _time_fdc(True, X, n_trials)
            times["FDC (Rust)"][n] = t
            print(f"    FDC (Rust):    {t:7.3f}s")
        else:
            times["FDC (Rust)"][n] = float("nan")

        # FDC (Python)
        t = _time_fdc(False, X, n_trials)
        times["FDC (Python)"][n] = t
        print(f"    FDC (Python):  {t:7.3f}s")

        # DBSCAN (C/Cython)
        t = _time_sklearn(DBSCAN, X, n_trials, eps=0.3, min_samples=5)
        times["DBSCAN (C)"][n] = t
        print(f"    DBSCAN (C):   {t:7.3f}s")

        # HDBSCAN (C/Cython)
        t = _time_sklearn(HDBSCAN, X, n_trials, min_cluster_size=15)
        times["HDBSCAN (C)"][n] = t
        print(f"    HDBSCAN (C):  {t:7.3f}s")

    return times


# ── Print tables ─────────────────────────────────────────────────────────────

def print_tables(times):
    print("\n=== Runtime (seconds, median) ===\n")
    rows = []
    for n in SIZES:
        row = [f"{n:,}"]
        for a in ALGO_NAMES:
            v = times[a].get(n, float("nan"))
            row.append(f"{v:.3f}" if not np.isnan(v) else "—")
        rows.append(row)
    print(tabulate(rows, headers=["n"] + ALGO_NAMES, tablefmt="simple"))

    # Speedup table: FDC (Rust) vs each competitor
    print("\n=== Speedup: how many times faster FDC (Rust) is ===\n")
    rows = []
    for n in SIZES:
        t_rust = times["FDC (Rust)"].get(n, float("nan"))
        row = [f"{n:,}"]
        for a in ALGO_NAMES:
            if a == "FDC (Rust)":
                row.append("—")
                continue
            t_other = times[a].get(n, float("nan"))
            if np.isnan(t_rust) or np.isnan(t_other) or t_rust == 0:
                row.append("—")
            else:
                ratio = t_other / t_rust
                if ratio >= 1:
                    row.append(f"{ratio:.1f}x faster")
                else:
                    row.append(f"{1/ratio:.1f}x slower")
        rows.append(row)
    print(tabulate(rows, headers=["n"] + ALGO_NAMES, tablefmt="simple"))
    print()


# ── CSV export ───────────────────────────────────────────────────────────────

def export_csv(times, outdir=None):
    import csv
    if outdir is None:
        outdir = os.path.dirname(__file__)

    with open(os.path.join(outdir, "rust_vs_c_runtime.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n"] + ALGO_NAMES)
        for n in SIZES:
            row = [n]
            for a in ALGO_NAMES:
                v = times[a].get(n, float("nan"))
                row.append(f"{v:.4f}" if not np.isnan(v) else "")
            w.writerow(row)
    print(f"CSV written to {outdir}/rust_vs_c_runtime.csv")


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_results(times, outfile="benchmark_rust_vs_python.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "FDC (Rust)":    "#e41a1c",
        "FDC (Python)":  "#377eb8",
        "DBSCAN (C)":    "#4daf4a",
        "HDBSCAN (C)":   "#984ea3",
    }
    markers = {
        "FDC (Rust)":    "s",
        "FDC (Python)":  "o",
        "DBSCAN (C)":    "D",
        "HDBSCAN (C)":   "^",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: log-log runtime comparison
    for algo in ALGO_NAMES:
        ns = sorted(times[algo].keys())
        ts = [times[algo][n] for n in ns]
        valid = [(n, t) for n, t in zip(ns, ts) if not np.isnan(t)]
        if not valid:
            continue
        vn, vt = zip(*valid)
        ax1.plot(vn, vt, marker=markers[algo], color=colors[algo],
                 linewidth=2, markersize=6, label=algo)

    # Reference lines
    n_ref = np.array([SIZES[0], SIZES[-1]], dtype=float)
    t_base = 0.005
    ax1.plot(n_ref, t_base * (n_ref / n_ref[0]) * np.log2(n_ref) / np.log2(n_ref[0]),
             "-.", color="gray", alpha=0.4, linewidth=1, label="O(n log n)")
    ax1.plot(n_ref, t_base * (n_ref / n_ref[0]),
             "--", color="gray", alpha=0.3, linewidth=1, label="O(n)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Dataset size (n)", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Clustering Runtime: FDC-Rust vs C Implementations", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # Right: speedup bars (FDC-Rust vs each competitor at largest size)
    competitors = ["FDC (Python)", "DBSCAN (C)", "HDBSCAN (C)"]
    bar_data = []
    for n in SIZES:
        t_rust = times["FDC (Rust)"].get(n, float("nan"))
        if np.isnan(t_rust):
            continue
        row = {"n": n}
        for comp in competitors:
            t_comp = times[comp].get(n, float("nan"))
            if not np.isnan(t_comp) and t_rust > 0:
                row[comp] = t_comp / t_rust
            else:
                row[comp] = float("nan")
        bar_data.append(row)

    x = np.arange(len(bar_data))
    width = 0.25
    comp_colors = {"FDC (Python)": "#377eb8", "DBSCAN (C)": "#4daf4a", "HDBSCAN (C)": "#984ea3"}

    for i, comp in enumerate(competitors):
        vals = [d[comp] for d in bar_data]
        ax2.bar(x + i * width, vals, width, label=f"vs {comp}",
                color=comp_colors[comp], alpha=0.8)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"{d['n']:,}" for d in bar_data], rotation=45, ha="right", fontsize=8)
    ax2.set_xlabel("Dataset size (n)", fontsize=12)
    ax2.set_ylabel("Speedup (competitor / FDC-Rust)", fontsize=12)
    ax2.set_title("FDC-Rust Speedup vs Competitors", fontsize=13)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="break-even")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    outpath = os.path.join(os.path.dirname(__file__), outfile)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {outpath}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FDC (Rust/Python) vs C-backed DBSCAN/HDBSCAN benchmark"
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per (algo, size) (default: 3)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    times = run_benchmark(n_trials=args.trials)
    print_tables(times)
    export_csv(times)

    if not args.no_plot:
        plot_results(times)


if __name__ == "__main__":
    main()
