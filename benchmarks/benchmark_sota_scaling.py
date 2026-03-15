"""
SOTA scalability benchmark: runtime and peak memory vs dataset size.

Compares FDC (Rust) against pydpc, DBSCAN++, and HDBSCAN on synthetic
2D Gaussian blobs with increasing n. Generates a log-log runtime plot.

Usage:
    uv run python benchmarks/benchmark_sota_scaling.py [--no-plot] [--trials 3]
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
from sklearn.cluster import HDBSCAN
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fdc import FDC

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
N_CLUSTERS = 10
N_FEATURES = 2
RANDOM_STATE = 42

# pydpc uses O(n^2) distance matrix — skip past this threshold
PYDPC_MAX_N = 10_000
# DPA is very slow (~1s/point); skip past this threshold
DPA_MAX_N = 5_000


# ── Algorithm wrappers ───────────────────────────────────────────────────────

def _has_pydpc():
    try:
        import pydpc
        return True
    except ImportError:
        return False


def _has_dbscanpp():
    try:
        from DBSCANPP import DBSCANPP
        return True
    except ImportError:
        return False


def _fit_fdc(X):
    _real_stdout = sys.__stdout__
    sys.__stdout__ = open(os.devnull, "w")
    _cur_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        FDC(eta=0.6, verbose=0, random_state=42).fit(X)
    finally:
        sys.__stdout__.close()
        sys.__stdout__ = _real_stdout
        sys.stdout = _cur_stdout


def _fit_hdbscan(X):
    HDBSCAN(min_cluster_size=15).fit(X)


def _fit_pydpc(X):
    import pydpc
    c = pydpc.Cluster(X, fraction=0.02, autoplot=False)
    c.assign(1.0, 1.0)


def _fit_dbscanpp(X):
    from DBSCANPP import DBSCANPP
    DBSCANPP(p=0.3, eps_density=0.5, eps_clustering=0.5, minPts=5).fit_predict(X)


def _has_dpa():
    try:
        from Pipeline import DPA
        return True
    except ImportError:
        return False


def _fit_dpa(X):
    from Pipeline import DPA
    DPA.DensityPeakAdvanced(Z=1.0).fit(X)


# ── Algorithm registry ───────────────────────────────────────────────────────

def _build_algorithms():
    algos = [
        {"name": "FDC", "fit": _fit_fdc, "max_n": None},
        {"name": "HDBSCAN*", "fit": _fit_hdbscan, "max_n": None},
    ]
    if _has_pydpc():
        algos.append({
            "name": "pydpc",
            "fit": _fit_pydpc,
            "max_n": PYDPC_MAX_N,
        })
    else:
        print("  pydpc not installed — skipping")

    if _has_dbscanpp():
        algos.append({
            "name": "DBSCAN++",
            "fit": _fit_dbscanpp,
            "max_n": None,
        })
    else:
        print("  DBSCAN++ not installed — skipping")

    if _has_dpa():
        algos.append({
            "name": "DPA",
            "fit": _fit_dpa,
            "max_n": DPA_MAX_N,
        })
    else:
        print("  DPA not installed — skipping")

    return algos


# ── Measurement ──────────────────────────────────────────────────────────────

def measure_once(fit_fn, X):
    """Return (wall_seconds, peak_memory_MiB)."""
    gc.collect()
    tracemalloc.start()

    t0 = time.perf_counter()
    try:
        fit_fn(X)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        elapsed = float("nan")
        print(f"    FAIL: {e}")

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak / (1024 * 1024)


# ── Run benchmark ────────────────────────────────────────────────────────────

def run_benchmark(n_trials=3):
    algos = _build_algorithms()
    algo_names = [a["name"] for a in algos]

    times = {a["name"]: {} for a in algos}
    memory = {a["name"]: {} for a in algos}

    for n in SIZES:
        print(f"\nn = {n:,}")
        X, _ = make_blobs(
            n_samples=n, n_features=N_FEATURES,
            centers=N_CLUSTERS, random_state=RANDOM_STATE,
        )
        X = StandardScaler().fit_transform(X)

        for algo in algos:
            name = algo["name"]
            if algo["max_n"] is not None and n > algo["max_n"]:
                print(f"  {name:>10s}: skipped (n > {algo['max_n']:,})")
                times[name][n] = float("nan")
                memory[name][n] = float("nan")
                continue

            trial_times = []
            trial_mem = []
            for _ in range(n_trials):
                elapsed, peak_mib = measure_once(algo["fit"], X)
                trial_times.append(elapsed)
                trial_mem.append(peak_mib)

            med_time = float(np.nanmedian(trial_times))
            med_mem = float(np.nanmedian(trial_mem))
            times[name][n] = med_time
            memory[name][n] = med_mem
            print(f"  {name:>10s}: {med_time:8.3f}s  {med_mem:8.1f} MiB")

    return algo_names, times, memory


# ── Print tables ─────────────────────────────────────────────────────────────

def print_tables(algo_names, times, memory):
    print("\n=== Runtime (seconds, median) ===\n")
    rows = []
    for n in SIZES:
        row = [f"{n:,}"]
        for a in algo_names:
            v = times[a].get(n, float("nan"))
            row.append(f"{v:.3f}" if not np.isnan(v) else "—")
        rows.append(row)
    print(tabulate(rows, headers=["n"] + algo_names, tablefmt="simple"))

    print("\n=== Peak Memory (MiB, median) ===\n")
    rows = []
    for n in SIZES:
        row = [f"{n:,}"]
        for a in algo_names:
            v = memory[a].get(n, float("nan"))
            row.append(f"{v:.1f}" if not np.isnan(v) else "—")
        rows.append(row)
    print(tabulate(rows, headers=["n"] + algo_names, tablefmt="simple"))
    print()


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_scaling(algo_names, times, outfile="benchmark_sota_scaling.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    styles = {
        "FDC":      {"marker": "o", "color": "#e41a1c"},
        "HDBSCAN*": {"marker": "D", "color": "#4daf4a"},
        "pydpc":    {"marker": "^", "color": "#984ea3"},
        "DBSCAN++": {"marker": "s", "color": "#377eb8"},
        "DPA":      {"marker": "P", "color": "#ff7f00"},
    }

    for name in algo_names:
        ns = sorted(times[name].keys())
        valid = [(n, times[name][n]) for n in ns if not np.isnan(times[name][n])]
        if not valid:
            continue
        vn, vt = zip(*valid)
        s = styles.get(name, {"marker": "x", "color": "gray"})
        ax.plot(vn, vt, marker=s["marker"], color=s["color"],
                label=name, linewidth=2, markersize=6)

    # Reference lines
    n_ref = np.array([SIZES[0], SIZES[-1]], dtype=float)
    t_base = 0.01
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
    ax.set_title("SOTA Scaling — 2D Gaussian Blobs (k=10)", fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    outpath = os.path.join(os.path.dirname(__file__), outfile)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {outpath}")


# ── CSV export ───────────────────────────────────────────────────────────────

def save_csvs(algo_names, times, memory):
    import csv

    benchdir = os.path.dirname(__file__)

    # Runtime CSV
    rt_path = os.path.join(benchdir, "sota_scaling_runtime.csv")
    with open(rt_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n"] + algo_names)
        for n in SIZES:
            row = [n]
            for a in algo_names:
                v = times[a].get(n, float("nan"))
                row.append(f"{v:.4f}" if not np.isnan(v) else "")
            w.writerow(row)
    print(f"Saved {rt_path}")

    # Memory CSV
    mem_path = os.path.join(benchdir, "sota_scaling_memory.csv")
    with open(mem_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n"] + algo_names)
        for n in SIZES:
            row = [n]
            for a in algo_names:
                v = memory[a].get(n, float("nan"))
                row.append(f"{v:.1f}" if not np.isnan(v) else "")
            w.writerow(row)
    print(f"Saved {mem_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOTA scaling benchmark")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"Running SOTA scaling benchmark (trials={args.trials})...")
    algo_names, times, memory = run_benchmark(n_trials=args.trials)
    print_tables(algo_names, times, memory)
    save_csvs(algo_names, times, memory)

    if not args.no_plot:
        plot_scaling(algo_names, times)


if __name__ == "__main__":
    main()
