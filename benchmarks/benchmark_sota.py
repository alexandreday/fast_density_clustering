"""
SOTA clustering benchmark: FDC vs modern density-based methods.

Evaluates ARI and AMI across the same dataset suites as benchmark_quality.py,
comparing FDC against state-of-the-art density clustering algorithms:
  - pydpc (Density Peaks Clustering, Rodriguez & Laio 2014)
  - DBSCAN++ (Jang et al., coreset-accelerated DBSCAN)

Also includes sklearn HDBSCAN as a well-known baseline.

Usage:
    uv run python benchmarks/benchmark_sota.py [--no-plot] [--suite sklearn|sipu|fcps|all]
"""

import argparse
import multiprocessing
import sys
import os
import time
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import HDBSCAN
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fdc import FDC
from benchmarks.benchmark_quality import load_datasets

warnings.filterwarnings("ignore")

# ── Algorithm wrappers ────────────────────────────────────────────────────────

# Each wrapper returns labels as an ndarray of ints.
# Parameter sweeps find the best ARI per dataset (like benchmark_quality.py).


def _run_fdc(X, eta=0.6):
    """FDC with given eta."""
    _real_stdout = sys.__stdout__
    sys.__stdout__ = open(os.devnull, "w")
    _cur_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    try:
        model = FDC(eta=eta, verbose=0, random_state=42)
        model.fit(X)
        return model.cluster_label
    finally:
        sys.__stdout__.close()
        sys.__stdout__ = _real_stdout
        sys.stdout = _cur_stdout


def _fdc_param_grid():
    return [{"eta": eta} for eta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]


def _worker(runner_fn, X, y_true, params, result_queue):
    """Worker for subprocess-isolated algorithm calls (guards against segfaults)."""
    try:
        labels = runner_fn(X, **params)
        ari = adjusted_rand_score(y_true, labels)
        result_queue.put((ari, labels))
    except Exception:
        result_queue.put(None)


def _sweep_best(X, y_true, runner_fn, param_grid, isolate=False):
    """Run an algorithm with multiple parameter settings, return best ARI result.

    If isolate=True, each call runs in a subprocess so that segfaults in C
    extensions don't kill the parent process.

    Returns (best_labels, best_params).
    """
    best_labels = None
    best_params = None
    best_ari = -2

    if not isolate:
        for params in param_grid:
            try:
                labels = runner_fn(X, **params)
                ari = adjusted_rand_score(y_true, labels)
                if ari > best_ari:
                    best_ari = ari
                    best_labels = labels
                    best_params = params
            except Exception:
                continue
    else:
        ctx = multiprocessing.get_context("fork")
        for params in param_grid:
            q = ctx.Queue()
            p = ctx.Process(target=_worker, args=(runner_fn, X, y_true, params, q))
            p.start()
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
                p.join()
                continue
            if p.exitcode != 0:
                continue
            try:
                result = q.get_nowait()
            except Exception:
                continue
            if result is not None:
                ari, labels = result
                if ari > best_ari:
                    best_ari = ari
                    best_labels = labels
                    best_params = params

    return best_labels, best_params


# ── pydpc ─────────────────────────────────────────────────────────────────────

def _has_pydpc():
    try:
        import pydpc
        return True
    except ImportError:
        return False


def _run_pydpc(X, fraction=0.02, density_threshold=1.0, delta_threshold=1.0):
    import pydpc
    c = pydpc.Cluster(X, fraction=fraction, autoplot=False)
    c.assign(density_threshold, delta_threshold)
    return np.array(c.membership)


def _pydpc_param_grid():
    """Parameter grid for pydpc sweep."""
    grid = []
    for fraction in [0.01, 0.02, 0.05]:
        for density_t in [0.5, 1.0, 3.0, 5.0]:
            for delta_t in [0.3, 0.5, 1.0, 2.0]:
                grid.append({
                    "fraction": fraction,
                    "density_threshold": density_t,
                    "delta_threshold": delta_t,
                })
    return grid


# ── DBSCAN++ ─────────────────────────────────────────────────────────────────

def _has_dbscanpp():
    try:
        from DBSCANPP import DBSCANPP
        return True
    except ImportError:
        return False


def _run_dbscanpp(X, p=0.1, eps=0.5, minPts=5):
    from DBSCANPP import DBSCANPP
    return DBSCANPP(p=p, eps_density=eps, eps_clustering=eps, minPts=minPts).fit_predict(X)


def _dbscanpp_param_grid():
    """Parameter grid for DBSCAN++ sweep."""
    grid = []
    for p in [0.1, 0.2, 0.3]:
        for eps in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
            for minPts in [3, 5, 10]:
                grid.append({"p": p, "eps": eps, "minPts": minPts})
    return grid


# ── HDBSCAN sweep ────────────────────────────────────────────────────────────

def _run_hdbscan(X, min_cluster_size=15):
    return HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)


def _hdbscan_param_grid():
    return [{"min_cluster_size": mcs} for mcs in [5, 10, 15, 20, 30, 50, 75, 100]]


# ── Algorithm registry ───────────────────────────────────────────────────────

ALGORITHMS = []


def _build_algorithm_list():
    global ALGORITHMS
    ALGORITHMS = [
        {
            "name": "FDC",
            "runner": _run_fdc,
            "grid": _fdc_param_grid(),
            "isolate": False,
        },
        {
            "name": "HDBSCAN*",
            "runner": _run_hdbscan,
            "grid": _hdbscan_param_grid(),
            "isolate": False,
        },
    ]
    if _has_pydpc():
        ALGORITHMS.append({
            "name": "pydpc",
            "runner": _run_pydpc,
            "grid": _pydpc_param_grid(),
            "isolate": True,  # C extension can segfault on certain param combos
        })
    else:
        print("  pydpc not installed — skipping (pip install pydpc)")

    if _has_dbscanpp():
        ALGORITHMS.append({
            "name": "DBSCAN++",
            "runner": _run_dbscanpp,
            "grid": _dbscanpp_param_grid(),
            "isolate": True,  # C extension can segfault on certain params
        })
    else:
        print("  DBSCAN++ not installed — skipping (pip install dbscanpp)")


# ── Benchmark runner ─────────────────────────────────────────────────────────

def run_benchmark(suite="all"):
    _build_algorithm_list()
    algo_names = [a["name"] for a in ALGORITHMS]

    suites = load_datasets(suite)
    if not suites:
        print("No datasets loaded.")
        return [], {}, {}, {}

    results = {}
    pred_labels = {}
    scaled_data = {}

    for suite_name, ds_dict in suites:
        for ds_name, (X, y_true) in ds_dict.items():
            n, d = X.shape
            k = len(np.unique(y_true[y_true >= 0]))
            X_scaled = StandardScaler().fit_transform(X)
            scaled_data[ds_name] = (X_scaled, y_true)
            results[ds_name] = {}
            pred_labels[ds_name] = {}

            short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
            print(f"  {short} (n={n}, d={d}, k={k})", end="", flush=True)

            for algo in ALGORITHMS:
                name = algo["name"]

                try:
                    labels, best_params = _sweep_best(
                        X_scaled, y_true, algo["runner"], algo["grid"],
                        isolate=algo.get("isolate", False),
                    )
                    if labels is None:
                        raise RuntimeError("all parameter settings failed")

                    # Time a single run with best params
                    t0 = time.perf_counter()
                    algo["runner"](X_scaled, **best_params)
                    dt = time.perf_counter() - t0
                except Exception as e:
                    print(f" [{name}: FAIL]", end="")
                    labels = np.full(n, -1)
                    dt = float("nan")

                ari = adjusted_rand_score(y_true, labels)
                ami = adjusted_mutual_info_score(y_true, labels)

                pred_labels[ds_name][name] = labels
                results[ds_name][name] = {
                    "ari": ari, "ami": ami, "time": dt,
                    "n": n, "d": d, "k": k,
                }
                print(f"  {name}={ari:.3f}", end="")

            print()

    return suites, results, pred_labels, scaled_data


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_score(val, is_best):
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

    algo_names = [a["name"] for a in ALGORITHMS]

    print()
    print("=" * 90)
    print("  SOTA CLUSTERING BENCHMARK")
    print("  Methods: " + ", ".join(algo_names))
    print("  Metrics: ARI / AMI  |  * = best in row  |  Sweep = best over param grid")
    print("=" * 90)

    for suite_name, ds_dict in suites:
        ds_names = [n for n in ds_dict if n in results]
        if not ds_names:
            continue

        print(f"\n── {suite_name.upper()} ", end="")
        print("─" * (86 - len(suite_name)))

        headers = ["dataset", "n", "d", "k"]
        for algo in algo_names:
            headers.extend([algo, ""])
        sub_headers = ["", "", "", ""]
        for algo in algo_names:
            sub_headers.extend(["ARI", "AMI"])

        rows = []
        for ds_name in ds_names:
            r = results[ds_name]
            meta = r[algo_names[0]]
            short = ds_name.split("/")[-1] if "/" in ds_name else ds_name

            aris = {a: r[a]["ari"] for a in algo_names}
            amis = {a: r[a]["ami"] for a in algo_names}
            best_ari = max(aris.values())
            best_ami = max(amis.values())

            row = [short, meta["n"], meta["d"], meta["k"]]
            for algo in algo_names:
                row.append(_fmt_score(aris[algo], aris[algo] == best_ari))
                row.append(_fmt_score(amis[algo], amis[algo] == best_ami))
            rows.append(row)

        sep_row = ["", "", "", ""] + ["─────"] * (len(algo_names) * 2)
        rows.append(sep_row)
        mean_row = ["MEAN", "", "", ""]
        for algo in algo_names:
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

    # Overall summary
    all_ds = list(results.keys())
    if len(all_ds) > 5:
        print(f"\n── OVERALL ", end="")
        print("─" * 78)
        headers = [""] + algo_names
        ari_row = ["Mean ARI"]
        ami_row = ["Mean AMI"]
        time_row = ["Total time (s)"]
        for algo in algo_names:
            ari_row.append(f"{np.mean([results[ds][algo]['ari'] for ds in all_ds]):.3f}")
            ami_row.append(f"{np.mean([results[ds][algo]['ami'] for ds in all_ds]):.3f}")
            time_row.append(f"{sum(results[ds][algo]['time'] for ds in all_ds):.1f}")
        print(tabulate([ari_row, ami_row, time_row], headers=headers, tablefmt="simple"))

    # Runtime table
    print(f"\n── RUNTIME (seconds) ", end="")
    print("─" * 69)
    headers = ["dataset", "n"] + algo_names
    rows = []
    for suite_name, ds_dict in suites:
        ds_names = [n for n in ds_dict if n in results]
        for ds_name in ds_names:
            r = results[ds_name]
            short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
            n = r[algo_names[0]]["n"]
            row = [short, n]
            for algo in algo_names:
                row.append(_fmt_time(r[algo]["time"]))
            rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="simple", numalign="right"))
    print()


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(suites, results, pred_labels, scaled_data,
                 outfile="benchmark_sota.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algo_names = [a["name"] for a in ALGORITHMS]
    ds_names = list(results.keys())
    n_ds = len(ds_names)
    n_alg = len(algo_names)

    fig, axes = plt.subplots(
        n_ds, n_alg, figsize=(n_alg * 2.6, n_ds * 2.2),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.50, wspace=0.15)
    cmap = plt.cm.tab10

    for row, ds_name in enumerate(ds_names):
        X, y_true = scaled_data[ds_name]
        for col, algo_name in enumerate(algo_names):
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


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csvs(suites, results):
    import csv

    if not results:
        return

    algo_names = [a["name"] for a in ALGORITHMS]
    benchdir = os.path.dirname(__file__)

    # ARI CSV
    ari_path = os.path.join(benchdir, "sota_ari.csv")
    with open(ari_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "dataset", "n", "k"] + algo_names)
        for suite_name, ds_dict in suites:
            for ds_name in ds_dict:
                if ds_name not in results:
                    continue
                r = results[ds_name]
                meta = r[algo_names[0]]
                short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
                row = [suite_name, short, meta["n"], meta["k"]]
                for a in algo_names:
                    row.append(f"{r[a]['ari']:.4f}")
                w.writerow(row)
    print(f"Saved {ari_path}")

    # Runtime CSV
    rt_path = os.path.join(benchdir, "sota_runtime.csv")
    with open(rt_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "dataset", "n"] + algo_names)
        for suite_name, ds_dict in suites:
            for ds_name in ds_dict:
                if ds_name not in results:
                    continue
                r = results[ds_name]
                meta = r[algo_names[0]]
                short = ds_name.split("/")[-1] if "/" in ds_name else ds_name
                row = [suite_name, short, meta["n"]]
                for a in algo_names:
                    t = r[a]["time"]
                    row.append(f"{t:.4f}" if not np.isnan(t) else "")
                w.writerow(row)
    print(f"Saved {rt_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOTA clustering benchmark")
    parser.add_argument(
        "--suite",
        choices=["sklearn", "sipu", "fcps", "all"],
        default="all",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"Running SOTA benchmark (suite={args.suite})...")
    suites, results, pred_labels, scaled_data = run_benchmark(args.suite)
    print_results(suites, results)
    save_csvs(suites, results)

    if not args.no_plot and results:
        plot_results(suites, results, pred_labels, scaled_data)


if __name__ == "__main__":
    main()
