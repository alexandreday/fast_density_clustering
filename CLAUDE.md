# CLAUDE.md — Fast Density Clustering (fdc)

## Project overview
PyPI package for density-based clustering of 2D/3D data. Uses kernel density estimation and a density graph to find clusters without specifying the number upfront. O(n log n) time, O(n) memory. Includes an optional Rust extension (`fdc_rs`) for 2–3x acceleration.

Author: Alexandre Day | License: MIT | Python ≥ 3.10

## Development setup
Uses `uv` for environment management and `maturin` to build the Rust extension.

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

# Build Rust extension + install in dev mode
uv pip install maturin
maturin develop --release
uv pip install -e ".[dev,plotting]"
```

For pure-Python development (no Rust toolchain needed):
```bash
uv pip install -e ".[dev,plotting]"
```

## Key commands
Always run commands inside the uv virtual environment. Prefix with `uv run` or activate first:

```bash
uv run pytest -v                            # run all tests (103 tests)
uv run pytest tests/test_fdc.py -v          # run a specific test file
uv run pytest tests/test_rust_backend.py -v # run Rust-specific parity tests
uv run python example/example.py            # run the main example
uv run python example/benchmark_sklearn.py  # run sklearn quality benchmark
uv run python benchmarks/benchmark_rust_vs_python.py  # Rust vs C scaling benchmark
maturin develop --release                   # rebuild Rust extension after changes
```

## Repo layout
```
fdc/                      # Python package
  fdc.py                  # FDC class — core algorithm orchestrator
  density_estimation.py   # KDE class with bandwidth optimization
  graph.py                # DGRAPH class — density graph with classifiers
  hierarchy.py            # dendrogram / linkage matrix utilities
  classify.py             # binary classifier (SVM/RF) for cluster boundaries
  tree.py                 # hierarchical tree structure
  plotting.py             # visualization helpers
  utils.py                # miscellaneous utilities

src/                      # Rust extension (fdc_rs), built by maturin
  lib.rs                  # PyO3 module entry point
  knn.rs                  # k-NN via kiddo KD-tree (dim 2–8, rayon parallel)
  kde.rs                  # Epanechnikov KDE + Brent's method bandwidth opt
  cluster.rs              # compute_delta, assign_cluster, stability_loop

tests/
  test_fdc.py             # FDC end-to-end tests
  test_kde.py             # KDE unit tests
  test_rust_backend.py    # Rust parity tests (k-NN, KDE, clustering)
  test_utils.py           # utility function tests

example/
  example.py              # Gaussian mixture demo
  benchmark_sklearn.py    # ARI/runtime comparison vs DBSCAN, HDBSCAN, etc.

benchmarks/
  benchmark_scaling.py    # runtime/memory vs dataset size (all algorithms)
  benchmark_quality.py    # ARI/AMI across 22 datasets vs competitors
  benchmark_rust_vs_python.py  # FDC-Rust vs C-backed DBSCAN/HDBSCAN
```

## Build system
- **Root `pyproject.toml`**: maturin backend — builds both the `fdc` Python package and the `fdc_rs` Rust extension into a single wheel
- **`Cargo.toml`**: Rust crate config — pyo3, kiddo, rayon, ndarray
- Wheels are built for Linux, macOS, Windows via `PyO3/maturin-action` in CI
- Pure-Python fallback: if `fdc_rs` is not installed, `fdc` works without Rust (just slower)

## CI
GitHub Actions (`.github/workflows/ci.yml`):
- `test-rust`: Builds Rust extension + runs all 103 tests (Python 3.10/3.11/3.12)
- `test-python-only`: Verifies pure-Python fallback works (Python 3.10/3.12)
- `typecheck`: mypy on core modules

## Code conventions
- Python 3.10+ — walrus operators and match syntax are allowed
- No type annotations required (adding them to new code is welcome, but don't annotate unchanged code)
- NumPy-style array operations throughout; avoid Python loops over large arrays
- Do not use deprecated numpy type aliases (`np.int`, `np.float`, `np.bool`, etc.) — use `np.int64`, `float`, `bool`, etc.
- Avoid `is` comparisons for array equality; use `==` or `np.array_equal`
- Rust code: use rayon for parallelism, avoid unsafe, keep PyO3 boundary thin

## What NOT to do
- Do not commit build artifacts: `build/`, `dist/`, `*.egg-info/`, `target/`
- Do not add backwards-compatibility shims or `_unused` variable renames
- Do not over-engineer: no extra abstraction layers, no feature flags, no speculative futures
- Do not add docstrings or comments to code you didn't change
- Do not add Claude as co-author in commit messages

## Modernization roadmap

- [x] Phase 1 — Modern packaging (`pyproject.toml`, drop `setup.py`)
- [x] Phase 2 — Add pytest test suite (`tests/`)
- [x] Phase 3 — Fix deprecated numpy aliases and identity comparisons
- [x] Phase 4 — GitHub Actions CI (Python 3.10 / 3.11 / 3.12)
- [x] Phase 5 — Clean up dead code (`tree_old.py`, unused imports, build artifacts in `.gitignore`)
- [x] Phase 6 — Type annotations on public API (`FDC`, `KDE`, `DGRAPH`)
- [x] Phase 7 — Improve test coverage (KDE, graph, hierarchy modules)
- [x] Phase 8 — Rust acceleration via PyO3 (2–3x speedup, maturin packaging)
