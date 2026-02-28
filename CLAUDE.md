# CLAUDE.md — Fast Density Clustering (fdc)

## Project overview
PyPI package for density-based clustering of 2D/3D data. Uses kernel density estimation and a density graph to find clusters without specifying the number upfront. O(n log n) time, O(n) memory.

Author: Alexandre Day | License: MIT | Python ≥ 3.9

## Development setup
Uses `uv` for environment management.

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,plotting]"
```

## Key commands
```bash
pytest -v                        # run all tests
pytest tests/test_fdc.py -v      # run a specific test file
python example/example.py        # run the main example
python example/benchmark_sklearn.py  # run sklearn benchmark
```

## Repo layout
```
fdc/
  fdc.py               # FDC class — core algorithm (fit, predict, etc.)
  density_estimation.py # KDE class with bandwidth optimization
  graph.py             # DGRAPH class — density graph construction/analysis
  hierarchy.py         # dendrogram / linkage matrix utilities
  classify.py          # classification module
  tree.py              # tree data structure
  plotting.py          # visualization helpers
  utils.py             # miscellaneous utilities
  robustness.py        # robustness / stability analysis
  widget.py            # interactive widget support
  mycolors.py          # color palette helpers
  special_datasets.py  # toy datasets for testing

tests/
  test_fdc.py          # FDC end-to-end tests
  test_kde.py          # KDE unit tests
  test_utils.py        # utility tests

example/
  example.py           # Gaussian mixture demo
  benchmark_sklearn.py # ARI/runtime comparison vs DBSCAN, HDBSCAN, etc.
```

## CI
GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` on Python 3.9, 3.11, and 3.12 via `uv`.

## Code conventions
- Python 3.9+ — no walrus operators or 3.10+ match syntax unless targeting ≥3.10
- No type annotations required (adding them to new code is welcome, but don't annotate unchanged code)
- NumPy-style array operations throughout; avoid Python loops over large arrays
- Do not use deprecated numpy type aliases (`np.int`, `np.float`, `np.bool`, etc.) — use `np.int64`, `float`, `bool`, etc.
- Avoid `is` comparisons for array equality; use `==` or `np.array_equal`

## What NOT to do
- Do not commit build artifacts: `build/`, `dist/`, `*.egg-info/`, `UNKNOWN.egg-info/`
- Do not add backwards-compatibility shims or `_unused` variable renames
- Do not edit `fdc/tree_old.py` — it is dead legacy code pending deletion
- Do not over-engineer: no extra abstraction layers, no feature flags, no speculative futures
- Do not add docstrings or comments to code you didn't change

## Modernization roadmap
The project is being brought up to date incrementally:

- [x] Phase 1 — Modern packaging (`pyproject.toml`, drop `setup.py`)
- [x] Phase 2 — Add pytest test suite (`tests/`)
- [x] Phase 3 — Fix deprecated numpy aliases and identity comparisons
- [x] Phase 4 — GitHub Actions CI (Python 3.9 / 3.11 / 3.12)
- [ ] Phase 5 — Clean up dead code (`tree_old.py`, unused imports, build artifacts in `.gitignore`)
- [ ] Phase 6 — Type annotations on public API (`FDC`, `KDE`, `DGRAPH`)
- [ ] Phase 7 — Improve test coverage (KDE, graph, hierarchy modules)
- [ ] Phase 8 — Performance profiling and optional Cython/Numba acceleration
