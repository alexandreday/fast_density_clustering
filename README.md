# fdc — Fast Density Clustering

A Python package for efficiently clustering low-dimensional data using kernel density maps and density graphs.
Ships with an optional Rust extension (`fdc_rs`) for 2–3x acceleration on the core algorithm.

- Works without specifying the number of clusters upfront
- Handles non-convex clusters and multiscale problems (varying densities, population sizes)
- O(n log n) time and O(n) memory complexity via KD-tree nearest-neighbor search
- Adaptive neighborhood sizing: automatically adjusts to the data's density scale using the optimized KDE bandwidth
- Regularized by two interpretable parameters: neighborhood size and noise threshold
- Optional Rust acceleration via PyO3: 2–3x faster than pure Python, 13x faster than C-backed DBSCAN at 100K points

## Installation

### From PyPI

```bash
pip install fdc

# With optional plotting support
pip install "fdc[plotting]"
```

Pre-built wheels include the Rust extension for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon), and Windows.
If no pre-built wheel is available for your platform, pip builds from source (requires a [Rust toolchain](https://rustup.rs/)).
The package works without Rust — it falls back to a pure-Python implementation automatically.

### From source (development)

We use [uv](https://docs.astral.sh/uv/) for environment management and [maturin](https://www.maturin.rs/) to build the Rust extension.

```bash
# 1. Clone and set up
git clone https://github.com/alexandreday/fast_density_clustering.git
cd fast_density_clustering
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate

# 2. Build the Rust extension and install in development mode
uv pip install maturin
maturin develop --release
uv pip install -e ".[dev,plotting]"

# 3. Run tests
uv run pytest -v
```

To work without Rust (pure-Python only):

```bash
uv pip install -e ".[dev,plotting]"
```

## Quick start

```python
from fdc import FDC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score as nmi

X, y = make_blobs(n_samples=10000, n_features=2, centers=15)
X = StandardScaler().fit_transform(X)

model = FDC(eta=0.01)
model.fit(X)

print("NMI:", nmi(y, model.cluster_label))
```

## Performance

FDC with its Rust extension (`fdc_rs`) is significantly faster than sklearn's C-backed clustering algorithms at scale:

| n | FDC (Rust) | FDC (Python) | DBSCAN (C) | HDBSCAN (C) |
|---|---|---|---|---|
| 5,000 | **0.04s** | 0.11s | 0.07s | 0.16s |
| 10,000 | **0.07s** | 0.24s | 0.24s | 0.60s |
| 50,000 | **0.71s** | 1.94s | 5.48s | 15.2s |
| 100,000 | **1.79s** | 4.81s | 23.7s | 61.8s |

The Rust extension accelerates:
- **k-NN search**: `kiddo` KD-tree with rayon parallelism (5x faster than sklearn)
- **KDE evaluation**: SIMD-friendly Epanechnikov kernel + parallel bandwidth optimization
- **Cluster stability**: Fused assign+check loop avoids Python↔Rust overhead

## Architecture

```
fdc/                     # Python package
  fdc.py                 # FDC class — core algorithm orchestrator
  density_estimation.py  # KDE class with bandwidth optimization
  graph.py               # DGRAPH — density graph with SVM/RF classifiers
  hierarchy.py           # Dendrogram / linkage matrix utilities
  classify.py            # Binary classifier (SVM/RF) for cluster boundaries
  tree.py                # Hierarchical tree structure
  plotting.py            # Visualization helpers

src/                     # Rust extension (fdc_rs), built by maturin
  lib.rs                 # PyO3 module — exports all functions to Python
  knn.rs                 # k-NN search via kiddo KD-tree (dim 2–8, rayon parallel)
  kde.rs                 # Epanechnikov KDE + Brent's method bandwidth optimization
  cluster.rs             # compute_delta, assign_cluster, stability_loop

tests/                   # pytest test suite
  test_fdc.py            # FDC end-to-end tests
  test_kde.py            # KDE unit tests
  test_rust_backend.py   # Rust parity and direct API tests
```

## Examples

`example/example.py` — Gaussian mixture clustering. Produces a plot like this:

![Gaussian mixture result](https://github.com/alexandreday/fast_density_clustering/blob/master/example/result.png)

`example/benchmark_sklearn.py` — Benchmark against standard [sklearn datasets](http://scikit-learn.org/stable/modules/clustering.html), comparing FDC with DBSCAN, HDBSCAN, OPTICS, and MeanShift:

![sklearn datasets benchmark](https://github.com/alexandreday/fast_density_clustering/blob/master/example/benchmark_sklearn.png)

## Citation

If you use this code in a scientific publication, please cite this repository. For further reading on clustering and machine learning:

```bibtex
@article{mehta2018high,
  title={A high-bias, low-variance introduction to Machine Learning for physicists},
  author={Mehta, Pankaj and Bukov, Marin and Wang, Ching-Hao and Day, Alexandre GR and Richardson, Clint and Fisher, Charles K and Schwab, David J},
  journal={arXiv preprint arXiv:1803.08823},
  year={2018}
}
```

## License

MIT
