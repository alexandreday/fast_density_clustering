# fdc — Fast Density Clustering

A Python package for clustering low-dimensional data using kernel density maps and density graphs.
Ships with a Rust extension for 2–3x acceleration out of the box.

## Features

- **No cluster count required** — automatically discovers the number of clusters
- **Non-convex clusters** — handles crescents, rings, and other complex shapes
- **Multiscale** — works with varying densities and population sizes
- **Fast** — O(n log n) time, O(n) memory via KD-tree search
- **Rust-accelerated** — optional native extension for 2–3x speedup (pre-built wheels for Linux, macOS, Windows)
- **Two parameters** — neighborhood size (auto) and noise threshold (eta)

## Installation

```bash
pip install fdc
```

Pre-built wheels include the Rust extension. Falls back to pure Python if no wheel is available.

## Quick start

```python
from fdc import FDC
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=5000, n_features=2, centers=10)
X = StandardScaler().fit_transform(X)

model = FDC(eta=0.5)
model.fit(X)
print(f"Found {len(model.idx_centers)} clusters")
```

## Performance vs C-backed competitors (100K points, 2D)

| Algorithm | Runtime |
|---|---|
| **FDC (Rust)** | **1.8s** |
| FDC (Python) | 4.8s |
| DBSCAN (sklearn/C) | 23.7s |
| HDBSCAN (sklearn/C) | 61.8s |

## Links

- [GitHub](https://github.com/alexandreday/fast_density_clustering)
- [Examples](https://github.com/alexandreday/fast_density_clustering/tree/master/example)

## Citation

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
