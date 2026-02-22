# fdc — Fast Density Clustering

A Python package for efficiently clustering low-dimensional data using kernel density maps and density graphs.

- Works without specifying the number of clusters upfront
- Handles non-convex clusters and multiscale problems (varying densities, population sizes)
- O(n log n) time and O(n) memory complexity via KD-tree nearest-neighbor search
- Regularized by two interpretable parameters: neighborhood size and noise threshold

## Installation

### From PyPI

```bash
pip install fdc

# With optional plotting support
pip install "fdc[plotting]"
```

### From source (development)

We use [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# 1. Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and set up
git clone https://github.com/alexandreday/fast_density_clustering.git
cd fast_density_clustering
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[plotting]"
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

## Examples

`example/example.py` — Gaussian mixture clustering. Produces a plot like this:

![Gaussian mixture result](https://github.com/alexandreday/fast_density_clustering/blob/master/example/result.png)

`example/example2.py` — Benchmark against standard [sklearn datasets](http://scikit-learn.org/stable/modules/clustering.html), using the same parameters across all datasets:

![sklearn datasets benchmark](https://github.com/alexandreday/fast_density_clustering/blob/master/example/sklearn_datasets.png)

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
