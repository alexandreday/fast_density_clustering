# Fast density clustering (fdc)
Python package for clustering two dimensional data using kernel density maps to construct a density graph. Examples for gaussian mixtures and some benchmarks are provided. Our algorithm solves multiscale problems (multiple variances/densities and population sizes) and works for non-convex clusters. It uses cross-validation and is regularized by two main global parameters : a neighborhood
size and a noise threshold measure. The later detects spurious cluster centers while the former guarantees that only local information is used to infer cluster centers (we avoid using long distance information). 

The underlying code is based on fast KD-trees for nearest-neighbor searches O(n log n). While the algorithm is well suited for small datasets with meaningful densities, it works quite well on large datasets 
(c.g. for N=10000, run time is a few seconds).

  # High-dimensional data clustering
  Our approach can be combined with high-dimensional data. Density maps are estimated on a projected or a low dimensional representation of the high-dimensional data. The statistical significance of each clusters (wether it is noise or not) is then evaluated using multi-class logistic regression. Example of the full method coming soon !

# Installing
I suggest you install the code using ```pip``` from an [Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html) Python 3 environment. From that environment:
```
git clone https://github.com/alexandreday/fast_density_clustering.git
cd fast_density_clustering
pip install .
```
That's it ! You can now import the package ```fdc``` from your python scripts. Check out the examples
in the file ```example``` and see if you can run the scripts provided.
# Examples and comparison with other methods
Check out the example for gaussian mixtures (example.py). You should be able to run it directly. It
should produce a plot similar to this: ![alt tag](https://github.com/alexandreday/fast_density_clustering/blob/master/example/result.png)

In another example (example2.py), the algorithm is benchmarked against some sklearn datasets (note that the same parameters are used across all datasets). This is to be compared with other clustering methods easily accesible from [sklearn](http://scikit-learn.org/stable/modules/clustering.html). ![alt tag](https://github.com/alexandreday/fast_density_clustering/blob/master/example/sklearn_datasets.png)
# Citation

If you use this code in a scientific publication, I would appreciate citation/reference to this repository
