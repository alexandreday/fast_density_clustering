# Fast density clustering
Clustering for two dimensional data using kernel density maps to construct a density graph. Example for gaussian mixtures is given.
The algorithm solves multiscale problems (multiple variances and population sizes) and works for non-convex structures. It uses cross-validation and is regularized by two main global parameters : a neighborhood
size and a noise threshold measure. The later detects spurious cluster centers. The underlying code is based on fast kd-trees nearest-neighbor searches
O(nlog n) and hash tables (O(1) insertion and item search). The codes runs for large datasets (for N=10000, run time is 2-3 seconds). More coming soon ...

# Running

Clone or download this repository and run the following command inside the top directory:

pip install .

That's it ! Check the example for gaussian mixtures.

# Citation

If you use this code in a scientific publication, I would appreciate citation/reference to this repository
