# Fast density clustering
Clustering for two dimensional data using kernel density maps to construct a density graph. Example for gaussian mixtures is given.
The algorithm solves multiscale problems (multiple variances and population sizes) and works for non-convex structures. It uses cross-validation and is regularized by a two main global parameters : a neighborhood
size and a noise threshold measure. The later detects spurious cluster centers. The underlying code is based on fast kd-trees nearest-neighbor searches
O(nlog n) and hash tables (O(1) insertion and item search). The codes runs for large datasets (for N=10000, run time is 2-3 seconds). More coming soon ...
