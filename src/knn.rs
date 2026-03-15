//! k-Nearest-Neighbor search using the `kiddo` KD-tree.
//!
//! Provides two entry points:
//! - `knn_query`: build a tree from X, query all points against it (self-query).
//! - `knn_query_cross`: build a tree from X_train, query X_test against it.
//!
//! Both use compile-time dimension specialization (const generics for D=2..8)
//! and rayon for parallel queries. Returns flat arrays that Python reshapes to (n, k).

use kiddo::{KdTree, SquaredEuclidean};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Build a kd-tree and query k nearest neighbors for all points.
/// Returns (nn_dist, nn_indices) as flat numpy arrays that Python reshapes to (n, k).
#[pyfunction]
pub fn knn_query<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let x_array = x.as_array();
    let n = x_array.nrows();
    let dim = x_array.ncols();

    match dim {
        2 => knn_query_dim::<2>(py, &x_array, n, k),
        3 => knn_query_dim::<3>(py, &x_array, n, k),
        4 => knn_query_dim::<4>(py, &x_array, n, k),
        5 => knn_query_dim::<5>(py, &x_array, n, k),
        6 => knn_query_dim::<6>(py, &x_array, n, k),
        7 => knn_query_dim::<7>(py, &x_array, n, k),
        8 => knn_query_dim::<8>(py, &x_array, n, k),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimensionality {dim} not supported (must be 2-8)"
        ))),
    }
}

fn knn_query_dim<'py, const D: usize>(
    py: Python<'py>,
    x_array: &ndarray::ArrayView2<f64>,
    n: usize,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let mut tree: KdTree<f64, D> = KdTree::with_capacity(n);
    for i in 0..n {
        let mut point = [0.0f64; D];
        for d in 0..D {
            point[d] = x_array[[i, d]];
        }
        tree.add(&point, i as u64);
    }

    let results: Vec<(Vec<f64>, Vec<i64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut point = [0.0f64; D];
            for d in 0..D {
                point[d] = x_array[[i, d]];
            }
            let neighbors = tree.nearest_n::<SquaredEuclidean>(&point, k);
            let mut dists = Vec::with_capacity(k);
            let mut indices = Vec::with_capacity(k);
            for nb in &neighbors {
                dists.push(nb.distance.sqrt());
                indices.push(nb.item as i64);
            }
            (dists, indices)
        })
        .collect();

    let mut dist_data = Vec::with_capacity(n * k);
    let mut idx_data = Vec::with_capacity(n * k);

    for (dists, indices) in &results {
        dist_data.extend_from_slice(dists);
        idx_data.extend_from_slice(indices);
    }

    let nn_dist = PyArray1::from_vec(py, dist_data);
    let nn_idx = PyArray1::from_vec(py, idx_data);

    Ok((nn_dist, nn_idx))
}

/// Build a kd-tree from x_train, query x_query for k nearest neighbors.
/// Returns (nn_dist, nn_indices) as flat numpy arrays, shape (n_query * k,).
#[pyfunction]
pub fn knn_query_cross<'py>(
    py: Python<'py>,
    x_train: PyReadonlyArray2<'py, f64>,
    x_query: PyReadonlyArray2<'py, f64>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let train = x_train.as_array();
    let query = x_query.as_array();
    let dim = train.ncols();

    match dim {
        2 => knn_cross_dim::<2>(py, &train, &query, k),
        3 => knn_cross_dim::<3>(py, &train, &query, k),
        4 => knn_cross_dim::<4>(py, &train, &query, k),
        5 => knn_cross_dim::<5>(py, &train, &query, k),
        6 => knn_cross_dim::<6>(py, &train, &query, k),
        7 => knn_cross_dim::<7>(py, &train, &query, k),
        8 => knn_cross_dim::<8>(py, &train, &query, k),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimensionality {dim} not supported (must be 2-8)"
        ))),
    }
}

fn knn_cross_dim<'py, const D: usize>(
    py: Python<'py>,
    x_train: &ndarray::ArrayView2<f64>,
    x_query: &ndarray::ArrayView2<f64>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<i64>>)> {
    let n_train = x_train.nrows();
    let n_query = x_query.nrows();

    let mut tree: KdTree<f64, D> = KdTree::with_capacity(n_train);
    for i in 0..n_train {
        let mut point = [0.0f64; D];
        for d in 0..D {
            point[d] = x_train[[i, d]];
        }
        tree.add(&point, i as u64);
    }

    let results: Vec<(Vec<f64>, Vec<i64>)> = (0..n_query)
        .into_par_iter()
        .map(|i| {
            let mut point = [0.0f64; D];
            for d in 0..D {
                point[d] = x_query[[i, d]];
            }
            let neighbors = tree.nearest_n::<SquaredEuclidean>(&point, k);
            let mut dists = Vec::with_capacity(k);
            let mut indices = Vec::with_capacity(k);
            for nb in &neighbors {
                dists.push(nb.distance.sqrt());
                indices.push(nb.item as i64);
            }
            (dists, indices)
        })
        .collect();

    let mut dist_data = Vec::with_capacity(n_query * k);
    let mut idx_data = Vec::with_capacity(n_query * k);

    for (dists, indices) in &results {
        dist_data.extend_from_slice(dists);
        idx_data.extend_from_slice(indices);
    }

    let nn_dist = PyArray1::from_vec(py, dist_data);
    let nn_idx = PyArray1::from_vec(py, idx_data);

    Ok((nn_dist, nn_idx))
}
