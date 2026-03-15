//! fdc_rs — Rust-accelerated core for the Fast Density Clustering (fdc) Python package.
//!
//! This crate provides PyO3 bindings for the performance-critical parts of the FDC algorithm:
//!
//! - **knn**: k-nearest-neighbor search via the `kiddo` KD-tree (supports dimensions 2–8,
//!   parallelized with rayon). Replaces sklearn's `NearestNeighbors`.
//!
//! - **kde**: Epanechnikov kernel density estimation from pre-computed k-NN distances,
//!   plus Brent's method bandwidth optimization. Replaces numpy-vectorized KDE and
//!   scipy's `fminbound`.
//!
//! - **cluster**: density gradient computation (`compute_delta`), cluster assignment
//!   (`assign_cluster`), BFS neighborhood search, and a fused stability convergence
//!   loop (`stability_loop`) that keeps all data in Rust to avoid repeated Python↔Rust
//!   conversion overhead.
//!
//! The Python `fdc` package imports this module opportunistically (`try: import fdc_rs`)
//! and falls back to pure-Python implementations when it is not available.

use pyo3::prelude::*;

mod cluster;
mod kde;
mod knn;

/// Fast Density Clustering — Rust accelerated core.
#[pymodule]
fn fdc_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // k-NN
    m.add_function(wrap_pyfunction!(knn::knn_query, m)?)?;
    m.add_function(wrap_pyfunction!(knn::knn_query_cross, m)?)?;

    // KDE
    m.add_function(wrap_pyfunction!(kde::epanechnikov_kde, m)?)?;
    m.add_function(wrap_pyfunction!(kde::find_optimal_bandwidth, m)?)?;
    m.add_function(wrap_pyfunction!(kde::bandwidth_estimate, m)?)?;
    m.add_function(wrap_pyfunction!(kde::round_float, m)?)?;

    // Clustering
    m.add_function(wrap_pyfunction!(cluster::compute_delta, m)?)?;
    m.add_function(wrap_pyfunction!(cluster::assign_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(cluster::find_nh_tree_search, m)?)?;
    m.add_function(wrap_pyfunction!(cluster::check_cluster_stability, m)?)?;
    m.add_function(wrap_pyfunction!(cluster::stability_loop, m)?)?;
    m.add_function(wrap_pyfunction!(cluster::adaptive_trim_size, m)?)?;

    Ok(())
}
