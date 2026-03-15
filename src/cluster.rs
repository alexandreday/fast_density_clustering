//! Core clustering algorithms: density gradient, cluster assignment, and stability merging.
//!
//! - `compute_delta`: for each point, find the nearest neighbor with higher density
//!   and build the density graph (adjacency list of density-gradient edges).
//! - `assign_cluster`: iterative DFS from cluster centers through the density graph.
//! - `find_nh_tree_search`: BFS neighborhood expansion within a density threshold.
//! - `check_cluster_stability`: one pass of false-positive center removal.
//! - `stability_loop`: fused assign+check loop that runs entirely in Rust until
//!   convergence, avoiding repeated Python↔Rust data conversion.
//! - `adaptive_trim_size`: compute effective neighborhood after bandwidth-based trimming.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute delta (distance to nearest higher-density neighbor) for all points.
///
/// Returns (delta, nn_delta, idx_centers, density_graph)
#[pyfunction]
pub fn compute_delta<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    nn_list: PyReadonlyArray2<'py, i64>,
    nn_dist: PyReadonlyArray2<'py, f64>,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    PyObject,
)> {
    let x_arr = x.as_array();
    let rho_arr = rho.as_array();
    let nn_list_arr = nn_list.as_array();
    let nn_dist_arr = nn_dist.as_array();

    let n = x_arr.nrows();
    let n_feat = x_arr.ncols();
    let k = nn_list_arr.ncols();

    // Compute max extent
    let mut maxdist_sq = 0.0;
    for d in 0..n_feat {
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        for i in 0..n {
            let v = x_arr[[i, d]];
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        let diff = max_val - min_val;
        maxdist_sq += diff * diff;
    }
    let maxdist = maxdist_sq.sqrt();

    // Phase 1: parallel scan
    let per_point: Vec<(f64, i64, Option<(usize, usize)>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let rho_i = rho_arr[i];
            for j in 1..k {
                let neighbor = nn_list_arr[[i, j]] as usize;
                if rho_arr[neighbor] > rho_i + 1e-8 {
                    return (nn_dist_arr[[i, j]], nn_list_arr[[i, j]], Some((neighbor, i)));
                }
            }
            (maxdist, -1i64, None)
        })
        .collect();

    // Phase 2: serial graph build
    let mut delta = Vec::with_capacity(n);
    let mut nn_delta = Vec::with_capacity(n);
    let mut density_graph: Vec<Vec<i64>> = vec![Vec::new(); n];

    for (_i, (d, nd, edge)) in per_point.into_iter().enumerate() {
        delta.push(d);
        nn_delta.push(nd);
        if let Some((parent, child)) = edge {
            density_graph[parent].push(child as i64);
        }
    }

    // Find centers
    let threshold = 0.999 * maxdist;
    let idx_centers: Vec<i64> = (0..n)
        .filter(|&i| delta[i] > threshold)
        .map(|i| i as i64)
        .collect();

    let py_delta = PyArray1::from_vec(py, delta);
    let py_nn_delta = PyArray1::from_vec(py, nn_delta);
    let py_centers = PyArray1::from_vec(py, idx_centers);

    let py_graph = pyo3::types::PyList::new(
        py,
        density_graph.iter().map(|children| {
            pyo3::types::PyList::new(py, children.iter().copied()).unwrap()
        }),
    )?;

    Ok((py_delta, py_nn_delta, py_centers, py_graph.into_any().unbind()))
}

/// Assign cluster labels by traversing the density tree (iterative DFS).
#[pyfunction]
pub fn assign_cluster<'py>(
    py: Python<'py>,
    idx_centers: PyReadonlyArray1<'py, i64>,
    nn_delta: PyReadonlyArray1<'py, i64>,
    density_graph: Vec<Vec<i64>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let centers = idx_centers.as_slice()?;
    let n = nn_delta.as_slice()?.len();

    let mut labels = vec![-1i64; n];

    for (label, &center) in centers.iter().enumerate() {
        let c = center as usize;
        labels[c] = label as i64;

        let mut stack: Vec<usize> = Vec::new();
        for &child in &density_graph[c] {
            stack.push(child as usize);
        }

        while let Some(node) = stack.pop() {
            labels[node] = label as i64;
            for &child in &density_graph[node] {
                stack.push(child as usize);
            }
        }
    }

    Ok(PyArray1::from_vec(py, labels))
}

/// BFS neighborhood search within density threshold.
#[pyfunction]
pub fn find_nh_tree_search<'py>(
    py: Python<'py>,
    idx: usize,
    eta: f64,
    rho: PyReadonlyArray1<'py, f64>,
    nn_list: PyReadonlyArray2<'py, i64>,
    cluster_label: PyReadonlyArray1<'py, i64>,
    search_size: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let rho_arr = rho.as_slice()?;
    let nn_arr = nn_list.as_array();
    let labels = cluster_label.as_slice()?;
    let n = rho_arr.len();
    let k = nn_arr.ncols();
    let ss = search_size.min(k);

    let current_label = labels[idx];
    let mut is_nh = vec![false; n];
    let mut new_leaves = vec![false; n];

    for j in 0..ss {
        let nb = nn_arr[[idx, j]] as usize;
        new_leaves[nb] = true;
    }

    for i in 0..n {
        if new_leaves[i] && rho_arr[i] > eta {
            is_nh[i] = true;
        }
    }

    loop {
        let mut update = false;
        let mut next_leaves = vec![false; n];

        for i in 0..n {
            if !new_leaves[i] || labels[i] != current_label {
                continue;
            }
            for j in 0..ss {
                let nb = nn_arr[[i, j]] as usize;
                if !is_nh[nb] && rho_arr[nb] > eta {
                    is_nh[nb] = true;
                    next_leaves[nb] = true;
                    update = true;
                }
            }
        }

        if !update {
            break;
        }
        new_leaves = next_leaves;
    }

    let nh_indices: Vec<i64> = (0..n).filter(|&i| is_nh[i]).map(|i| i as i64).collect();
    Ok(PyArray1::from_vec(py, nh_indices))
}

/// Check cluster stability: identify and remove false positive centers.
#[pyfunction]
pub fn check_cluster_stability<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    idx_centers: PyReadonlyArray1<'py, i64>,
    nn_delta: PyReadonlyArray1<'py, i64>,
    delta: PyReadonlyArray1<'py, f64>,
    nn_list: PyReadonlyArray2<'py, i64>,
    cluster_label: PyReadonlyArray1<'py, i64>,
    density_graph: Vec<Vec<i64>>,
    threshold: f64,
    search_size: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    usize,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    PyObject,
)> {
    let x_arr = x.as_array();
    let rho_arr = rho.as_slice()?;
    let centers = idx_centers.as_slice()?;
    let labels = cluster_label.as_slice()?;
    let nn_arr = nn_list.as_array();
    let n = rho_arr.len();
    let k = nn_arr.ncols();
    let ss = search_size.min(k);
    let n_feat = x_arr.ncols();

    let mut nn_delta_vec: Vec<i64> = nn_delta.as_slice()?.to_vec();
    let mut delta_vec: Vec<f64> = delta.as_slice()?.to_vec();
    let mut dg = density_graph;

    let mut n_false_pos = 0usize;
    let mut true_centers: Vec<i64> = Vec::new();

    for &cidx in centers {
        let idx = cidx as usize;
        let rho_center = rho_arr[idx];
        let delta_rho = rho_center - threshold;

        let nh: Vec<usize> = if threshold < 1e-3 {
            (1..ss).map(|j| nn_arr[[idx, j]] as usize).collect()
        } else {
            bfs_nh_search(idx, delta_rho, rho_arr, &nn_arr, labels, n, ss)
        };

        // Find unique cluster labels in neighborhood
        let mut label_set: Vec<i64> = nh.iter().map(|&i| labels[i]).collect();
        label_set.sort();
        label_set.dedup();

        // Find center with highest density among neighboring clusters
        let mut best_center_idx = idx;
        let mut best_rho = rho_arr[idx];
        for &lbl in &label_set {
            if (lbl as usize) < centers.len() {
                let center_of_lbl = centers[lbl as usize] as usize;
                if rho_arr[center_of_lbl] > best_rho {
                    best_rho = rho_arr[center_of_lbl];
                    best_center_idx = center_of_lbl;
                }
            }
        }

        if rho_arr[idx] < best_rho && idx != best_center_idx {
            nn_delta_vec[idx] = best_center_idx as i64;
            let mut dist = 0.0;
            for d in 0..n_feat {
                let diff = x_arr[[best_center_idx, d]] - x_arr[[idx, d]];
                dist += diff * diff;
            }
            delta_vec[idx] = dist.sqrt();
            dg[best_center_idx].push(idx as i64);
            n_false_pos += 1;
        } else {
            true_centers.push(cidx);
        }
    }

    let py_centers = PyArray1::from_vec(py, true_centers);
    let py_nn_delta = PyArray1::from_vec(py, nn_delta_vec);
    let py_delta = PyArray1::from_vec(py, delta_vec);

    let py_graph = pyo3::types::PyList::new(
        py,
        dg.iter().map(|children| {
            pyo3::types::PyList::new(py, children.iter().copied()).unwrap()
        }),
    )?;

    Ok((py_centers, n_false_pos, py_nn_delta, py_delta, py_graph.into_any().unbind()))
}

/// Internal BFS neighborhood search.
fn bfs_nh_search(
    idx: usize,
    eta: f64,
    rho: &[f64],
    nn: &ndarray::ArrayView2<i64>,
    labels: &[i64],
    n: usize,
    ss: usize,
) -> Vec<usize> {
    let current_label = labels[idx];
    let mut is_nh = vec![false; n];
    let mut new_leaves = vec![false; n];

    for j in 0..ss {
        let nb = nn[[idx, j]] as usize;
        new_leaves[nb] = true;
        if rho[nb] > eta {
            is_nh[nb] = true;
        }
    }

    loop {
        let mut update = false;
        let mut next_leaves = vec![false; n];

        for i in 0..n {
            if !new_leaves[i] || labels[i] != current_label {
                continue;
            }
            for j in 0..ss {
                let nb = nn[[i, j]] as usize;
                if !is_nh[nb] && rho[nb] > eta {
                    is_nh[nb] = true;
                    next_leaves[nb] = true;
                    update = true;
                }
            }
        }

        if !update {
            break;
        }
        new_leaves = next_leaves;
    }

    (0..n).filter(|&i| is_nh[i]).collect()
}

/// Full stability loop: assign clusters + check stability until convergence.
/// This avoids repeated Python↔Rust data conversion by keeping everything in Rust.
///
/// Returns (final_idx_centers, final_cluster_label, final_nn_delta, final_delta, final_density_graph)
#[pyfunction]
pub fn stability_loop<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    rho: PyReadonlyArray1<'py, f64>,
    idx_centers_in: PyReadonlyArray1<'py, i64>,
    nn_delta_in: PyReadonlyArray1<'py, i64>,
    delta_in: PyReadonlyArray1<'py, f64>,
    nn_list: PyReadonlyArray2<'py, i64>,
    density_graph_in: Vec<Vec<i64>>,
    threshold: f64,
    search_size: usize,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,  // idx_centers
    Bound<'py, PyArray1<i64>>,  // cluster_label
    Bound<'py, PyArray1<i64>>,  // nn_delta
    Bound<'py, PyArray1<f64>>,  // delta
    PyObject,                    // density_graph
)> {
    let x_arr = x.as_array();
    let rho_arr = rho.as_slice()?;
    let nn_arr = nn_list.as_array();
    let n = rho_arr.len();
    let k = nn_arr.ncols();
    let ss = search_size.min(k);
    let n_feat = x_arr.ncols();

    let mut centers: Vec<i64> = idx_centers_in.as_slice()?.to_vec();
    let mut nn_delta_vec: Vec<i64> = nn_delta_in.as_slice()?.to_vec();
    let mut delta_vec: Vec<f64> = delta_in.as_slice()?.to_vec();
    let mut dg = density_graph_in;

    loop {
        // assign_cluster
        let mut labels = vec![-1i64; n];
        for (label, &center) in centers.iter().enumerate() {
            let c = center as usize;
            labels[c] = label as i64;
            let mut stack: Vec<usize> = Vec::new();
            for &child in &dg[c] {
                stack.push(child as usize);
            }
            while let Some(node) = stack.pop() {
                labels[node] = label as i64;
                for &child in &dg[node] {
                    stack.push(child as usize);
                }
            }
        }

        // check_cluster_stability
        let mut n_false_pos = 0usize;
        let mut true_centers: Vec<i64> = Vec::new();

        for &cidx in &centers {
            let idx = cidx as usize;
            let rho_center = rho_arr[idx];
            let delta_rho = rho_center - threshold;

            let nh: Vec<usize> = if threshold < 1e-3 {
                (1..ss).map(|j| nn_arr[[idx, j]] as usize).collect()
            } else {
                bfs_nh_search(idx, delta_rho, rho_arr, &nn_arr, &labels, n, ss)
            };

            let mut label_set: Vec<i64> = nh.iter().map(|&i| labels[i]).collect();
            label_set.sort();
            label_set.dedup();

            let mut best_center_idx = idx;
            let mut best_rho = rho_arr[idx];
            for &lbl in &label_set {
                if (lbl as usize) < centers.len() {
                    let center_of_lbl = centers[lbl as usize] as usize;
                    if rho_arr[center_of_lbl] > best_rho {
                        best_rho = rho_arr[center_of_lbl];
                        best_center_idx = center_of_lbl;
                    }
                }
            }

            if rho_arr[idx] < best_rho && idx != best_center_idx {
                nn_delta_vec[idx] = best_center_idx as i64;
                let mut dist = 0.0;
                for d in 0..n_feat {
                    let diff = x_arr[[best_center_idx, d]] - x_arr[[idx, d]];
                    dist += diff * diff;
                }
                delta_vec[idx] = dist.sqrt();
                dg[best_center_idx].push(idx as i64);
                n_false_pos += 1;
            } else {
                true_centers.push(cidx);
            }
        }

        centers = true_centers;

        if n_false_pos == 0 {
            // Final assign
            let mut final_labels = vec![-1i64; n];
            for (label, &center) in centers.iter().enumerate() {
                let c = center as usize;
                final_labels[c] = label as i64;
                let mut stack: Vec<usize> = Vec::new();
                for &child in &dg[c] {
                    stack.push(child as usize);
                }
                while let Some(node) = stack.pop() {
                    final_labels[node] = label as i64;
                    for &child in &dg[node] {
                        stack.push(child as usize);
                    }
                }
            }

            let py_centers = PyArray1::from_vec(py, centers);
            let py_labels = PyArray1::from_vec(py, final_labels);
            let py_nn_delta = PyArray1::from_vec(py, nn_delta_vec);
            let py_delta = PyArray1::from_vec(py, delta_vec);
            let py_graph = pyo3::types::PyList::new(
                py,
                dg.iter().map(|children| {
                    pyo3::types::PyList::new(py, children.iter().copied()).unwrap()
                }),
            )?;
            return Ok((py_centers, py_labels, py_nn_delta, py_delta, py_graph.into_any().unbind()));
        }
    }
}

/// Compute effective neighborhood size after adaptive trimming.
#[pyfunction]
pub fn adaptive_trim_size(
    nn_dist: PyReadonlyArray2<f64>,
    bandwidth: f64,
    alpha: f64,
    min_nh: usize,
) -> PyResult<usize> {
    let dist = nn_dist.as_array();
    let n = dist.nrows();
    let k = dist.ncols();
    let cutoff = alpha * bandwidth;

    let mut counts: Vec<usize> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut count = 0;
            for j in 0..k {
                if dist[[i, j]] <= cutoff {
                    count += 1;
                }
            }
            count
        })
        .collect();

    counts.sort();
    let median = counts[n / 2];

    Ok(median.max(min_nh))
}
