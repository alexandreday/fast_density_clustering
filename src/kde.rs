//! Kernel Density Estimation (Epanechnikov kernel) and bandwidth optimization.
//!
//! - `epanechnikov_kde`: compute log-density from pre-computed k-NN distances.
//! - `find_optimal_bandwidth`: Brent's method on test-set negative log-likelihood.
//! - `bandwidth_estimate`: heuristic bounds (h_min, h_max) from nearest-neighbor distances.
//! - `round_float`: round to first significant digit (for xtol computation).
//!
//! The KDE evaluation is parallelized with rayon. The bandwidth optimizer calls KDE
//! internally via `epanechnikov_kde_raw` (Vec-based, no Python round-trips per iteration).

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Volume of unit d-ball
fn volume_unit_ball(dim: usize) -> f64 {
    match dim {
        1 => 2.0,
        2 => std::f64::consts::PI,
        3 => 4.0 / 3.0 * std::f64::consts::PI,
        _ => {
            let half_d = dim as f64 / 2.0;
            std::f64::consts::PI.powf(half_d) / gamma(half_d + 1.0)
        }
    }
}

/// Lanczos approximation for gamma function
fn gamma(x: f64) -> f64 {
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let g = 7.0_f64;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];
        let mut sum = c[0];
        for (i, &ci) in c.iter().enumerate().skip(1) {
            sum += ci / (x + i as f64);
        }
        let t = x + g + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

/// Compute log-density using Epanechnikov kernel from pre-computed k-NN distances.
#[pyfunction]
pub fn epanechnikov_kde<'py>(
    py: Python<'py>,
    nn_dist: PyReadonlyArray2<'py, f64>,
    bandwidth: f64,
    dim: usize,
    n_total: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let dist = nn_dist.as_array();
    let n = dist.nrows();
    let k = dist.ncols();

    let v_d = volume_unit_ball(dim);
    let c_d = (dim as f64 + 2.0) / (2.0 * v_d);
    let norm = n_total as f64 * bandwidth.powi(dim as i32);

    let log_density: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut sum = 0.0;
            for j in 0..k {
                let u = dist[[i, j]] / bandwidth;
                if u <= 1.0 {
                    sum += c_d * (1.0 - u * u);
                }
            }
            let density = (sum / norm).max(1e-300);
            density.ln()
        })
        .collect();

    Ok(PyArray1::from_vec(py, log_density))
}

/// Compute log-density from raw distance data (Vec-based, for internal use).
pub fn epanechnikov_kde_raw(
    nn_dist: &[f64],
    k: usize,
    bandwidth: f64,
    dim: usize,
    n_total: usize,
) -> Vec<f64> {
    let n = nn_dist.len() / k;
    let v_d = volume_unit_ball(dim);
    let c_d = (dim as f64 + 2.0) / (2.0 * v_d);
    let norm = n_total as f64 * bandwidth.powi(dim as i32);

    (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &nn_dist[i * k..(i + 1) * k];
            let mut sum = 0.0;
            for &d in row {
                let u = d / bandwidth;
                if u <= 1.0 {
                    sum += c_d * (1.0 - u * u);
                }
            }
            let density = (sum / norm).max(1e-300);
            density.ln()
        })
        .collect()
}

/// Brent's method (fminbound) for 1D optimization.
fn brent_minimize<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    xtol: f64,
    maxfun: usize,
) -> (f64, f64, usize) {
    let golden: f64 = 0.3819660112501051;

    let mut xa = a;
    let mut xb = b;

    let mut x: f64 = xa + golden * (xb - xa);
    let mut w: f64 = x;
    let mut v: f64 = x;
    let mut e: f64 = 0.0;

    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;
    let mut nfev: usize = 1;

    for _ in 0..maxfun {
        let xm = 0.5 * (xa + xb);
        let tol1 = xtol * x.abs() + 1e-10;
        let tol2 = 2.0 * tol1;

        if (x - xm).abs() <= (tol2 - 0.5 * (xb - xa)) {
            break;
        }

        let mut d: f64;
        if e.abs() > tol1 {
            // Parabolic interpolation
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            let r2 = e;
            if p.abs() < (0.5 * q * r2).abs() && p > q * (xa - x) && p < q * (xb - x) {
                d = p / q;
                let u = x + d;
                if (u - xa) < tol2 || (xb - u) < tol2 {
                    d = if x < xm { tol1 } else { -tol1 };
                }
            } else {
                e = if x < xm { xb - x } else { xa - x };
                d = golden * e;
            }
        } else {
            e = if x < xm { xb - x } else { xa - x };
            d = golden * e;
        }

        let u: f64 = if d.abs() >= tol1 {
            x + d
        } else {
            x + if d > 0.0 { tol1 } else { -tol1 }
        };

        let fu = f(u);
        nfev += 1;

        if fu <= fx {
            if u < x {
                xb = x;
            } else {
                xa = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                xa = u;
            } else {
                xb = u;
            }
            if fu <= fw || (w - x).abs() < 1e-15 {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || (v - x).abs() < 1e-15 || (v - w).abs() < 1e-15 {
                v = u;
                fv = fu;
            }
        }
    }

    (x, fx, nfev)
}

/// Find optimal bandwidth using Brent's method on test-set negative log-likelihood.
#[pyfunction]
pub fn find_optimal_bandwidth(
    nn_dist_test: PyReadonlyArray2<f64>,
    h_min: f64,
    h_max: f64,
    dim: usize,
    n_train: usize,
    xtol: f64,
) -> PyResult<(f64, usize)> {
    let dist = nn_dist_test.as_array();
    let n = dist.nrows();
    let k = dist.ncols();

    let flat: Vec<f64> = dist.iter().copied().collect();

    let neg_ll = |h: f64| -> f64 {
        let log_dens = epanechnikov_kde_raw(&flat, k, h, dim, n_train);
        let sum: f64 = log_dens.iter().sum();
        -sum / n as f64
    };

    let (h_opt, _score, nfev) = brent_minimize(&neg_ll, h_min, h_max * 0.2, xtol, 100);

    Ok((h_opt, nfev))
}

/// Estimate bandwidth bounds from k-NN distances.
#[pyfunction]
pub fn bandwidth_estimate<'py>(
    nn_dist: PyReadonlyArray2<'py, f64>,
    x_train: PyReadonlyArray2<'py, f64>,
    x_test: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, f64, f64)> {
    let dist = nn_dist.as_array();
    let train = x_train.as_array();
    let test = x_test.as_array();
    let dim = train.ncols();

    let n = dist.nrows();
    let mut mean_nn2 = 0.0;
    for i in 0..n {
        let col = 1.min(dist.ncols() - 1);
        let d = dist[[i, col]];
        mean_nn2 += d * d;
    }
    mean_nn2 /= n as f64;
    let h_min = (mean_nn2 / dim as f64).sqrt();

    let max_size = train.nrows().min(test.nrows()).min(1000);
    let mut mean_dist2 = 0.0;
    for i in 0..max_size {
        let mut d2 = 0.0;
        for d in 0..dim {
            let diff = train[[i, d]] - test[[i, d]];
            d2 += diff * diff;
        }
        mean_dist2 += d2;
    }
    mean_dist2 /= max_size as f64;
    let h_max = (mean_dist2 / dim as f64).sqrt();

    let h_est = 10.0 * h_min;

    Ok((h_est, h_min, h_max))
}

/// Round a float to its first significant digit.
#[pyfunction]
pub fn round_float(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let magnitude = x.abs().log10().floor();
    10.0f64.powf(magnitude)
}
