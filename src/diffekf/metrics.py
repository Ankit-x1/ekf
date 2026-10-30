"""Evaluation metrics.

Inclination RMSE and heading drift are reported separately and never combined.
Per-trial values are paired across methods; bootstrap confidence intervals are
computed on the improvement over the best tuned baseline. NEES (3-dof attitude
error) and NIS (accelerometer innovation) are compared against chi-square
95% bounds.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .quaternion import geodesic_angle, qconj, qmul

# Chi-square 95% region, 3 degrees of freedom (two-sided): [ppf(0.025), ppf(0.975)].
# ppf(0.95, 3) = 7.8147 (one-sided upper 95% bound).
CHI2_3_95_ONE_SIDED_UPPER: float = 7.814727903251178


def _error_arrays(q_est: np.ndarray, q_gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    q_est = q_est / np.linalg.norm(q_est, axis=1, keepdims=True)
    q_gt = q_gt / np.linalg.norm(q_gt, axis=1, keepdims=True)
    q_err = np.stack([qmul(q_est[i], qconj(q_gt[i])) for i in range(len(q_est))])
    q_err = np.where(q_err[:, 0:1] < 0, -q_err, q_err)
    w = q_err[:, 0]
    z = q_err[:, 3]
    inclination = 2 * np.arccos(np.clip(np.sqrt(w * w + z * z), 0.0, 1.0))
    heading = 2 * np.arctan(np.abs(z / np.maximum(np.abs(w), 1e-12)))
    return inclination, heading


def inclination_rmse_deg(q_est, q_gt, movement=None) -> float:
    """Inclination RMSE in degrees (tilt away from vertical)."""
    incl, _ = _error_arrays(np.asarray(q_est, dtype=np.float64), np.asarray(q_gt, dtype=np.float64))
    incl = np.asarray(incl)
    if movement is not None:
        movement = np.asarray(movement)
        if movement.any():
            incl = incl[movement]
    return float(np.degrees(np.sqrt(np.mean(incl ** 2))))


def heading_drift_deg(q_est, q_gt, movement=None) -> float:
    """Heading error (yaw about world z) RMSE in degrees, reported separately."""
    _, heading = _error_arrays(np.asarray(q_est, dtype=np.float64), np.asarray(q_gt, dtype=np.float64))
    heading = np.asarray(heading)
    if movement is not None:
        movement = np.asarray(movement)
        if movement.any():
            heading = heading[movement]
    return float(np.degrees(np.sqrt(np.mean(heading ** 2))))


def total_error_deg(q_est, q_gt, movement=None) -> float:
    """Total geodesic error RMSE (degrees) -- supplementary only, never headline."""
    q_est = np.asarray(q_est, dtype=np.float64)
    q_gt = np.asarray(q_gt, dtype=np.float64)
    ang = np.array([geodesic_angle(q_est[i], q_gt[i]) for i in range(len(q_est))])
    if movement is not None:
        movement = np.asarray(movement)
        if movement.any():
            ang = ang[movement]
    return float(np.degrees(np.sqrt(np.mean(ang ** 2))))


def nees(attitude_error_vec, cov_attitude) -> np.ndarray:
    """NEES = e^T P^{-1} e for the 3-dof attitude error, per time step."""
    e = np.asarray(attitude_error_vec, dtype=np.float64)
    P = np.asarray(cov_attitude, dtype=np.float64)
    Pinv = np.linalg.inv(P + 1e-9 * np.eye(3))
    return np.einsum("ti,tij,tj->t", e, Pinv, e)


def nees_bounds(alpha: float = 0.05, dof: int = 3) -> Tuple[float, float]:
    """Chi-square bounds for NEES/NIS."""
    from scipy.stats import chi2

    lo = float(chi2.ppf(alpha / 2, dof))
    hi = float(chi2.ppf(1 - alpha / 2, dof))
    return lo, hi


def nees_fraction_inside(nees_values, alpha: float = 0.05) -> float:
    """Fraction of time with NEES inside the chi-square 95% region."""
    _, hi = nees_bounds(alpha)
    nees_values = np.asarray(nees_values, dtype=np.float64)
    return float(np.mean(nees_values <= hi))


def nis_values(innovation, innovation_cov) -> np.ndarray:
    """NIS = r^T S^{-1} r for the accelerometer innovation, per time step."""
    r = np.asarray(innovation, dtype=np.float64)
    S = np.asarray(innovation_cov, dtype=np.float64)
    Sinv = np.linalg.inv(S + 1e-9 * np.eye(3))
    return np.einsum("ti,tij,tj->t", r, Sinv, r)


def nis_fraction_inside(innovation, innovation_cov, alpha: float = 0.05) -> float:
    _, hi = nees_bounds(alpha)
    return float(np.mean(nis_values(innovation, innovation_cov) <= hi))


def paired_bootstrap_ci(improvements, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Paired bootstrap percentile CI for the mean improvement.

    ``improvements`` = baseline_metric - learned_metric per test trial
    (positive means the learned filter is better).
    """
    improvements = np.asarray(improvements, dtype=np.float64)
    n = len(improvements)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(improvements, size=n, replace=True)
        means[i] = np.mean(sample)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(np.mean(improvements)), lo, hi
