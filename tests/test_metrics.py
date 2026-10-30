"""Metric sanity checks: bootstrap CI behaviour and NEES/NIS bounds.
"""

from __future__ import annotations

import numpy as np

from diffekf.metrics import (
    CHI2_3_95_ONE_SIDED_UPPER,
    inclination_rmse_deg,
    nees_bounds,
    nees_fraction_inside,
    nis_fraction_inside,
    paired_bootstrap_ci,
)
from scipy.stats import chi2


def test_bootstrap_ci_recovers_sign():
    improvements = np.array([0.5, 0.6, 0.4, 0.7, 0.55])
    mean, lo, hi = paired_bootstrap_ci(improvements, n_resamples=500, seed=0)
    assert lo > 0 and hi > 0
    improvements_neg = -improvements
    mean2, lo2, hi2 = paired_bootstrap_ci(improvements_neg, n_resamples=500, seed=0)
    assert lo2 < 0 and hi2 < 0


def test_nees_bounds_known_value():
    lo, hi = nees_bounds()
    assert abs(lo - chi2.ppf(0.025, 3)) < 1e-9
    assert abs(hi - chi2.ppf(0.975, 3)) < 1e-9
    assert abs(CHI2_3_95_ONE_SIDED_UPPER - chi2.ppf(0.95, 3)) < 1e-9


def test_nees_and_nis_fractions_in_range():
    nees_vals = np.array([0.5, 2.0, 8.0, 1.0])
    assert 0.0 <= nees_fraction_inside(nees_vals) <= 1.0
    innov = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    S = np.array([np.eye(3), np.eye(3)])
    assert 0.0 <= nis_fraction_inside(innov, S) <= 1.0


def test_inclination_rmse_identity_is_zero():
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (10, 1))
    assert inclination_rmse_deg(q, q) == 0.0
