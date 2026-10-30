"""Gyro-only integration baseline (drifts; no gravity correction).

Same initialisation as the EKF: tilt from the first static accelerometer
samples, heading from the first ground-truth sample, never ground truth after.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..quaternion import exp_so3, qmul, qnormalize, tilt_heading_init


def run(
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: np.ndarray,
    yaw_radians: float,
    init_window: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate gyro only.

    Parameters
    ----------
    gyro : (T, 3) rad / s
    accel : (T, 3) m / s^2 (used only for the static tilt initialisation)
    dt : (T,) seconds
    yaw_radians : heading from the first ground-truth sample (initialisation only)
    init_window : samples used for the static tilt initialisation

    Returns
    -------
    quat : (T, 4) estimated attitude
    bias : (T, 3) gyro bias estimate (stays at zero here)
    """
    q0 = tilt_heading_init(np.mean(accel[:init_window], axis=0), yaw_radians)
    T = len(gyro)
    quat = np.zeros((T, 4))
    bias = np.zeros((T, 3))
    q = q0.copy()
    b = np.zeros(3)
    for k in range(T):
        q = qnormalize(qmul(q, exp_so3((gyro[k] - b) * dt[k])))
        quat[k] = q
        bias[k] = b
    return quat, bias
