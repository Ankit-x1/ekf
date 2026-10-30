"""Mahony AHRS baseline, 6-axis (gyro + accelerometer, no magnetometer).

Proportional-integral attitude estimator: the accelerometer gravity direction
yields a tilt error (proportional term) fed back on the gyro together with an
integral gyro-bias estimate. Heading is unobservable and drifts.

Reference: R. Mahony et al., "Nonlinear complementary filters on the special
orthogonal group" (2008).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..quaternion import GRAVITY, exp_so3, qmul, qnormalize, quat_to_rotmat, tilt_heading_init


def run(
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: np.ndarray,
    yaw_radians: float,
    Kp: float = 2.0,
    Ki: float = 0.0,
    init_window: int = 100,
    gate_deadband: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the 6-axis Mahony filter.

    Parameters
    ----------
    gyro : (T, 3) rad / s
    accel : (T, 3) m / s^2
    dt : (T,) seconds
    yaw_radians : heading initialisation from the first ground-truth sample only
    Kp : proportional gain on the tilt error
    Ki : integral gain (gyro-bias learning)
    init_window : static samples for tilt initialisation
    gate_deadband : m / s^2; skip accel correction when ||a|| far from g

    Returns
    -------
    quat : (T, 4)
    bias : (T, 3) learned gyro bias estimate (rad / s)
    """
    q0 = tilt_heading_init(np.mean(accel[:init_window], axis=0), yaw_radians)
    T = len(gyro)
    quat = np.zeros((T, 4))
    bias = np.zeros((T, 3))
    q = q0.copy()
    b = np.zeros(3)
    g_up = np.array([0.0, 0.0, 1.0])
    for k in range(T):
        a = accel[k]
        norm_a = float(np.linalg.norm(a))
        if norm_a > 1e-6 and abs(norm_a - GRAVITY) <= gate_deadband:
            a_hat = a / norm_a
            R = quat_to_rotmat(q)
            g_hat = R.T @ g_up          # predicted gravity direction, body frame
            g_hat = g_hat / max(np.linalg.norm(g_hat), 1e-9)
            e = np.cross(a_hat, g_hat)  # tilt error, body frame
            b = b + Ki * e * dt[k]
            omega_corr = gyro[k] - b + Kp * e
        else:
            omega_corr = gyro[k] - b
        q = qnormalize(qmul(q, exp_so3(omega_corr * dt[k])))
        quat[k] = q
        bias[k] = b
    return quat, bias
