"""Complementary filter baseline (gyro integration + accelerometer tilt correction).

The gyro propagates the attitude; the accelerometer gravity direction corrects
tilt (never heading) with gain ``kp``. Gated by accelerometer-magnitude proximity
to 1 g so linear acceleration does not corrupt the tilt estimate.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..quaternion import (
    GRAVITY,
    exp_so3,
    log_so3,
    qmul,
    qnormalize,
    quat_to_rotmat,
    skew,
    tilt_heading_init,
)


def run(
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: np.ndarray,
    yaw_radians: float,
    kp: float = 2.0,
    init_window: int = 100,
    gate_deadband: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the complementary filter.

    Parameters
    ----------
    gyro : (T, 3) rad / s
    accel : (T, 3) m / s^2
    dt : (T,) seconds
    yaw_radians : heading initialisation from the first ground-truth sample only
    kp : tilt-correction gain (1 / s)
    init_window : static samples for tilt initialisation
    gate_deadband : m / s^2 deadband around 1 g for the accel-magnitude gate

    Returns
    -------
    quat : (T, 4)
    bias : (T, 3) (stays at zero; no bias estimation)
    """
    q0 = tilt_heading_init(np.mean(accel[:init_window], axis=0), yaw_radians)
    T = len(gyro)
    quat = np.zeros((T, 4))
    bias = np.zeros((T, 3))
    q = q0.copy()
    b = np.zeros(3)
    g_up = np.array([0.0, 0.0, GRAVITY])
    for k in range(T):
        # propagate
        q = qnormalize(qmul(q, exp_so3((gyro[k] - b) * dt[k])))
        # accelerometer tilt correction
        g_hat = quat_to_rotmat(q).T @ g_up
        a = accel[k]
        norm_a = float(np.linalg.norm(a))
        if abs(norm_a - GRAVITY) <= gate_deadband:
            a_hat = a / max(norm_a, 1e-9)
            g_hat = g_hat / max(np.linalg.norm(g_hat), 1e-9)
            # rotation aligning g_hat with a_hat: delta_theta = kp * cross(g_hat, a_hat)
            delta_theta = kp * np.cross(g_hat, a_hat) * dt[k]
            q = qnormalize(qmul(q, exp_so3(delta_theta)))
        quat[k] = q
        bias[k] = b
    return quat, bias
