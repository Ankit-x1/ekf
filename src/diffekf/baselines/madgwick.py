"""Madgwick AHRS baseline, 6-axis (gyro + accelerometer, no magnetometer).

Implementation of S. O. H. Madgwick's stochastic gradient descent attitude
estimate using only the accelerometer as the reference vector (world z up).
Quaternion order (w, x, y, z), Hamilton, body -> world -- matching this
repository. The filter estimates tilt from the accelerometer and integrates
the gyro; heading is unobservable and drifts.

Reference: Madgwick et al., "An efficient orientation filter for inertial and
inertial/magnetic sensor arrays" (2011), x-io Technologies.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..quaternion import GRAVITY, exp_so3, qmul, qnormalize, tilt_heading_init


def run(
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: np.ndarray,
    yaw_radians: float,
    beta: float = 0.1,
    init_window: int = 100,
    gate_deadband: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the 6-axis Madgwick filter.

    Parameters
    ----------
    gyro : (T, 3) rad / s
    accel : (T, 3) m / s^2
    dt : (T,) seconds
    yaw_radians : heading initialisation from the first ground-truth sample only
    beta : algorithm gain
    init_window : static samples for tilt initialisation
    gate_deadband : m / s^2; skip the accel correction when ||a|| is far from g

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

    for k in range(T):
        w, x, y, z = q
        wx, wy, wz = w * x, w * y, w * z
        x2, y2, z2 = x * x, y * y, z * z

        # --- predicted rotation matrix column 3 = R(q).T @ world z (body frame)
        r31 = 2.0 * (x * z + wy)      # R[2,0]
        r32 = 2.0 * (y * z - wx)      # R[2,1]
        r33 = 1.0 - 2.0 * (x2 + y2)   # R[2,2]

        a = accel[k]
        norm_a = float(np.linalg.norm(a))
        if norm_a > 1e-6 and abs(norm_a - GRAVITY) <= gate_deadband:
            ax, ay, az = a / norm_a
            # Gradient-descent objective: minimise |R(q).T @ [0,0,1] - a_hat|
            f1 = 2.0 * (x * z - w * y) - ax   # R[0,2] ... note convention below
            f2 = 2.0 * (y * z + w * x) - ay
            f3 = 2.0 * (0.5 - x2 - y2) - az
            # Jacobian rows df/d(q) ; gradient step direction = J^T f
            j11, j12, j13, j14 = -2.0 * y, 2.0 * z, -2.0 * w, 2.0 * x
            j21, j22, j23, j24 = 2.0 * x, 2.0 * y, 2.0 * w, 2.0 * z
            j31, j32, j33, j34 = 0.0, -4.0 * x, -4.0 * y, 0.0
            step1 = j11 * f1 + j21 * f2 + j31 * f3
            step2 = j12 * f1 + j22 * f2 + j32 * f3
            step3 = j13 * f1 + j23 * f2 + j33 * f3
            step4 = j14 * f1 + j24 * f2 + j34 * f3
            norm_step = np.sqrt(step1**2 + step2**2 + step3**2 + step4**2)
            if norm_step > 0:
                q_dot = np.array([w, x, y, z], dtype=float)
                # q_dot = 0.5 q (x) (0, omega) - beta * step / |step|
                gx, gy, gz = gyro[k]
                qw_dot = 0.5 * (-x * gx - y * gy - z * gz)
                qx_dot = 0.5 * (w * gx + y * gz - z * gy)
                qy_dot = 0.5 * (w * gy - x * gz + z * gx)
                qz_dot = 0.5 * (w * gz + x * gy - y * gx)
                q_dot = np.array([qw_dot, qx_dot, qy_dot, qz_dot])
                q_dot -= beta * np.array([step1, step2, step3, step4]) / norm_step
                q = q + q_dot * dt[k]
        else:
            gx, gy, gz = gyro[k]
            q_dot = 0.5 * np.array(
                [-x * gx - y * gy - z * gz,
                 w * gx + y * gz - z * gy,
                 w * gy - x * gz + z * gx,
                 w * gz + x * gy - y * gx]
            )
            q = q + q_dot * dt[k]
        q = qnormalize(q)
        quat[k] = q
        bias[k] = np.zeros(3)
    return quat, bias
