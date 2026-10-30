"""Plain-NumPy multiplicative error-state EKF.

Independent correctness reference for the JAX filter in ekf.py: same equations
(attitude propagation on SO(3), discrete transition from the continuous
Jacobian, exact Van Loan process noise, Joseph-form accelerometer update,
accel-magnitude gate), written in NumPy only with no JAX imports. Used to
validate the JAX implementation at tight tolerance.

Conventions: Hamilton, scalar-first, body -> world, world z up.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import expm

GRAVITY = 9.80665


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _qnormalize(q):
    return q / np.linalg.norm(q)


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _exp_so3(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-6:
        return _qnormalize(np.array([1.0, phi[0] / 2, phi[1] / 2, phi[2] / 2]))
    axis = phi / theta
    return _qnormalize(np.array([np.cos(theta / 2), *(axis * np.sin(theta / 2))]))


def _rotmat(q):
    q = _qnormalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _right_jacobian(phi):
    theta = np.linalg.norm(phi)
    if theta < 1e-6:
        return np.eye(3) - 0.5 * _skew(phi) + (1.0 / 6.0) * (_skew(phi) @ _skew(phi))
    kx = _skew(phi)
    c1 = (1 - np.cos(theta)) / theta**2
    c2 = (theta - np.sin(theta)) / theta**3
    return np.eye(3) - c1 * kx + c2 * (kx @ kx)


def _tilt_heading_init(gravity_body, yaw):
    g = gravity_body / np.linalg.norm(gravity_body)
    target = np.array([0.0, 0.0, 1.0])
    v = np.cross(g, target)
    c = float(g @ target)
    s = np.linalg.norm(v)
    K = _skew(v)
    coef = (1.0 - c) / (s * s) if s > 1e-9 else 1.0
    R = np.eye(3) + K + coef * (K @ K)
    m = R
    t = np.trace(m)
    if t > 0:
        s2 = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s2, (m[2, 1] - m[1, 2]) / s2,
                      (m[0, 2] - m[2, 0]) / s2, (m[1, 0] - m[0, 1]) / s2])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s2 = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s2, 0.25 * s2,
                      (m[0, 1] + m[1, 0]) / s2, (m[0, 2] + m[2, 0]) / s2])
    elif m[1, 1] > m[2, 2]:
        s2 = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s2, (m[0, 1] + m[1, 0]) / s2,
                      0.25 * s2, (m[1, 2] + m[2, 1]) / s2])
    else:
        s2 = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s2, (m[0, 2] + m[2, 0]) / s2,
                      (m[1, 2] + m[2, 1]) / s2, 0.25 * s2])
    return _qnormalize(q)


def _log_so3(q):
    q = _qnormalize(q)
    if q[0] < 0:
        q = -q
    w, vec = q[0], q[1:]
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return 2 * vec
    return 2 * np.arctan2(n, np.clip(w, -1, 1)) * vec / n


def run(
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: np.ndarray,
    yaw_radians: float,
    gyro_noise_density: float = 1e-3,
    bias_rw_density: float = 1e-5,
    accel_noise_std: float = 2e-2,
    gate_gain: float = 10.0,
    gate_deadband: float = 0.2,
    init_window: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the NumPy error-state EKF.

    Returns
    -------
    quat : (T, 4)
    bias : (T, 3)
    p_diag : (T, 6)
    innovation : (T, 3)
    nis : (T,)
    """
    T = len(gyro)
    g_up = np.array([0.0, 0.0, GRAVITY])
    q0 = _tilt_heading_init(np.mean(accel[:init_window], axis=0), yaw_radians)
    q = q0.copy()
    b = np.zeros(3)
    P = np.diag(np.concatenate([np.full(3, 0.5**2), np.full(3, 1e-2**2)]))

    quat = np.zeros((T, 4))
    bias = np.zeros((T, 3))
    p_diag = np.zeros((T, 6))
    innovation = np.zeros((T, 3))
    nis = np.zeros(T)

    R0 = accel_noise_std**2 * np.eye(3)

    for k in range(T):
        # ----- predict -----
        u = gyro[k] - b
        q = _qnormalize(_qmul(q, _exp_so3(u * dt[k])))
        F = np.zeros((6, 6))
        F[:3, :3] = _rotmat(_exp_so3(-u * dt[k]))
        F[:3, 3:] = -_right_jacobian(u * dt[k]) * dt[k]
        F[3:, 3:] = np.eye(3)
        A = np.zeros((6, 6))
        A[:3, :3] = -_skew(u)
        A[:3, 3:] = -np.eye(3)
        GQcGt = np.diag(np.concatenate([
            np.full(3, gyro_noise_density**2),
            np.full(3, bias_rw_density**2),
        ]))
        M = np.zeros((12, 12))
        M[:6, :6] = -A * dt[k]
        M[:6, 6:] = GQcGt * dt[k]
        M[6:, 6:] = A.T * dt[k]
        X = expm(M)
        Q = X[6:, :6].T @ X[:6, 6:]
        Q = 0.5 * (Q + Q.T)
        P = F @ P @ F.T + Q
        P = 0.5 * (P + P.T)

        # ----- update -----
        g_hat = _rotmat(q).T @ g_up
        r = accel[k] - g_hat
        H = np.concatenate([-_skew(g_hat), np.zeros((3, 3))], axis=1)
        norm_a = float(np.linalg.norm(accel[k]))
        scale = 1.0 + gate_gain * max(abs(norm_a - GRAVITY) - gate_deadband, 0.0) ** 2 / GRAVITY**2
        R_eff = R0 * scale
        S = H @ P @ H.T + R_eff
        K = P @ H.T @ np.linalg.inv(S)
        dx = K @ r
        dtheta, db = dx[:3], dx[3:]
        q = _qnormalize(_qmul(q, _exp_so3(dtheta)))
        b = b + db
        A_cov = np.eye(6) - K @ H
        P = A_cov @ P @ A_cov.T + K @ R_eff @ K.T
        P = 0.5 * (P + P.T)

        quat[k] = q
        bias[k] = b
        p_diag[k] = np.diag(P)
        innovation[k] = r
        nis[k] = float(r @ np.linalg.inv(S) @ r)

    return quat, bias, p_diag, innovation, nis
