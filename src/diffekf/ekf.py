"""Multiplicative error-state EKF for 6-axis (gyro + accelerometer) attitude estimation.

Nominal state
-------------
``q`` : unit quaternion (w, x, y, z), Hamilton, scalar-first, mapping body -> world.
``b`` : gyro bias, rad / s, body frame.

Error state (6 dims, body frame)
--------------------------------
``x = [delta_theta; delta_b]`` with ``q_true = q (x) exp(delta_theta)``.

Propagation (exact on SO(3))
----------------------------
``q_{k+1} = q_k (x) exp((omega_k - b_k) dt)``
``b_{k+1} = b_k``

Discrete error-state transition (derived below and checked against
``jax.jacfwd`` in the tests):

``F_k = [ [R(exp(-u dt)),  -Jr(u dt) dt] ]``  top row, 3 x 6
        ``[ [0_3x3,         I_3          ] ]``  bottom row

with ``u = omega - b``. This is exactly ``exp(A dt)`` for the continuous
error-state dynamics ``delta_theta_dot = -[u]_x delta_theta - delta_b``,
``delta_b_dot = 0``:

``A = [ [-[u]_x, -I] ]``
    ``[ [0,      0] ]``

because ``exp(-[u]_x dt) = R(exp_so3(-u dt))`` and
``integral_0^dt exp(-[u]_x s) ds = Jr(u dt) dt``.

Process noise
-------------
Continuous model ``x_dot = A x + G w`` with
``G = [I 0; 0 I]`` and spectral density ``Q_c = diag(sigma_g^2 I3, sigma_b^2 I3)``
from the gyro noise density and the bias random-walk density. Exact discrete
covariance by Van Loan's method on the 12 x 12 matrix

``M = [ [-A dt,  G Q_c G^T dt] ]``
    ``[ [0,      A^T dt      ] ]``

``X = expm(M) = [ [X11, X12] ]``
                ``[ [0,    X21] ]``

``Phi = exp(A dt) = X21.T`` and ``Q_d = X21.T @ X12``.

Update
------
Predicted gravity in the body frame: ``g_hat = R(q_pred).T @ [0, 0, g]``.
Innovation: ``r = a_meas - g_hat``.
Linearisation (checked against ``jax.jacfwd``):

``r(delta_theta) = a_meas - R(q exp(delta_theta)).T @ g_up
                 approx r - skew(g_hat) delta_theta``

so ``H = [-skew(g_hat), 0_3x3]`` (3 x 6). Equivalently ``+skew(g_hat)`` under the
transposed skew convention ``[v]_x u = u x v``; the sign used here matches the
residual exactly as implemented.

Joseph-form covariance update with explicit symmetrisation:

``P = (I - K H) P_pred (I - K H).T + K R_eff K.T``
``P = 0.5 (P + P.T)``

Error injection with quaternion renormalisation:

``q <- normalize(q_pred (x) exp(delta_theta))``
``b <- b_pred + delta_b``

The reset map sends the post-update error state to zero; its Jacobian with
respect to the pre-injection error state is the identity to first order, so the
posterior covariance is carried over directly after renormalisation (no
double counting, no information loss).

Accelerometer-magnitude gate
----------------------------
When ``||a_meas|`` deviates from ``g`` (linear acceleration corrupts the
gravity-direction measurement), the measurement covariance is inflated:

``scale = 1 + gate_gain * max(0, | ||a_meas|| - g | - deadband)^2 / g^2``
``R_eff = R * scale``

Heading is unobservable from 6 axes: tilt is initialised from the first static
accelerometer samples and heading from the first ground-truth sample once;
ground truth is never read by the filter afterwards.

The public entry point :func:`run` is a single ``jax.lax.scan`` over a
time-major sequence, returning states, covariance diagonals, innovations and
innovation covariances; it is ``jax.vmap``-able over padded batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .quaternion import (
    GRAVITY,
    exp_so3,
    geodesic_angle,
    qmul,
    qnormalize,
    quat_to_rotmat,
    right_jacobian_so3,
    skew,
)
from .config import Config


@dataclass(frozen=True)
class EKFState:
    """Filter carry inside the scan."""

    q: jax.Array          # (4,)
    b: jax.Array          # (3,)
    P: jax.Array          # (6, 6)


@dataclass(frozen=True)
class EKFStepOutput:
    """Per-sample output of the scan.

    ``covariance`` is the full 6 x 6 error-state covariance (needed for a
    proper Mahalanobis NEES); ``p_diag`` is its diagonal for compact logging.
    """

    q: jax.Array          # (4,)
    b: jax.Array          # (3,)
    p_diag: jax.Array     # (6,)
    covariance: jax.Array # (6, 6)
    innovation: jax.Array # (3,)
    innovation_cov: jax.Array  # (3, 3)


def _rotation_from_exp_so3_minus(u_dt: jax.Array) -> jax.Array:
    return quat_to_rotmat(exp_so3(-u_dt))


def predict(
    q: jax.Array, b: jax.Array, omega: jax.Array, dt: float
) -> Tuple[jax.Array, jax.Array]:
    """Nominal propagation: ``q <- q (x) exp((omega - b) dt)``, ``b <- b``.

    Parameters
    ----------
    q, b : nominal state.
    omega : (3,) gyro measurement, rad / s, body frame.
    dt : float, seconds.

    Returns
    -------
    q_pred, b_pred
    """
    u = omega - b
    q_pred = qnormalize(qmul(q, exp_so3(u * dt)))
    return q_pred, b


def transition_F(u: jax.Array, dt: float) -> jax.Array:
    """Discrete error-state transition ``F_k`` (6 x 6)."""
    F11 = _rotation_from_exp_so3_minus(u * dt)
    F12 = -right_jacobian_so3(u * dt) * dt
    top = jnp.concatenate([F11, F12], axis=1)
    bottom = jnp.concatenate([jnp.zeros((3, 3)), jnp.eye(3)], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


def process_noise_Q(
    u: jax.Array,
    dt: float,
    gyro_noise_density: float,
    bias_rw_density: float,
) -> jax.Array:
    """Exact discrete process covariance by Van Loan's method (6 x 6)."""
    A = jnp.zeros((6, 6))
    A = A.at[:3, :3].set(-skew(u))
    A = A.at[:3, 3:].set(-jnp.eye(3))
    GQcGt = jnp.diag(
        jnp.concatenate(
            [
                jnp.full(3, gyro_noise_density**2),
                jnp.full(3, bias_rw_density**2),
            ]
        )
    )
    top = jnp.concatenate([-A * dt, GQcGt * dt], axis=1)
    bottom = jnp.concatenate([jnp.zeros((6, 6)), A.T * dt], axis=1)
    M = jnp.concatenate([top, bottom], axis=0)
    X = lax.linalg.expm(M)
    X12 = X[:6, 6:]
    X21 = X[6:, :6]
    Q_d = X21.T @ X12
    Q_d = 0.5 * (Q_d + Q_d.T)
    return Q_d


def innovation_and_H(
    q_pred: jax.Array, a_meas: jax.Array, g: float = GRAVITY
) -> Tuple[jax.Array, jax.Array]:
    """Accelerometer innovation and measurement Jacobian.

    Returns ``r = a_meas - R(q_pred).T @ [0, 0, g]`` and
    ``H = [-skew(R(q_pred).T @ [0,0,g]), 0]``.
    """
    g_up = jnp.array([0.0, 0.0, g])
    g_hat = quat_to_rotmat(q_pred).T @ g_up
    r = a_meas - g_hat
    H = jnp.concatenate([-skew(g_hat), jnp.zeros((3, 3))], axis=1)
    return r, H, g_hat


def gate_scale(
    a_meas: jax.Array,
    g: float,
    gate_gain: float,
    deadband: float,
) -> jax.Array:
    """Inflation factor for the accelerometer measurement covariance."""
    norm_a = jnp.linalg.norm(a_meas)
    deviation = jnp.abs(norm_a - g)
    excess = jnp.maximum(deviation - deadband, 0.0) / g
    return 1.0 + gate_gain * excess * excess


# The real update needs omega as well; the single step function takes both inputs.
def step(
    state: EKFState,
    omega: jax.Array,
    a_meas: jax.Array,
    dt: float,
    config: Config,
    delta_b: jax.Array | None = None,
    r_scale: jax.Array | None = None,
) -> Tuple[EKFState, EKFStepOutput]:
    """One predict-update cycle of the error-state EKF.

    See module docstring for the equations.
    """
    q, b, P = state.q, state.b, state.P

    # Learned bias correction is added to the bias estimate before propagation.
    b_eff = b if delta_b is None else b + delta_b
    u = omega - b_eff
    q_pred, b_pred = predict(q, b_eff, omega, dt)

    F = transition_F(u, dt)
    Q = process_noise_Q(u, dt, config.gyro_noise_density, config.bias_rw_density)
    P_pred = F @ P @ F.T + Q
    P_pred = 0.5 * (P_pred + P_pred.T)

    r, H, g_hat = innovation_and_H(q_pred, a_meas)
    R = (config.accel_noise_std**2) * jnp.eye(3)
    scale = gate_scale(a_meas, GRAVITY, config.gate_gain, config.gate_deadband)
    if r_scale is not None:
        scale = scale * r_scale
    R_eff = R * scale

    S = H @ P_pred @ H.T + R_eff
    S = 0.5 * (S + S.T)
    K = jnp.linalg.solve(S.T, (P_pred @ H.T).T).T  # K = P_pred H^T S^{-1}
    dx = K @ r
    d_theta, d_b = dx[:3], dx[3:]

    q_new = qnormalize(qmul(q_pred, exp_so3(d_theta)))
    b_new = b_pred + d_b

    I = jnp.eye(6)
    A_cov = I - K @ H
    P_new = A_cov @ P_pred @ A_cov.T + K @ R_eff @ K.T
    P_new = 0.5 * (P_new + P_new.T)

    state_new = EKFState(q=q_new, b=b_new, P=P_new)
    output = EKFStepOutput(
        q=q_new,
        b=b_new,
        p_diag=jnp.diag(P_new),
        covariance=P_new,
        innovation=r,
        innovation_cov=S,
    )
    return state_new, output


def run(
    omega: jax.Array,
    accel: jax.Array,
    dt: jax.Array,
    config: Config,
    init_state: EKFState,
    delta_b_seq: jax.Array | None = None,
    r_scale_seq: jax.Array | None = None,
) -> Tuple[EKFState, EKFStepOutput]:
    """Run the filter over a time-major sequence with a single ``jax.lax.scan``.

    No Python loop over time.

    Parameters
    ----------
    omega : (T, 3) gyro, rad / s.
    accel : (T, 3) accelerometer, m / s^2.
    dt : (T,) sample intervals, seconds.
    config : Config
    init_state : EKFState
    delta_b_seq : (T, 3) optional learned bias correction per sample.
    r_scale_seq : (T,) optional learned noise multiplier per sample.

    Returns
    -------
    final_state : EKFState
    outputs : EKFStepOutput with stacked (T, ...) arrays.
    """
    if delta_b_seq is None:
        delta_b_seq = jnp.zeros((omega.shape[0], 3))
    if r_scale_seq is None:
        r_scale_seq = jnp.ones((omega.shape[0],))

    def scan_fn(carry: EKFState, inputs):
        omega_k, a_k, dt_k, db_k, rs_k = inputs
        state_new, out = step(carry, omega_k, a_k, dt_k, config, db_k, rs_k)
        return state_new, out

    inputs = (omega, accel, dt, delta_b_seq, r_scale_seq)
    final_state, outputs = lax.scan(scan_fn, init_state, inputs)
    return final_state, outputs


def initialise_state(
    accel: jax.Array,
    omega: jax.Array,
    yaw_radians: float,
    config: Config,
) -> EKFState:
    """Initialise tilt from static accelerometer samples, heading from yaw.

    Tilt: rotation sending the mean body-frame gravity direction to world z,
    computed from the first ``config.init_window`` samples (chosen static by the
    caller). Heading: yaw about world z from the first ground-truth sample.
    Covariance starts with large attitude uncertainty and small bias uncertainty.
    """
    from .quaternion import tilt_heading_init

    static_acc = accel[: config.init_window]
    gravity_body = jnp.mean(static_acc, axis=0)
    q0 = tilt_heading_init(gravity_body, yaw_radians)
    b0 = jnp.zeros(3)
    P0 = jnp.diag(jnp.concatenate([
        jnp.full(3, (0.5) **2),     # attitude variance, rad^2
        jnp.full(3, (1e-2)**2),     # bias variance, (rad/s)^2
    ]))
    return EKFState(q=q0, b=b0, P=P0)


def attitude_error_vector(q_est: jax.Array, q_gt: jax.Array) -> jax.Array:
    """3-vector attitude error ``log(conj(q_est) (x) q_gt)`` (world frame)."""
    from .quaternion import qconj, log_so3, qmul

    return log_so3(qmul(qconj(q_est), q_gt))
