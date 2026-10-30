"""Quaternion mathematics for the attitude filter.

Conventions (stated once, obeyed everywhere in this repository):

* Hamilton quaternions, scalar-first: ``q = (w, x, y, z)``, ``i^2 = j^2 = k^2 = ijk = -1``.
* ``q`` maps body-frame vectors to world-frame vectors: ``v_world = R(q) @ v_body``.
* World frame is ENU: z is up. Gravity in the world frame is ``g_up = [0, 0, +g]``.
* The accelerometer measures specific force, so a static sensor reads
  ``a_meas = R(q).T @ [0, 0, +g]``.
* Error-state convention: ``q_true = q @ exp(delta_theta)`` (right-multiplicative,
  body-frame error).

All functions are pure (no side effects), accept JAX arrays, and are written to be
`jax.jit`-able and `jax.vmap`-able. Leading batch dimensions are supported wherever
shapes show ``(...)``.

Docstrings state shapes (``N`` = batch/time), units and frames.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

# Earth gravity magnitude, m/s^2 (standard gravity).
GRAVITY: float = 9.80665

# Threshold below which the Taylor branches of exp_so3 / log_so3 are used.
_SMALL: float = 1e-6


def qmul(a: jax.Array, b: jax.Array) -> jax.Array:
    """Hamilton quaternion product.

    Parameters
    ----------
    a, b : (..., 4)
        Quaternions (w, x, y, z), not required to be unit. Broadcasting rules apply.

    Returns
    -------
    (..., 4)
        Hamilton product ``a (x) b``. Active rotations compose as
        ``R(q1 (x) q2) = R(q1) @ R(q2)``.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return jnp.stack([w, x, y, z], axis=-1)


def qconj(q: jax.Array) -> jax.Array:
    """Quaternion conjugate (inverse for unit quaternions).

    Parameters
    ----------
    q : (..., 4)

    Returns
    -------
    (..., 4)
        ``(w, -x, -y, -z)``.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    return jnp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def qnormalize(q: jax.Array, eps: float = 1e-12) -> jax.Array:
    """Normalise quaternion(s) to unit norm.

    Parameters
    ----------
    q : (..., 4)
    eps : float
        Floor for the norm to avoid division by zero.

    Returns
    -------
    (..., 4)
        ``q / ||q||``.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    return q / jnp.maximum(norm, eps)


def qinverse(q: jax.Array) -> jax.Array:
    """Quaternion inverse: ``conj(q) / ||q||^2``.

    Parameters
    ----------
    q : (..., 4)

    Returns
    -------
    (..., 4)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    return qconj(q) / jnp.sum(q * q, axis=-1, keepdims=True)


def skew(v: jax.Array) -> jax.Array:
    """Skew-symmetric cross-product matrix of a 3-vector.

    Convention: ``skew(v) @ u = v x u`` (``x`` is the cross product).

    Parameters
    ----------
    v : (..., 3)

    Returns
    -------
    (..., 3, 3)
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = jnp.zeros_like(x)
    row1 = jnp.stack([zero, -z, y], axis=-1)
    row2 = jnp.stack([z, zero, -x], axis=-1)
    row3 = jnp.stack([-y, x, zero], axis=-1)
    return jnp.stack([row1, row2, row3], axis=-2)


def exp_so3(phi: jax.Array) -> jax.Array:
    """Exponential map SO(3): rotation vector -> unit quaternion.

    Maps a rotation vector ``phi`` (axis-angle, magnitude = angle in radians) to
    the unit quaternion ``(cos(|phi|/2), (phi/|phi|) sin(|phi|/2))``. Uses a
    fourth-order Taylor branch near zero for numerical stability.

    Parameters
    ----------
    phi : (..., 3)
        Rotation vector, radians.

    Returns
    -------
    (..., 4)
        Unit quaternion (w, x, y, z).
    """
    phi = jnp.asarray(phi, dtype=jnp.float64)
    theta2 = jnp.sum(phi * phi, axis=-1, keepdims=True)
    theta = jnp.sqrt(theta2)
    half = 0.5 * theta
    # cos(theta/2) and sin(theta/2)/theta with stable small-angle expansion.
    cos_half = jnp.cos(half)
    # sin(theta/2) / theta = 1/2 * sinc(theta/pi-ish); expand directly:
    # sin(theta/2)/theta = 1/2 - theta^2/48 + theta^4/3840 + O(theta^6)
    theta2_s = theta2[..., 0]
    sinc_half = (
        0.5
        - theta2_s / 48.0
        + theta2_s * theta2_s / 3840.0
    )
    # Where theta is not tiny, use the exact expression.
    exact = jnp.where(
        theta2_s > _SMALL**2,
        jnp.sin(half) / jnp.maximum(theta, 1e-300),
        sinc_half,
    )
    vec = phi * exact[..., None]
    w = jnp.where(theta2_s > _SMALL**2, cos_half, jnp.ones_like(half)[..., 0])
    return qnormalize(jnp.stack([w, vec[..., 0], vec[..., 1], vec[..., 2]], axis=-1))


def log_so3(q: jax.Array) -> jax.Array:
    """Logarithm map SO(3): unit quaternion -> rotation vector.

    Returns the rotation vector ``phi`` (axis-angle, radians) of the rotation
    represented by ``q``, taking the shortest path (sign of ``q`` normalised so
    ``w >= 0``). Uses a Taylor branch near the identity.

    Parameters
    ----------
    q : (..., 4)
        Unit quaternions (w, x, y, z).

    Returns
    -------
    (..., 3)
        Rotation vector, radians.
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    q = qnormalize(q)
    # Shortest path: flip sign so w >= 0.
    q = jnp.where(q[..., 0:1] < 0, -q, q)
    w = q[..., 0]
    vec = q[..., 1:]
    vec_norm = jnp.linalg.norm(vec, axis=-1, keepdims=True)
    # angle = 2 * atan2(|v|, w); phi = angle * v/|v| = 2 * atan2(|v|, w) * v/|v|
    # Near identity: atan2(|v|, w) ~ |v|, so phi ~ 2 v.
    angle_times_dir = jnp.where(
        vec_norm[..., 0] > _SMALL,
        2.0 * jnp.arctan2(vec_norm, jnp.clip(w, -1.0, 1.0)) / jnp.maximum(vec_norm, 1e-300),
        2.0,
    )
    phi = vec * angle_times_dir
    return phi


def quat_to_rotmat(q: jax.Array) -> jax.Array:
    """Rotation matrix of a unit quaternion (body -> world).

    Parameters
    ----------
    q : (..., 4)
        Unit quaternion (w, x, y, z).

    Returns
    -------
    (..., 3, 3)
        ``R`` such that ``v_world = R @ v_body``.
    """
    q = qnormalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1.0 - 2.0 * (y2 + z2)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (x2 + z2)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (x2 + y2)
    row1 = jnp.stack([m00, m01, m02], axis=-1)
    row2 = jnp.stack([m10, m11, m12], axis=-1)
    row3 = jnp.stack([m20, m21, m22], axis=-1)
    return jnp.stack([row1, row2, row3], axis=-2)


def rotate_vector(q: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate vector(s) by quaternion(s): ``R(q) @ v`` (body -> world).

    Parameters
    ----------
    q : (..., 4)
    v : (..., 3)

    Returns
    -------
    (..., 3)
    """
    return jnp.einsum("...ij,...j->...i", quat_to_rotmat(q), v)


def right_jacobian_so3(phi: jax.Array) -> jax.Array:
    """Right Jacobian of SO(3) at rotation vector ``phi``.

    ``Jr(phi) = I - (1 - cos theta)/theta^2 [phi]_x + (theta - sin theta)/theta^3 [phi]_x^2``,
    with a Taylor branch near zero: ``Jr = I - 0.5 [phi]_x + 1/6 [phi]_x^2 + O(phi^3)``.

    Parameters
    ----------
    phi : (..., 3)
        Rotation vector, radians.

    Returns
    -------
    (..., 3, 3)
    """
    phi = jnp.asarray(phi, dtype=jnp.float64)
    theta2 = jnp.sum(phi * phi, axis=-1, keepdims=True)
    theta = jnp.sqrt(theta2)
    t2 = theta2[..., 0]
    # Coefficients with stable small-angle expansion.
    # (1 - cos theta)/theta^2 = 1/2 - theta^2/24 + theta^4/720 + ...
    c1 = jnp.where(
        t2 > _SMALL**2,
        (1.0 - jnp.cos(theta)) / jnp.maximum(theta2, 1e-300),
        0.5 - t2 / 24.0 + t2 * t2 / 720.0,
    )
    # (theta - sin theta)/theta^3 = 1/6 - theta^2/120 + theta^4/5040 + ...
    c2 = jnp.where(
        t2 > _SMALL**2,
        (theta - jnp.sin(theta)) / jnp.maximum(theta ** 3, 1e-300),
        1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0,
    )
    kx = skew(phi)
    kx2 = jnp.einsum("...ij,...jk->...ik", kx, kx)
    eye = jnp.eye(3, dtype=jnp.float64)
    Jr = eye[None] if kx.ndim > 2 else eye
    Jr = eye - c1[..., None, None] * kx + c2[..., None, None] * kx2
    return Jr


def geodesic_angle(q1: jax.Array, q2: jax.Array) -> jax.Array:
    """Geodesic (great-circle) angle between two unit quaternions.

    Parameters
    ----------
    q1, q2 : (..., 4)

    Returns
    -------
    (...)
        Angle in radians, in [0, pi]. Unsigned: ``q`` and ``-q`` are identical.
    """
    q1 = qnormalize(q1)
    q2 = qnormalize(q2)
    dot = jnp.sum(q1 * q2, axis=-1)
    dot = jnp.clip(jnp.abs(dot), 0.0, 1.0)
    return 2.0 * jnp.arccos(dot)


def inclination_heading_error(q_est: jax.Array, q_gt: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Inclination and heading decomposition of the attitude error, matching BROAD.

    The earth-frame error quaternion is ``q_err = q_est (x) conj(q_gt)`` (error
    expressed in the world frame, as in BROAD's ``calculateErrorQuatEarth``). With
    world z up:

    * inclination error = tilt away from the vertical =
      ``2 * acos(clip(sqrt(w^2 + z^2)))``  (roll/pitch magnitude)
    * heading error = yaw about the world z axis =
      ``2 * atan(|z / w|)``

    Parameters
    ----------
    q_est, q_gt : (..., 4)

    Returns
    -------
    inclination : (...) radians
    heading : (...) radians
    """
    q_est = qnormalize(q_est)
    q_gt = qnormalize(q_gt)
    q_err = qmul(q_est, qconj(q_gt))
    # Shortest path for a well-defined decomposition.
    q_err = jnp.where(q_err[..., 0:1] < 0, -q_err, q_err)
    w = q_err[..., 0]
    z = q_err[..., 3]
    inclination = 2.0 * jnp.arccos(jnp.clip(jnp.sqrt(w * w + z * z), 0.0, 1.0))
    heading = 2.0 * jnp.arctan(jnp.abs(z / jnp.maximum(jnp.abs(w), 1e-12)))
    return inclination, heading


def _tilt_from_gravity(gravity_body: jax.Array) -> jax.Array:
    """Quaternion rotating the measured gravity direction to world z (heading free).

    Returns the minimal rotation sending ``gravity_body`` to ``[0, 0, 1]``; the
    roll/pitch (tilt) is fixed, yaw about world z is zero.
    """
    g = gravity_body / jnp.maximum(jnp.linalg.norm(gravity_body), 1e-12)
    target = jnp.array([0.0, 0.0, 1.0])
    v = jnp.cross(g, target)
    c = jnp.dot(g, target)
    # handle the degenerate case g == -target
    s = jnp.linalg.norm(v)
    K = skew(v)
    # Rodrigues: R = I + [v]_x + [v]_x^2 * (1-c)/s^2
    coef = jnp.where(s > 1e-9, (1.0 - c) / (s * s + 1e-30), 1.0)
    R = jnp.eye(3) + K + coef * (K @ K)
    # Rotation matrix -> quaternion, scalar-first (Shepperd's method, robust).
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    t = m00 + m11 + m22
    w = 0.5 * jnp.sqrt(jnp.maximum(1.0 + t, 0.0))
    x = 0.5 * jnp.sqrt(jnp.maximum(1.0 + m00 - m11 - m22, 0.0))
    y = 0.5 * jnp.sqrt(jnp.maximum(1.0 - m00 + m11 - m22, 0.0))
    z = 0.5 * jnp.sqrt(jnp.maximum(1.0 - m00 - m11 + m22, 0.0))
    x = jnp.where(m21 - m12 < 0, -x, x)
    y = jnp.where(m02 - m20 < 0, -y, y)
    z = jnp.where(m10 - m01 < 0, -z, z)
    q = qnormalize(jnp.stack([w, x, y, z]))
    return q


def tilt_heading_init(
    gravity_body: jax.Array, yaw_world_radians: float
) -> jax.Array:
    """Initial attitude: tilt from gravity direction, heading from a yaw angle.

    ``q_init = q_yaw(yaw) (x) q_tilt`` so the world-frame yaw comes from ground
    truth once at initialisation and is never touched by the filter afterwards.

    Parameters
    ----------
    gravity_body : (3,)
        Mean accelerometer reading of static samples (m/s^2); its direction is
        taken as the body-frame gravity direction.
    yaw_world_radians : float
        Heading about the world z axis (radians), e.g. from the first
        ground-truth sample.

    Returns
    -------
    (4,)
        Unit quaternion (w, x, y, z).
    """
    q_tilt = _tilt_from_gravity(gravity_body)
    half = 0.5 * yaw_world_radians
    q_yaw = jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)], dtype=jnp.float64)
    q_yaw = qnormalize(q_yaw)
    return qnormalize(qmul(q_yaw, q_tilt))
