"""Quaternion unit tests: identities, exp/log round trip, frame conversion on
a real BROAD excerpt (tests/fixtures, CC BY 4.0, see ATTRIBUTION.md).

float64 throughout for correctness.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diffekf.quaternion import (
    exp_so3,
    geodesic_angle,
    inclination_heading_error,
    log_so3,
    qconj,
    qmul,
    qnormalize,
    quat_to_rotmat,
    rotate_vector,
    right_jacobian_so3,
    skew,
)

FIXTURE = "tests/fixtures/fixture_broad_trial01_excerpt.hdf5"


def _load_excerpt():
    import h5py

    with h5py.File(FIXTURE, "r") as f:
        accel = np.asarray(f["imu_acc"], dtype=np.float64)
        quat = np.asarray(f["opt_quat"], dtype=np.float64)
    return accel, quat


def test_qmul_is_hamilton_and_conj_inverts():
    key = jax.random.PRNGKey(0)
    q1 = qnormalize(jax.random.normal(key, (4,)))
    q2 = qnormalize(jax.random.normal(jax.random.split(key)[0], (4,)))
    ident = jnp.array([1.0, 0.0, 0.0, 0.0])
    assert jnp.allclose(qmul(q1, qconj(q1)), ident, atol=1e-12)
    assert jnp.allclose(qmul(qconj(q1), q1), ident, atol=1e-12)
    i = jnp.array([0.0, 1.0, 0.0, 0.0])
    j = jnp.array([0.0, 0.0, 1.0, 0.0])
    k = jnp.array([0.0, 0.0, 0.0, 1.0])
    assert jnp.allclose(qmul(i, j), k, atol=1e-12)
    assert jnp.allclose(qmul(j, i), -k, atol=1e-12)


def test_exp_log_round_trip():
    key = jax.random.PRNGKey(1)
    for scale in (1e-8, 1e-4, 1e-2, 0.5, 2.0):
        phi = jax.random.normal(key, (3,)) * scale
        q = exp_so3(phi)
        assert jnp.abs(jnp.linalg.norm(q) - 1.0) < 1e-10
        phi_back = log_so3(q)
        assert jnp.allclose(jnp.abs(phi_back), jnp.abs(phi), atol=1e-8) or \
            jnp.allclose(jnp.abs(phi_back), jnp.abs(-phi), atol=1e-8)
    assert jnp.allclose(exp_so3(jnp.zeros(3)), jnp.array([1.0, 0.0, 0.0, 0.0]), atol=1e-12)
    assert jnp.allclose(log_so3(jnp.array([1.0, 0.0, 0.0, 0.0])), jnp.zeros(3), atol=1e-12)


def test_rotate_vector_matches_rotmat():
    key = jax.random.PRNGKey(2)
    q = qnormalize(jax.random.normal(key, (4,)))
    v = jax.random.normal(jax.random.split(key)[0], (3,))
    assert jnp.allclose(rotate_vector(q, v), quat_to_rotmat(q) @ v, atol=1e-12)


def test_skew_identity():
    v = jnp.array([1.0, 2.0, 3.0])
    u = jnp.array([4.0, 5.0, 6.0])
    assert jnp.allclose(skew(v) @ u, jnp.cross(v, u), atol=1e-12)
    assert jnp.allclose(skew(v), -skew(v).T, atol=1e-12)


def test_right_jacobian_identity():
    """Defining identity: exp(phi) (x) exp(eps * Jr(:,i)) ~= exp(phi + eps e_i)."""
    key = jax.random.PRNGKey(3)
    for scale in (1e-7, 1e-3, 0.7):
        phi = jax.random.normal(key, (3,)) * scale
        Jr = right_jacobian_so3(phi)
        eps = 1e-5
        for i in range(3):
            lhs = qmul(exp_so3(phi), exp_so3(eps * Jr[:, i]))
            rhs = exp_so3(phi + eps * jnp.eye(3)[i])
            assert jnp.allclose(lhs, rhs, atol=1e-4), f"axis {i} scale {scale}"


def test_frame_conversion_on_real_excerpt():
    """BROAD ground truth equals our convention: static accel direction rotated
    into the world frame is [0, 0, 1]."""
    accel, quat = _load_excerpt()
    g_dir = np.asarray(accel[:200].mean(axis=0))
    g_dir = g_dir / np.linalg.norm(g_dir)
    world_up = np.asarray(rotate_vector(jnp.asarray(quat[0]), jnp.asarray(g_dir)))
    assert np.allclose(world_up, np.array([0.0, 0.0, 1.0]), atol=1e-2), world_up


def test_inclination_heading_match_broad_definitions():
    import h5py

    with h5py.File(FIXTURE, "r") as f:
        quat = np.asarray(f["opt_quat"], dtype=np.float64)
    q = jnp.asarray(quat)
    incl, heading = inclination_heading_error(q, q)
    assert jnp.allclose(incl, 0.0, atol=1e-9)
    assert jnp.allclose(heading, 0.0, atol=1e-9)
    half = 0.15
    q_yaw = jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)])
    q_pert = jax.vmap(lambda qi: qmul(q_yaw, qi))(q[:100])
    incl, heading = inclination_heading_error(q_pert, q[:100])
    assert jnp.allclose(incl, 0.0, atol=1e-6), incl
    assert jnp.allclose(heading, 0.3, atol=1e-6), heading


def test_geodesic_angle_properties():
    key = jax.random.PRNGKey(4)
    q1 = qnormalize(jax.random.normal(key, (4,)))
    q2 = qnormalize(jax.random.normal(jax.random.split(key)[0], (4,)))
    a = geodesic_angle(q1, q2)
    assert 0.0 <= a <= jnp.pi + 1e-6
    assert jnp.allclose(geodesic_angle(q1, q1), 0.0, atol=1e-12)
    assert jnp.allclose(geodesic_angle(q1, -q1), 0.0, atol=1e-12)
    assert jnp.allclose(geodesic_angle(q1, q2), geodesic_angle(q2, q1), atol=1e-12)


def test_vmap_and_jit_work():
    key = jax.random.PRNGKey(5)
    qs = qnormalize(jax.random.normal(key, (16, 4)))
    vs = jax.random.normal(jax.random.split(key)[0], (16, 3))
    f = jax.jit(jax.vmap(lambda q, v: rotate_vector(q, v)))
    out = f(qs, vs)
    assert out.shape == (16, 3)
    for i in range(16):
        assert jnp.allclose(out[i], rotate_vector(qs[i], vs[i]), atol=1e-10)
