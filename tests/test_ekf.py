"""EKF verification: Jacobians vs jax.jacfwd (float64, tight tolerance),
covariance symmetry/PSD over thousands of updates on a real excerpt, JAX vs
NumPy reference agreement, zero-initialised learned filter == classical,
finite-difference loss gradient, determinism.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from diffekf.config import Config
from diffekf.data.broad import load_trial
from diffekf.ekf import EKFState, initialise_state, run, transition_F
from diffekf.nets import LearnedEKF, bias_corrector_init, compute_corrections, noise_scaler_init
from diffekf.quaternion import (
    exp_so3,
    inclination_heading_error,
    log_so3,
    qconj,
    qmul,
    qnormalize,
    quat_to_rotmat,
    skew,
)
from diffekf.baselines.eskf_numpy import run as eskf_numpy_run

FIXTURE = "tests/fixtures/fixture_broad_trial01_excerpt.hdf5"
TOL = 1e-6  # tight tolerance; never loosened to pass


def _load_seq():
    return load_trial(FIXTURE, "01_undisturbed_slow_rotation_A")


def _init_state(seq, yaw_radians=0.0):
    t, gyro, accel, quat, movement = seq.as_jax()
    return initialise_state(accel, gyro, yaw_radians, Config())


def test_transition_jacobian_matches_autodiff():
    seq = _load_seq()
    t, gyro, accel, quat, movement = seq.as_jax()
    config = Config()
    state = _init_state(seq)
    k = 100
    omega = gyro[k]
    dt = float(jnp.diff(t)[k])
    u = omega - state.b

    def transition_map(pert):
        dtheta = pert[:3]
        db = pert[3:]
        q_pert = qnormalize(qmul(state.q, exp_so3(dtheta)))
        q_pred = qnormalize(qmul(state.q, exp_so3(u * dt)))
        q_pert_pred = qnormalize(qmul(q_pert, exp_so3((omega - (state.b + db)) * dt)))
        dtheta_new = log_so3(qmul(qconj(q_pred), q_pert_pred))
        return jnp.concatenate([dtheta_new, db])

    J = jax.jacfwd(transition_map)(jnp.zeros(6))
    F = transition_F(u, dt)
    assert J.shape == (6, 6)
    assert jnp.allclose(J, F, atol=TOL, rtol=TOL), float(jnp.abs(J - F).max())


def test_measurement_jacobian_matches_autodiff():
    seq = _load_seq()
    t, gyro, accel, quat, movement = seq.as_jax()
    config = Config()
    state = _init_state(seq)
    k = 100
    u = gyro[k] - state.b
    q_pred = qnormalize(qmul(state.q, exp_so3(u * dt)))

    def meas_map(pert):
        dtheta = pert[:3]
        q_p = qnormalize(qmul(q_pred, exp_so3(dtheta)))
        g_hat = quat_to_rotmat(q_p).T @ jnp.array([0.0, 0.0, 9.80665])
        return jnp.asarray(accel[k]) - g_hat

    J = jax.jacfwd(meas_map)(jnp.zeros(6))
    g_hat = quat_to_rotmat(q_pred).T @ jnp.array([0.0, 0.0, 9.80665])
    H_expected = jnp.concatenate([-skew(g_hat), jnp.zeros((3, 3))], axis=1)
    assert J.shape == (3, 6)
    assert jnp.allclose(J, H_expected, atol=TOL, rtol=TOL), float(jnp.abs(J - H_expected).max())


def test_covariance_symmetry_and_psd_on_real_excerpt():
    """Thousands of updates on a real excerpt: P stays symmetric PSD."""
    seq = _load_seq()
    t, gyro, accel, quat, movement = seq.as_jax()
    config = Config()
    min_eig = np.inf
    max_asym = 0.0
    for _ in range(20):
        state = _init_state(seq)
        _, out = run(gyro, accel, jnp.asarray(seq.dt(), dtype=jnp.float64), config, state)
        P_seq = np.asarray(out.covariance)
        for P in P_seq:
            max_asym = max(max_asym, float(np.abs(P - P.T).max()))
            min_eig = min(min_eig, float(np.linalg.eigvalsh(P).min()))
    assert max_asym < 1e-8, max_asym
    assert min_eig >= -1e-6, min_eig


def test_jax_matches_numpy_reference_on_excerpt():
    seq = _load_seq()
    t, gyro, accel, quat, movement = seq.as_jax()
    gyro_n = np.asarray(gyro)
    accel_n = np.asarray(accel)
    dt_n = np.asarray(seq.dt(), dtype=np.float64)
    R0 = np.asarray(quat_to_rotmat(jnp.asarray(quat[0])))
    yaw0 = float(np.arctan2(R0[1, 0], R0[0, 0]))
    config = Config()
    state0 = initialise_state(accel, gyro, yaw0, config)

    _, out_jax = run(gyro, accel, jnp.asarray(dt_n), config, state0)
    q_jax = np.asarray(out_jax.q)

    q_np, _, _, _, _ = eskf_numpy_run(gyro_n, accel_n, dt_n, yaw0)
    max_q_diff = float(np.max(np.abs(q_jax - q_np)))
    assert max_q_diff < 1e-4, max_q_diff


def test_zero_initialised_learned_equals_classical():
    seq = _load_seq()
    t, gyro, accel, quat, movement = seq.as_jax()
    config = Config()
    state = _init_state(seq)
    dt = jnp.asarray(seq.dt(), dtype=jnp.float64)
    key = jax.random.PRNGKey(0)
    bias_params = bias_corrector_init(key, config.window * 6, config.hidden)
    key, _ = jax.random.split(key)
    scale_params = noise_scaler_init(key, config.window * 2, config.hidden)
    delta_b, r_scale = compute_corrections(gyro, accel, dt, config, bias_params, scale_params)
    assert jnp.allclose(delta_b, 0.0, atol=1e-12)
    assert jnp.allclose(r_scale, 1.0, atol=1e-12)
    net = LearnedEKF(config, bias_params, scale_params)
    _, out_learned = net.run(gyro, accel, dt, state)
    _, out_classical = run(gyro, accel, dt, config, state)
    assert jnp.allclose(out_learned.q, out_classical.q, atol=1e-12)
    assert jnp.allclose(out_learned.b, out_classical.b, atol=1e-12)
    assert jnp.allclose(out_learned.p_diag, out_classical.p_diag, atol=1e-12)


def test_loss_gradient_finite_difference():
    seq = _load_seq()
    t, gyro, accel, quat, movement = seq.as_jax()
    config = Config()
    state = _init_state(seq)
    short = 40
    gyro_s = gyro[:short]
    accel_s = accel[:short]
    dt_s = jnp.asarray(seq.dt()[:short])
    quat_s = quat[:short]

    key = jax.random.PRNGKey(7)
    bias_params = bias_corrector_init(key, config.window * 6, config.hidden)
    key, _ = jax.random.split(key)
    scale_params = noise_scaler_init(key, config.window * 2, config.hidden)
    params = {"bias": bias_params, "scale": scale_params}

    def loss_val(p):
        net = LearnedEKF(config, p["bias"], p["scale"])
        _, out = net.run(gyro_s, accel_s, dt_s, state)
        incl, _ = inclination_heading_error(out.q, quat_s)
        return float(jnp.mean(incl))

    grads = jax.grad(loss_val)(params)
    eps = 1e-5
    fc2_w = params["bias"]["fc2"]["w"]
    w_flat = fc2_w.reshape(-1)
    g_flat = grads["bias"]["fc2"]["w"].reshape(-1)
    n_check = min(3, w_flat.size)
    base_shape = fc2_w.shape
    for i in range(n_check):
        orig = float(w_flat[i])
        vals = []
        for sign in (1.0, -1.0):
            w_mod = w_flat.at[i].set(orig + sign * eps)
            p_mod = {
                "bias": {
                    "fc1": params["bias"]["fc1"],
                    "fc2": {"w": w_mod.reshape(base_shape), "b": params["bias"]["fc2"]["b"]},
                },
                "scale": params["scale"],
            }
            vals.append(loss_val(p_mod))
        fd = (vals[0] - vals[1]) / (2 * eps)
        assert abs(fd - g_flat[i]) < 1e-3, (i, fd, g_flat[i])


def test_determinism_fixed_key():
    seq = _load_seq()
    t, gyro, accel, quat, movement = seq.as_jax()
    config = Config()

    def run_once(key):
        bias_params = bias_corrector_init(key, config.window * 6, config.hidden)
        key, _ = jax.random.split(key)
        scale_params = noise_scaler_init(key, config.window * 2, config.hidden)
        net = LearnedEKF(config, bias_params, scale_params)
        _, out = net.run(gyro[:80], accel[:80], jnp.asarray(seq.dt()[:80]), _init_state(seq))
        return float(jnp.mean(jnp.asarray(out.p_diag[:, 0])))

    assert run_once(jax.random.PRNGKey(42)) == run_once(jax.random.PRNGKey(42))
