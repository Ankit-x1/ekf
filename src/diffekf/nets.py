"""Small learned components for the differentiable filter.

Two modules, both deliberately small (well under 100,000 parameters) so CPU
training is practical:

1. ``BiasCorrector``: a causal-window corrector over the last ``W`` samples of
   gyro and accelerometer (scaled by dt and by g), outputting a 3-vector added
   to the gyro-bias estimate.
2. ``NoiseScaler``: a positive multiplier on the accelerometer measurement
   covariance, driven by the accelerometer-magnitude deviation and the
   short-window accelerometer variance, through ``1 + softplus(x) - ln(2)``.

Both output layers are zero-initialised, so the untrained learned filter is
*exactly* the classical error-state EKF (enforced by a test). All functions are
pure and vmap / jit able; parameters are plain pytrees of JAX arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from .config import Config
from .ekf import EKFState, run, step

_LOG2: float = 0.6931471805599453


# --------------------------------------------------------------------------- #
# Small MLP building blocks
# --------------------------------------------------------------------------- #
def _linear(key, in_dim, out_dim, zero_init=False):
    lim = 0.0 if zero_init else 1.0 / jnp.sqrt(in_dim)
    w = jax.random.uniform(key, (in_dim, out_dim), minval=-lim, maxval=lim)
    b = jnp.zeros(out_dim)
    return {"w": w, "b": b}


def _linear_apply(params, x):
    return x @ params["w"] + params["b"]


def _mlp(params, x):
    h = jnp.tanh(_linear_apply(params["fc1"], x))
    return _linear_apply(params["fc2"], h)


def bias_corrector_init(key, input_dim, hidden):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "fc1": _linear(k1, input_dim, hidden),
        "fc2": _linear(k2, hidden, 3, zero_init=True),
    }


def noise_scaler_init(key, input_dim, hidden):
    k1, k2 = jax.random.split(key, 2)
    return {
        "fc1": _linear(k1, input_dim, hidden),
        "fc2": _linear(k2, hidden, 1, zero_init=True),
    }


# --------------------------------------------------------------------------- #
# Causal-window feature extraction
# --------------------------------------------------------------------------- #
def _roll_insert(buf, x):
    return jnp.roll(buf, -1, axis=0).at[-1].set(x)


def bias_corrector_features(omega, accel, dt, window, g=9.80665):
    """Causal window features for the bias corrector.

    At sample k the buffer holds samples ``k-window+1 .. k`` (padded with zeros
    at the start). Each sample is ``[gyro * dt, accel / g]`` (dimensionless).

    Returns
    -------
    (T, window * 6)
    """
    T = omega.shape[0]
    x = jnp.stack([omega * dt[:, None], accel / g], axis=1)  # (T, 2, 3)
    x = jnp.transpose(x, (0, 2, 1)).reshape(T, 6)            # (T, 6)

    def scan_fn(buf, x_k):
        buf_new = _roll_insert(buf, x_k)
        return buf_new, buf_new

    buf0 = jnp.zeros((window, 6))
    _, buffers = jax.lax.scan(scan_fn, buf0, x)              # (T, window, 6)
    return buffers.reshape(T, window * 6)


def noise_scaler_features(accel, window, g=9.80665):
    """Features for the noise scaler: magnitude deviation and short-window variance.

    Returns
    -------
    (T, 2)  [| ||a|| - g | / g, var(||a|| over window) / g^2]
    """
    norm_a = jnp.linalg.norm(accel, axis=1)                  # (T,)

    def scan_fn(buf, n_k):
        buf_new = _roll_insert(buf, n_k)
        return buf_new, buf_new

    buf0 = jnp.zeros(window)
    _, buffers = jax.lax.scan(scan_fn, buf0, norm_a)         # (T, window)
    mean = jnp.mean(buffers, axis=1)
    var = jnp.mean(buffers * buffers, axis=1) - mean * mean
    dev = jnp.abs(norm_a - g) / g
    return jnp.stack([dev, jnp.maximum(var, 0.0) / (g * g)], axis=1)


# --------------------------------------------------------------------------- #
# Per-step corrections produced from the raw input sequence (no filter state)
# --------------------------------------------------------------------------- #
def compute_corrections(omega, accel, dt, config, bias_params, scale_params):
    """Precompute per-sample learned corrections from the input sequence.

    Parameters
    ----------
    omega : (T, 3) gyro, rad / s.
    accel : (T, 3) accelerometer, m / s^2.
    dt : (T,) seconds.
    config : Config
    bias_params, scale_params : network parameters.

    Returns
    -------
    delta_b_seq : (T, 3)   added to the bias estimate.
    r_scale_seq : (T,)     positive multiplier on the accel covariance.
    """
    feat_b = bias_corrector_features(omega, accel, dt, config.window)
    delta_b = jax.vmap(lambda f: _mlp(bias_params, f))(feat_b)   # (T, 3)

    feat_s = noise_scaler_features(accel, config.window)
    raw = jax.vmap(lambda f: _mlp(scale_params, f).squeeze(-1))(feat_s)  # (T,)
    r_scale = 1.0 + jax.nn.softplus(raw) - _LOG2                # >= 1 - ln2 > 0
    return delta_b, r_scale


@dataclass
class LearnedEKF:
    """Classical EKF wrapped with the two learned corrections."""

    config: Config
    bias_params: dict
    scale_params: dict

    def run(self, omega, accel, dt, init_state):
        delta_b_seq, r_scale_seq = compute_corrections(
            omega, accel, dt, self.config, self.bias_params, self.scale_params
        )
        return run(omega, accel, dt, self.config, init_state,
                   delta_b_seq=delta_b_seq, r_scale_seq=r_scale_seq)

    def step(self, state, omega_k, a_k, dt_k):
        """Single step with corrections from the current sample's causal window.

        Used only for latency benchmarks: benchmarks the EKF core plus one
        forward pass of both networks (window state degenerates to the current
        sample).
        """
        feat_b = bias_corrector_features(
            omega_k[None, :], a_k[None, :], jnp.array([dt_k]), self.config.window
        )
        delta_b = _mlp(self.bias_params, feat_b[0])
        feat_s = noise_scaler_features(a_k[None, :], self.config.window)
        raw = _mlp(self.scale_params, feat_s[0]).squeeze(-1)
        r_scale = 1.0 + jax.nn.softplus(raw) - _LOG2
        return step(state, omega_k, a_k, dt_k, self.config,
                    delta_b=delta_b, r_scale=r_scale)
