"""Training the learned bias corrector / noise scaler.

Deterministic from a seed. optax Adam with global-norm clipping. Truncated
backpropagation through time: each training sequence is cut into windows of
``config.bptt_window`` samples; the filter state (q, b, P) is carried across
windows and detached between them. Loss = mean geodesic inclination error
against ground truth (heading excluded) plus a small NIS-consistency term.

Best (by validation inclination error) and last checkpoints are saved to
results/checkpoints/ and training is resumable from the last checkpoint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import Config
from .data.common import Sequence
from .ekf import EKFState, initialise_state, run
from .nets import LearnedEKF, bias_corrector_init, noise_scaler_init
from .quaternion import inclination_heading_error, quat_to_rotmat


def _as_jax(seq: Sequence):
    return seq.as_jax()


def _window_spans(T: int, window: int) -> List[Tuple[int, int]]:
    spans = []
    start = 0
    while start < T:
        end = min(T, start + window)
        spans.append((start, end))
        if end >= T:
            break
        start = end
    return spans


def _init_state_for(seq: Sequence, config: Config) -> EKFState:
    """Tilt from static accelerometer samples; heading from ground truth once."""
    _, gyro, accel, quat, _ = _as_jax(seq)
    R0 = quat_to_rotmat(quat[0])
    yaw0 = float(jnp.arctan2(R0[1, 0], R0[0, 0]))
    return initialise_state(accel, gyro, yaw0, config)


def _window_loss_and_state(
    params: Dict,
    omega: jax.Array,
    accel: jax.Array,
    dt: jax.Array,
    quat_gt: jax.Array,
    movement: jax.Array,
    init_state: EKFState,
    config: Config,
) -> Tuple[jax.Array, EKFState]:
    """Loss over one window plus the end filter state (carried, then detached)."""
    net = LearnedEKF(config, params["bias"], params["scale"])
    final_state, out = net.run(omega, accel, dt, init_state)
    inclination, _heading = inclination_heading_error(out.q, quat_gt)
    if jnp.any(movement):
        denom = jnp.maximum(jnp.sum(movement), 1)
        incl_err = jnp.sum(jnp.where(movement, inclination, jnp.zeros_like(inclination))) / denom
    else:
        incl_err = jnp.mean(inclination)
    innov = out.innovation
    S = out.innovation_cov
    nis = jnp.einsum("ti,tij,tj->t", innov, jnp.linalg.inv(S), innov) / 3.0
    nis_term = jnp.mean((nis - 1.0) ** 2)
    loss = incl_err + config.nis_lambda * nis_term
    return loss, final_state


def _sequence_loss(
    params: Dict, seq: Sequence, config: Config
) -> Tuple[jax.Array, Dict[str, float]]:
    """Truncated-BPTT loss over one full sequence.

    Windows are backpropagated through individually; the carry is detached
    between windows.
    """
    _, gyro, accel, quat_gt, movement = _as_jax(seq)
    dt = jnp.asarray(seq.dt(), dtype=jnp.float64)
    init_state = _init_state_for(seq, config)
    spans = _window_spans(len(gyro), config.bptt_window)
    carry = init_state
    total = jnp.array(0.0)
    nis_total = jnp.array(0.0)
    incl_total = jnp.array(0.0)
    count = 0
    for (start, end) in spans:
        loss, carry = _window_loss_and_state(
            params, gyro[start:end], accel[start:end], dt[start:end],
            quat_gt[start:end], movement[start:end], carry, config,
        )
        carry = jax.lax.stop_gradient(carry)
        total = total + loss
        count += 1
    return total / max(count, 1), {}


def _epoch_grad(params, sequences, config):
    total = jnp.array(0.0)
    for seq in sequences:
        loss, _ = _sequence_loss(params, seq, config)
        total = total + loss
    return total / len(sequences)


@jax.jit
def _apply_grad(params, opt_state, grads, optimizer):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state


def fit(
    config: Config,
    train_seqs: List[Sequence],
    val_seqs: List[Sequence],
    results_dir: str = "results",
    quick: bool = False,
    key: int | None = None,
) -> Dict:
    """Full training loop: early stopping, best/last checkpoints, resumable."""
    if key is None:
        key = config.seed
    rng = jax.random.PRNGKey(key)

    bias_params = bias_corrector_init(rng, config.window * 6, config.hidden)
    rng, _ = jax.random.split(rng)
    scale_params = noise_scaler_init(rng, config.window * 2, config.hidden)
    params = {"bias": bias_params, "scale": scale_params}

    optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                            optax.adam(config.lr))
    opt_state = optimizer.init(params)

    results_path = Path(results_dir)
    ckpt_dir = results_path / "checkpoints"
    log_dir = results_path / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    epochs = 3 if quick else config.epochs
    patience = 2 if quick else config.patience

    best_val = math.inf
    start_epoch = 0
    history: List[Dict] = []

    last_ckpt = ckpt_dir / "last.npz"
    if last_ckpt.exists():
        loaded = dict(np.load(last_ckpt, allow_pickle=False))
        params = {"bias": _subtree(loaded, "bias"), "scale": _subtree(loaded, "scale")}
        start_epoch = int(loaded["epoch"]) + 1
        best_val = float(loaded["best_val"])

    def val_inclination(p) -> float:
        total = 0.0
        for seq in val_seqs:
            _, gyro, accel, quat_gt, movement = _as_jax(seq)
            dt = jnp.asarray(seq.dt(), dtype=jnp.float64)
            init_state = _init_state_for(seq, config)
            net = LearnedEKF(config, p["bias"], p["scale"])
            _, out = net.run(gyro, accel, dt, init_state)
            inclination, _ = inclination_heading_error(out.q, quat_gt)
            if jnp.any(movement):
                denom = jnp.maximum(jnp.sum(movement), 1)
                total += float(jnp.sum(jnp.where(movement, inclination, 0.0)) / denom)
            else:
                total += float(jnp.mean(inclination))
        return total / max(len(val_seqs), 1)

    grad_fn = jax.jit(jax.grad(_epoch_grad))

    for epoch in range(start_epoch, epochs):
        loss_value, grads = jax.value_and_grad(_epoch_grad)(params, train_seqs, config)
        params, opt_state = _apply_grad(params, opt_state, grads, optimizer)
        train_loss = float(loss_value)
        vloss = val_inclination(params)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_inclination_err": vloss})
        print(f"epoch {epoch:3d} train_loss={train_loss:.4f} val_incl_err={vloss:.4f}")
        if vloss < best_val:
            best_val = vloss
            _save_ckpt(ckpt_dir / "best.npz", params, epoch, best_val)
        _save_ckpt(last_ckpt, params, epoch, best_val)
        # early stopping: no improvement for `patience` epochs
        recent = [h["val_inclination_err"] for h in history[-patience - 1:-1]]
        if recent and vloss >= max(recent):
            print(f"early stopping at epoch {epoch}")
            break

    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "history.json").write_text(json.dumps(history, indent=2))
    _save_ckpt(ckpt_dir / "final.npz", params, epochs - 1, best_val)
    return {"best_val_inclination_err": best_val, "history": history}


def _subtree(loaded: Dict, prefix: str) -> Dict:
    out = {}
    for k in loaded.keys():
        if k.startswith(prefix + "/"):
            out[k[len(prefix) + 1:]] = np.asarray(loaded[k])
    return out


def _save_ckpt(path, params, epoch, best_val):
    def flatten(p, pref=""):
        for k, v in p.items():
            if isinstance(v, dict):
                yield from flatten(v, pref + k + "/")
            else:
                yield pref + k, jax.device_get(v)

    fields = dict(flatten(params))
    fields["epoch"] = np.array(epoch)
    fields["best_val"] = np.array(best_val)
    np.savez(path, **fields)
