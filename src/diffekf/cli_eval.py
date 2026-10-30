"""Remaining CLI subcommands: eval, ablate, plot, bench, reproduce, fill-docs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer
from rich.console import Console

from .cli import app, console, _ensure_data, _init_for_eval, _select_device, _subtree
from .config import Config
from .data.broad import TRIAL_GROUPS
from .ekf import run
from .metrics import (
    heading_drift_deg,
    inclination_rmse_deg,
    nees_fraction_inside,
    nis_fraction_inside,
    paired_bootstrap_ci,
    total_error_deg,
)
from .nets import LearnedEKF
from .quaternion import log_so3, qconj, qmul
from .train import fit


@app.command()
def eval(
    quick: bool = typer.Option(False, "--quick"),
    device: str = typer.Option("cpu", "--device"),
    force: bool = typer.Option(False, "--force", help="allow re-running eval on the same checkpoint"),
):
    """Evaluate every method on the held-out test split.

    Refuses to run twice on the same checkpoint without --force; logs every run.
    """
    config = Config()
    results_dir = Path(config.results_dir)
    log_path = results_dir / "eval.log"
    if log_path.exists() and not force:
        raise RuntimeError(
            "eval already ran on this checkpoint (results/eval.log exists). "
            "Use --force to overwrite, or remove results/ first."
        )
    _select_device(device)
    sequences = _ensure_data(config, quick)
    split = json.loads(Path(config.splits_file).read_text())
    test_names = [n for n in split["test"] if n in sequences]
    if not test_names:
        raise ValueError("no test sequences available")

    ckpt = results_dir / "checkpoints" / "best.npz"
    params = None
    if ckpt.exists():
        loaded = dict(np.load(ckpt, allow_pickle=False))
        params = {"bias": _subtree(loaded, "bias"), "scale": _subtree(loaded, "scale")}

    results = {"trials": {}, "summary": {}}
    for name in test_names:
        od = _run_all_methods(sequences[name], config, params)
        movement = od["_meta"]["movement"]
        trial_out = {}
        for method in od:
            if method == "_meta":
                continue
            trial_out[method] = _metrics_from_output(od[method], movement, config)
        results["trials"][name] = trial_out

    methods = [m for m in results["trials"][test_names[0]]]
    summary = {}
    for method in methods:
        vals = [results["trials"][n][method]["inclination_rmse_deg"] for n in test_names]
        summary[method] = {"mean_inclination_rmse_deg": float(np.mean(vals))}
    if "learned" in methods and "eskf_numpy" in methods:
        base = np.array([results["trials"][n]["eskf_numpy"]["inclination_rmse_deg"] for n in test_names])
        learned = np.array([results["trials"][n]["learned"]["inclination_rmse_deg"] for n in test_names])
        mean, lo, hi = paired_bootstrap_ci(base - learned, n_resamples=config.bootstrap_n, seed=config.seed)
        summary["improvement_over_eskf_numpy_deg"] = {"mean": mean, "ci_lo": lo, "ci_hi": hi}
    results["summary"] = summary

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{datetime.now().isoformat()} eval test={test_names} quick={quick}\n")
    console.print(json.dumps(results, indent=2, default=float))


def _run_all_methods(seq: Sequence, config: Config, params=None):
    """Run every method on one sequence; return dict of per-method outputs."""
    t, gyro, accel, quat, movement = seq.as_jax()
    gyro = np.asarray(gyro)
    accel = np.asarray(accel)
    dt = np.asarray(seq.dt(), dtype=np.float64)
    quat = np.asarray(quat)
    movement = np.asarray(movement)
    init_state = _init_for_eval(seq, config)
    out = {}

    if params is not None:
        net = LearnedEKF(config, params["bias"], params["scale"])
        _, o = net.run(jnp.asarray(gyro), jnp.asarray(accel), jnp.asarray(dt), init_state)
        out["learned"] = _pack(o)
    _, o = run(jnp.asarray(gyro), jnp.asarray(accel), jnp.asarray(dt), config, init_state)
    out["ekf_classical"] = _pack(o)

    from .baselines import gyro as gyro_mod, complementary, madgwick, mahony
    from .eskf_numpy import run as eskf_run

    qg, _ = gyro_mod.run(gyro, accel, dt, 0.0)
    out["gyro"] = dict(q=qg)
    qc, _ = complementary.run(gyro, accel, dt, 0.0)
    out["complementary"] = dict(q=qc)
    qm, _ = madgwick.run(gyro, accel, dt, 0.0)
    out["madgwick"] = dict(q=qm)
    qmh, _ = mahony.run(gyro, accel, dt, 0.0)
    out["mahony"] = dict(q=qmh)
    qn, _, pdiag, innov, nis_arr = eskf_run(gyro, accel, dt, 0.0)
    out["eskf_numpy"] = dict(q=qn, p_diag=pdiag, innovation=innov, nis=nis_arr)
    out["_meta"] = dict(quat=quat, movement=movement, dt=dt, t=t)
    return out


def _pack(o):
    return dict(
        q=np.asarray(o.q),
        p_diag=np.asarray(o.p_diag),
        covariance=np.asarray(o.covariance),
        innovation=np.asarray(o.innovation),
        innovation_cov=np.asarray(o.innovation_cov),
    )


def _attitude_error(q_est, quat_gt):
    q_est = np.asarray(q_est, dtype=np.float64)
    quat_gt = np.asarray(quat_gt, dtype=np.float64)
    return np.stack([log_so3(qmul(q_est[i], qconj(quat_gt[i]))) for i in range(len(q_est))])


def _metrics_from_output(od, movement, config):
    quat = od["_meta"]["quat"]
    q = od["q"]
    m = {
        "inclination_rmse_deg": inclination_rmse_deg(q, quat, movement),
        "heading_drift_deg": heading_drift_deg(q, quat, movement),
        "total_error_deg": total_error_deg(q, quat, movement),
    }
    if "p_diag" in od:
        cov = od.get("covariance")
        if cov is not None:
            p_att = np.asarray(cov)[:, :3, :3]
            err = _attitude_error(q, quat)
            from .metrics import nees as _nees

            nees_vals = _nees(err, p_att)
            m["nees_mean"] = float(np.mean(nees_vals))
            m["nees_frac_inside"] = nees_fraction_inside(nees_vals, config.nis_alpha)
        if "innovation" in od and "innovation_cov" in od:
            m["nis_frac_inside"] = nis_fraction_inside(
                od["innovation"], od["innovation_cov"], config.nis_alpha
            )
    return m


@app.command()
def ablate(
    quick: bool = typer.Option(False, "--quick"),
    device: str = typer.Option("cpu", "--device"),
):
    """Ablation study on test trials: no bias corrector, no noise scaler, no gate."""
    _select_device(device)
    config = Config()
    sequences = _ensure_data(config, quick)
    split = json.loads(Path(config.splits_file).read_text())
    test_names = [n for n in split["test"] if n in sequences]
    ckpt = Path(config.results_dir) / "checkpoints" / "best.npz"
    params = None
    if ckpt.exists():
        loaded = dict(np.load(ckpt, allow_pickle=False))
        params = {"bias": _subtree(loaded, "bias"), "scale": _subtree(loaded, "scale")}

    out = {}
    for name in test_names:
        seq = sequences[name]
        t, gyro, accel, quat, movement = seq.as_jax()
        gyro = np.asarray(gyro)
        accel = np.asarray(accel)
        dt = np.asarray(seq.dt(), dtype=np.float64)
        quat = np.asarray(quat)
        movement = np.asarray(movement)
        init_state = _init_for_eval(seq, config)
        trial_out = {}

        def evaluate(bias_params, scale_params, label):
            net = LearnedEKF(config, bias_params, scale_params)
            _, o = net.run(jnp.asarray(gyro), jnp.asarray(accel), jnp.asarray(dt), init_state)
            od = _pack(o)
            od["_meta"] = dict(quat=quat, movement=movement, dt=dt, t=t)
            return _metrics_from_output(od, movement, config)

        trial_out["learned_full"] = evaluate(params["bias"], params["scale"], "learned_full") if params else None
        if params:
            import jax.tree_util as jtu

            zero_bias = jtu.map(lambda x: jnp.zeros_like(x), params["bias"])
            one_scale = jtu.map(lambda x: jnp.ones_like(x), params["scale"])
            trial_out["no_bias_corrector"] = evaluate(zero_bias, params["scale"], "no_bias")
            trial_out["no_noise_scaler"] = evaluate(params["bias"], one_scale, "no_scale")
        # classical: no learned components at all (gate remains as configured)
        _, o = run(jnp.asarray(gyro), jnp.asarray(accel), jnp.asarray(dt), config, init_state)
        od = _pack(o)
        od["_meta"] = dict(quat=quat, movement=movement, dt=dt, t=t)
        trial_out["classical_ekf"] = _metrics_from_output(od, movement, config)
        # gate off
        gate_off = Config(
            seed=config.seed, gate_gain=0.0,
            train_frac=config.train_frac, val_frac=config.val_frac, test_frac=config.test_frac,
            gyro_noise_density=config.gyro_noise_density,
            bias_rw_density=config.bias_rw_density,
            accel_noise_std=config.accel_noise_std,
            gate_deadband=config.gate_deadband,
            init_window=config.init_window,
            window=config.window, hidden=config.hidden, lr=config.lr, grad_clip=config.grad_clip,
            nis_lambda=config.nis_lambda,
            epochs=config.epochs, patience=config.patience,
            bptt_window=config.bptt_window, batch_sequences=config.batch_sequences,
            tune_trials=config.tune_trials, tune_bptt_windows=config.tune_bptt_windows,
            nis_alpha=config.nis_alpha, bootstrap_n=config.bootstrap_n,
            data_raw=config.data_raw, results_dir=config.results_dir,
            splits_file=config.splits_file, results_file=config.results_file,
            manifest_file=config.manifest_file,
        )
        _, o = run(jnp.asarray(gyro), jnp.asarray(accel), jnp.asarray(dt), gate_off, init_state)
        od = _pack(o)
        od["_meta"] = dict(quat=quat, movement=movement, dt=dt, t=t)
        trial_out["gate_off"] = _metrics_from_output(od, movement, gate_off)
        out[name] = trial_out

    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.results_dir) / "ablations.json").write_text(json.dumps(out, indent=2, default=float))
    console.print(json.dumps(out, indent=2, default=float))


@app.command()
def plot(
    quick: bool = typer.Option(False, "--quick"),
):
    """Figures from results/results.json."""
    config = Config()
    results_path = Path(config.results_dir) / "results.json"
    if not results_path.exists():
        console.print("[yellow]no results/results.json; nothing to plot[/yellow]")
        return
    results = json.loads(results_path.read_text())
    figures_dir = Path(config.results_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]matplotlib not installed; skipping plots[/yellow]")
        return

    names = list(results["trials"].keys())
    if names:
        methods = [m for m in results["trials"][names[0]]]
        x = np.arange(len(names))
        width = 0.8 / max(len(methods), 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, method in enumerate(methods):
            vals = [results["trials"][n][method]["inclination_rmse_deg"] for n in names]
            ax.bar(x + i * width, vals, width, label=method)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("inclination RMSE (deg)")
        ax.set_title("Per-trial inclination RMSE (test trials)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "per_trial_inclination.png")
        plt.close(fig)
    console.print(f"wrote figures to {figures_dir}")


@app.command()
def bench(
    quick: bool = typer.Option(False, "--quick"),
    device: str = typer.Option("cpu", "--device"),
):
    """Latency / throughput benchmarks."""
    _select_device(device)
    from .benchmarks.latency import main as bench_main

    bench_main(quick=quick)


@app.command()
def reproduce(
    quick: bool = typer.Option(False, "--quick"),
    device: str = typer.Option("cpu", "--device"),
):
    """Run the full pipeline end to end: data, split, tune, train, eval,
    ablate, plot, fill-docs."""
    _select_device(device)
    for sub, kwargs in [
        ("data", dict(quick=quick)),
        ("split", dict(quick=quick)),
        ("tune", dict(quick=quick, device=device)),
        ("train", dict(quick=quick, device=device)),
        ("eval", dict(quick=quick, device=device, force=True)),
        ("ablate", dict(quick=quick, device=device)),
        ("plot", dict(quick=quick)),
    ]:
        app.main([sub] + _cli_args(**kwargs), standalone_mode=False)
    app.main(["fill-docs"], standalone_mode=False)


def _cli_args(quick=False, device="cpu", force=False):
    args = []
    if quick:
        args.append("--quick")
    args += ["--device", device]
    if force:
        args.append("--force")
    return args


@app.command(name="fill-docs")
def fill_docs():
    """Render README results section and portfolio templates from results.json."""
    from .scripts.fill_docs import main as fill_main

    fill_main()
