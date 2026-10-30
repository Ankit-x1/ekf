"""Command line interface: ``python -m diffekf <subcommand>``.

Subcommands: data, split, tune, train, eval, ablate, plot, bench, reproduce,
fill-docs. Each supports ``--quick`` (shrunk budget for smoke testing) and
``--device`` (cpu / gpu). Progress is logged to results/.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer
from rich.console import Console

from .config import Config
from .data.broad import load_all, load_trial
from .data.common import Sequence
from .ekf import EKFState, initialise_state, run
from .metrics import heading_drift_deg, inclination_rmse_deg, paired_bootstrap_ci, total_error_deg
from .nets import LearnedEKF
from .train import fit

app = typer.Typer(help="diffekf: learned error-state EKF attitude estimation")
console = Console()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _select_device(device: str) -> str:
    if device == "gpu":
        gpus = jax.devices("gpu")
        if gpus:
            console.print(f"[green]Using GPU: {gpus[0]}[/green]")
            return "gpu"
        console.print("[yellow]No GPU visible to JAX; falling back to CPU[/yellow]")
    console.print("[green]Using CPU[/green]")
    return "cpu"


def _ensure_data(config: Config, quick: bool) -> dict:
    """Return {trial_name: Sequence} for available real data.

    In quick mode, if nothing has been downloaded, fall back to the licensed
    fixture excerpt in tests/fixtures so the pipeline can smoke-test offline.
    """
    raw = Path(config.data_raw)
    files = sorted(raw.glob("*.hdf5")) if raw.exists() else []
    if files:
        names = [f.stem for f in files]
        return load_all(config.data_raw, tuple(names))
    if quick:
        fixture = Path("tests/fixtures/fixture_broad_trial01_excerpt.hdf5")
        if not fixture.exists():
            raise FileNotFoundError(
                "No data in data/raw and no tests/fixtures fixture. "
                "Run: python -m diffekf data  (full BROAD download), or see "
                "DATASETS.md for manual download instructions."
            )
        seq = load_trial(str(fixture), "01_undisturbed_slow_rotation_A")
        return {"01_undisturbed_slow_rotation_A": seq}
    raise FileNotFoundError(
        f"No data found in {config.data_raw}. Run `python -m diffekf data` "
        "first (see DATASETS.md for manual download instructions)."
    )


def _init_for_eval(seq: Sequence, config: Config) -> EKFState:
    from .quaternion import quat_to_rotmat

    _, gyro, accel, quat, _ = seq.as_jax()
    R0 = quat_to_rotmat(quat[0])
    yaw0 = float(jnp.arctan2(R0[1, 0], R0[0, 0]))
    return initialise_state(accel, gyro, yaw0, config)


def _subtree(loaded: dict, prefix: str) -> dict:
    out = {}
    for k in loaded.keys():
        if k.startswith(prefix + "/"):
            out[k[len(prefix) + 1:]] = np.asarray(loaded[k])
    return out


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@app.command()
def data(
    quick: bool = typer.Option(False, "--quick", help="smoke mode: use local fixture only"),
):
    """Download and verify datasets under data/raw; write results/data_manifest.json."""
    if quick:
        console.print("[yellow]--quick: skipping download; fixture will be used by other stages[/yellow]")
        _write_manifest_quick()
        return
    from .scripts.download_data import main as download_main

    download_main()


def _write_manifest_quick():
    import hashlib

    fixture = Path("tests/fixtures/fixture_broad_trial01_excerpt.hdf5")
    results = Path("results")
    results.mkdir(parents=True, exist_ok=True)
    entries = []
    if fixture.exists():
        blob = fixture.read_bytes()
        entries.append({
            "name": "fixture_broad_trial01_excerpt.hdf5",
            "path": str(fixture),
            "size": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "dataset": "BROAD (excerpt, CC BY 4.0)",
            "trials": ["01_undisturbed_slow_rotation_A"],
            "note": "licensed real excerpt; full BROAD not downloaded in --quick mode",
        })
    (results / "data_manifest.json").write_text(json.dumps(entries, indent=2))
    console.print(f"wrote {results / 'data_manifest.json'}")


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
@app.command()
def split(
    quick: bool = typer.Option(False, "--quick"),
):
    """Stratified train/val/test split by BROAD trial -> results/splits.json."""
    config = Config()
    sequences = _ensure_data(config, quick)
    from .data.broad import TRIAL_GROUPS

    groups = {}
    for name in sequences.keys():
        g = tuple(TRIAL_GROUPS.get(name, ("all_trials",)))
        groups.setdefault(g, []).append(name)

    rng = np.random.default_rng(config.seed)
    train, val, test = [], [], []
    for g, names in groups.items():
        names = list(names)
        rng.shuffle(names)
        n = len(names)
        n_train = max(1, int(round(config.train_frac * n)))
        n_val = max(0, int(round(config.val_frac * n)))
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_train -= 1
            n_test = 1
        train += names[:n_train]
        val += names[n_train:n_train + n_val]
        test += names[n_train + n_val:]

    split = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "seed": config.seed,
        "fractions": [config.train_frac, config.val_frac, config.test_frac],
    }
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.splits_file).write_text(json.dumps(split, indent=2))
    console.print(json.dumps(split, indent=2))


# --------------------------------------------------------------------------- #
# tune
# --------------------------------------------------------------------------- #
@app.command()
def tune(
    quick: bool = typer.Option(False, "--quick"),
    device: str = typer.Option("cpu", "--device"),
):
    """Random-search tuning of every method on validation trials only."""
    _select_device(device)
    config = Config()
    sequences = _ensure_data(config, quick)
    split = json.loads(Path(config.splits_file).read_text())
    val_names = split["val"]
    n_iter = 3 if quick else config.tune_trials
    console.print(f"tuning on validation trials {val_names} ({n_iter} random draws per method)")
    _tune_random(sequences, val_names, config, n_iter)


def _tune_random(sequences, val_names, config, n_iter):
    from .baselines import complementary, madgwick, mahony, gyro as gyro_mod

    results = {}
    for name in val_names:
        seq = sequences[name]
        t, gyro, accel, quat, movement = seq.as_jax()
        gyro = np.asarray(gyro)
        accel = np.asarray(accel)
        dt = np.asarray(seq.dt(), dtype=np.float64)
        quat = np.asarray(quat)
        movement = np.asarray(movement)

        def score(q_est):
            return inclination_rmse_deg(q_est, quat, movement)

        results[name] = {}
        qg, _ = gyro_mod.run(gyro, accel, dt, 0.0)
        results[name]["gyro"] = score(qg)
        rng = np.random.default_rng(config.seed)
        best = {
            "complementary": (1e9, None),
            "madgwick": (1e9, None),
            "mahony": (1e9, None),
            "eskf": (1e9, None),
        }
        for _ in range(n_iter):
            kp = float(10 ** rng.uniform(-1, 1))
            qc, _ = complementary.run(gyro, accel, dt, 0.0, kp=kp)
            s = score(qc)
            if s < best["complementary"][0]:
                best["complementary"] = (s, kp)
            beta = float(10 ** rng.uniform(-2, 0.5))
            qm, _ = madgwick.run(gyro, accel, dt, 0.0, beta=beta)
            s = score(qm)
            if s < best["madgwick"][0]:
                best["madgwick"] = (s, beta)
            Kp = float(10 ** rng.uniform(-1, 1))
            Ki = float(10 ** rng.uniform(-4, -1))
            qmh, _ = mahony.run(gyro, accel, dt, 0.0, Kp=Kp, Ki=Ki)
            s = score(qmh)
            if s < best["mahony"][0]:
                best["mahony"] = (s, (Kp, Ki))
        results[name].update({k: v[0] for k, v in best.items()})
        results[name]["best_params"] = {k: v[1] for k, v in best.items()}
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir, "tuning.json").write_text(json.dumps(results, indent=2, default=float))
    console.print(json.dumps(results, indent=2, default=float))


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
@app.command()
def train(
    quick: bool = typer.Option(False, "--quick"),
    device: str = typer.Option("cpu", "--device"),
):
    """Train the learned corrections on training trials; save checkpoints."""
    _select_device(device)
    config = Config()
    sequences = _ensure_data(config, quick)
    split = json.loads(Path(config.splits_file).read_text())
    train_seqs = [sequences[n] for n in split["train"] if n in sequences]
    val_seqs = [sequences[n] for n in split["val"] if n in sequences]
    if not train_seqs:
        raise ValueError("no training sequences available")
    if not val_seqs:
        val_seqs = train_seqs
    fit(config, train_seqs, val_seqs, config.results_dir, quick=quick, key=config.seed)
