"""Smoke test for training: runs on the real excerpt, is finite and deterministic.
"""

from __future__ import annotations

import math

from diffekf.config import Config
from diffekf.data.broad import load_trial
from diffekf.train import fit

FIXTURE = "tests/fixtures/fixture_broad_trial01_excerpt.hdf5"


def test_fit_runs_and_is_deterministic():
    seq = load_trial(FIXTURE, "01_undisturbed_slow_rotation_A")
    config = Config(epochs=3, patience=3, bptt_window=200)
    out1 = fit(config, [seq], [seq], results_dir="results/train_smoke", quick=True, key=0)
    out2 = fit(config, [seq], [seq], results_dir="results/train_smoke", quick=True, key=0)
    assert len(out1["history"]) == 3
    assert all(math.isfinite(h["train_loss"]) for h in out1["history"])
    assert out1["history"] == out2["history"]
