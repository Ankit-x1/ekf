"""Loader test on the licensed real BROAD excerpt in tests/fixtures (CC BY 4.0;
see tests/fixtures/ATTRIBUTION.md). No reported number is produced from it.
"""

from __future__ import annotations

import numpy as np

from diffekf.data.broad import BROAD_TRIALS, DISTURBED_TRIALS, UNDISTURBED_TRIALS, load_trial

FIXTURE = "tests/fixtures/fixture_broad_trial01_excerpt.hdf5"


def test_trial_list_counts():
    assert len(BROAD_TRIALS) == 39
    assert len(UNDISTURBED_TRIALS) == 23
    assert len(DISTURBED_TRIALS) == 16
    assert set(UNDISTURBED_TRIALS) | set(DISTURBED_TRIALS) == set(BROAD_TRIALS)


def test_fixture_loads_and_validate():
    seq = load_trial(FIXTURE, "01_undisturbed_slow_rotation_A")
    assert seq.trial == "01_undisturbed_slow_rotation_A"
    assert seq.gyro.shape == (600, 3)
    assert seq.accel.shape == (600, 3)
    assert seq.quat.shape == (600, 4)
    assert seq.movement.shape == (600,)
    assert np.all(np.isfinite(seq.gyro)) and np.all(np.isfinite(seq.accel))
    assert np.all(np.abs(np.linalg.norm(seq.quat, axis=1) - 1.0) < 1e-3)
    assert seq.source.startswith("BROAD/")


def test_dt_monotonic_and_positive():
    seq = load_trial(FIXTURE, "01_undisturbed_slow_rotation_A")
    dt = seq.dt()
    assert len(dt) == len(seq)
    assert np.all(dt > 0)
    assert np.all(np.isfinite(dt))


def test_as_jax_returns_float64():
    import jax.numpy as jnp

    seq = load_trial(FIXTURE, "01_undisturbed_slow_rotation_A")
    t, gyro, accel, quat, movement = seq.as_jax()
    for arr in (t, gyro, accel, quat):
        assert arr.dtype == jnp.float64
