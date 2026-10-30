"""Baselines: classical filter beats gyro-only integration on the real excerpt.
"""

from __future__ import annotations

import numpy as np

from diffekf.baselines import complementary, gyro as gyro_mod, madgwick, mahony
from diffekf.data.broad import load_trial
from diffekf.metrics import inclination_rmse_deg
from diffekf.quaternion import quat_to_rotmat

FIXTURE = "tests/fixtures/fixture_broad_trial01_excerpt.hdf5"


def _yaw0(quat):
    R0 = np.asarray(quat_to_rotmat(quat[0]))
    return float(np.arctan2(R0[1, 0], R0[0, 0]))


def test_classical_filter_beats_gyro_only_on_excerpt():
    seq = load_trial(FIXTURE, "01_undisturbed_slow_rotation_A")
    gyro = np.asarray(seq.gyro)
    accel = np.asarray(seq.accel)
    dt = seq.dt()
    quat = np.asarray(seq.quat)
    movement = np.asarray(seq.movement)
    yaw0 = _yaw0(quat)

    q_gyro, _ = gyro_mod.run(gyro, accel, dt, yaw0)
    q_comp, _ = complementary.run(gyro, accel, dt, yaw0)
    q_mah, _ = mahony.run(gyro, accel, dt, yaw0)
    q_mad, _ = madgwick.run(gyro, accel, dt, yaw0)

    err_gyro = inclination_rmse_deg(q_gyro, quat, movement)
    err_comp = inclination_rmse_deg(q_comp, quat, movement)
    err_mah = inclination_rmse_deg(q_mah, quat, movement)
    err_mad = inclination_rmse_deg(q_mad, quat, movement)

    assert err_comp < err_gyro, (err_comp, err_gyro)
    assert err_mah < err_gyro, (err_mah, err_gyro)
    assert err_mad < err_gyro, (err_mad, err_gyro)
