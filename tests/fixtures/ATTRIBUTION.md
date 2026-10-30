# Test fixture attribution

File: `fixture_broad_trial01_excerpt.hdf5`

This is a short excerpt (first 600 samples, ~2.1 s at 285.714 Hz) of BROAD trial
`01_undisturbed_slow_rotation_A`. It is real data, not synthetic, and is used
only inside unit tests (frame conversion, loader test, baseline comparison). No
reported number in this repository is produced from it.

Source
------
Berlin Robust Orientation Estimation Assessment Dataset (BROAD)
D. Laidig, M. Caruso, A. Cereatti, T. Seel, "BROAD -- A Benchmark for Robust
Inertial Orientation Estimation", Data, 6(7):1-20, 2021.
https://doi.org/10.3390/data6070072
Repository: https://github.com/dlaidig/broad  (data_hdf5/01_undisturbed_slow_rotation_A.hdf5)

License
-------
CC BY 4.0 (Creative Commons Attribution 4.0 International).
Credit: Daniel Laidig, Kai Brands (recording), contributors listed in the
dataset publication.

Channels in the excerpt
-----------------------
imu_gyr (rad/s), imu_acc (m/s^2, specific force), imu_mag (µT, unused),
movement (bool mask), opt_quat (unit quaternion w/x/y/z, body -> world, ENU),
opt_pos (m), sampling_rate (Hz attribute).

Frames
------
opt_quat was verified numerically to follow the same convention as this
repository: Hamilton, scalar-first, body -> world, world z up; a static
sensor reads R(q).T @ [0, 0, +g].
