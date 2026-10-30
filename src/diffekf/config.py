"""Single frozen dataclass holding every hyperparameter of the project.

The dataclass is frozen so tuning runs cannot accidentally mutate a shared
config instance; every stage (split, tune, train, eval, ablate) receives an
instance of this class and treats it as read-only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ data
    seed: int = 20261030
    train_frac: float = 0.60
    val_frac: float = 0.15
    test_frac: float = 0.25
    # BROAD group names used for stratified splitting (motion / disturbance).
    motion_groups: tuple[str, ...] = ("rotation", "translation", "combined")
    speed_groups: tuple[str, ...] = ("slow", "fast")
    break_groups: tuple[str, ...] = ("no_breaks", "with_breaks")
    disturbance_groups: tuple[str, ...] = (
        "tapping",
        "vibration",
        "stationary_magnet",
        "attached_magnet",
        "office_environment",
        "mixed",
    )

    # ------------------------------------------------------------------ filter
    gyro_noise_density: float = 1e-3      # rad / s / sqrt(Hz)
    bias_rw_density: float = 1e-5         # rad / s^2 / sqrt(Hz)
    accel_noise_std: float = 2e-2         # m / s^2 per sample
    gate_gain: float = 10.0               # accel-magnitude gate gain
    gate_deadband: float = 0.2            # m / s^2 deadband around 1 g
    init_window: int = 100                # samples used for static tilt initialisation

    # ------------------------------------------------------------------ learned components
    window: int = 50                      # causal window (samples) for the nets
    hidden: int = 16                      # hidden width of the small MLPs
    lr: float = 1e-3                      # Adam learning rate
    grad_clip: float = 1.0                # global-norm clipping
    nis_lambda: float = 1e-2              # weight of the NIS-consistency term

    # ------------------------------------------------------------------ training
    epochs: int = 200
    patience: int = 20                    # early-stopping patience (epochs)
    bptt_window: int = 250                # truncated-BPTT window length (samples)
    batch_sequences: int = 4              # sequences per training step (vmap batch)

    # ------------------------------------------------------------------ tuning
    tune_trials: int = 24                 # random-search iterations per method
    tune_bptt_windows: int = 6            # windows used per sequence while tuning

    # ------------------------------------------------------------------ evaluation
    nis_alpha: float = 0.05               # chi-square significance level
    bootstrap_n: int = 2000               # paired bootstrap resamples

    # ------------------------------------------------------------------ paths
    data_raw: str = "data/raw"
    results_dir: str = "results"
    splits_file: str = "results/splits.json"
    results_file: str = "results/results.json"
    manifest_file: str = "results/data_manifest.json"

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"train/val/test fractions must sum to 1, got {total}")
        for name in ("train_frac", "val_frac", "test_frac"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value}")
