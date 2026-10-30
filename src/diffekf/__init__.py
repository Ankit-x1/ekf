"""diffekf: learned differentiable correction for a multiplicative error-state
EKF attitude estimator (6-axis IMU, no magnetometer).

Conventions: Hamilton quaternions, scalar-first, body -> world, world z up,
a static accelerometer reads R(q).T @ [0, 0, +g].

float64 is enabled globally because this codebase relies on it for
correctness (covariance PSD, Jacobian agreement with autodiff).
"""
import jax

jax.config.update("jax_enable_x64", True)

from .config import Config
from .ekf import EKFState, EKFStepOutput, initialise_state, run, step
from .nets import LearnedEKF, bias_corrector_init, compute_corrections, noise_scaler_init
from .data.common import Sequence
from .data.broad import BROAD_TRIALS, DISTURBED_TRIALS, UNDISTURBED_TRIALS, load_all, load_trial
from .data.euroc import EUROC_VICON_SEQUENCES, estimate_time_offset, load_sequence

__all__ = [
    "Config",
    "EKFState",
    "EKFStepOutput",
    "initialise_state",
    "run",
    "step",
    "LearnedEKF",
    "bias_corrector_init",
    "compute_corrections",
    "noise_scaler_init",
    "Sequence",
    "BROAD_TRIALS",
    "DISTURBED_TRIALS",
    "UNDISTURBED_TRIALS",
    "load_all",
    "load_trial",
    "EUROC_VICON_SEQUENCES",
    "estimate_time_offset",
    "load_sequence",
]

__version__ = "0.1.0"
