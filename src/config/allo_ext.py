from dataclasses import dataclass
from src.config.shared import SharedArgs


@dataclass
class AlloExtArgs(SharedArgs):
    # Data collection (override SharedArgs defaults to match al reference: 200k samples, 50-step episodes)
    num_envs: int = 4000
    num_steps: int = 50
    max_episode_length: int = 50

    # Encoder optimizer
    learning_rate: float = 0.001

    # Barrier (quadratic penalty)
    barrier_initial_val: float = 2.0      # Initial barrier coefficient
    max_barrier_coefs: float = 10000.0    # Cap on barrier coefficient
    lr_barrier_coefs: float = 1.0         # Barrier growth rate (applied to mean positive quad error)
    min_barrier_coefs: float = 0.0        # Floor on barrier coefficient

    # Dual variables (updated externally like laplacian_dual_dynamics/al.py)
    duals_initial_val: float = 0.0        # Initial dual value (full lower-triangular)
    lr_duals: float = 0.0001              # Dual learning rate
    min_duals: float = -100.0             # Clip floor for dual variables
    max_duals: float = 100.0              # Clip ceiling for dual variables

    # EMA smoothing for error estimates
    error_update_rate: float = 1.0        # EMA rate for linear errors (1.0 = no smoothing)
    q_error_update_rate: float = 0.1      # EMA rate for quadratic errors

    # Ground truth eigenvector computation
    sym_eig: bool = False  # If True, use eigh (symmetric solver) for GT eigenvectors; imaginary parts are set to 0

    # Graph perturbation (kept for compatibility, default off)
    graph_epsilon: float = 0.01
    graph_variance_scale: float = 0.1
    perturbation_type: str = 'none'
