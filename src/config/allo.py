from dataclasses import dataclass
from src.config.shared import SharedArgs


@dataclass
class AlloArgs(SharedArgs):
    # ALLO-specific (augmented Lagrangian)
    duals_initial_val: float = -2.0       # Initial value for dual variables
    barrier_initial_val: float = 0.5      # Initial barrier coefficient
    max_barrier_coefs: float = 0.5        # Maximum barrier coefficient value
    step_size_duals: float = 1.0          # SGD step size for dual variables
    step_size_duals_I: float = 0.0        # Integral term step size for duals
    integral_decay: float = 0.99          # EMA decay for constraint-error integral
    init_dual_diag: bool = False          # If True, init only the diagonal; if False (default), init the full lower triangle
    graph_epsilon: float = 0.01           # Graph perturbation strength
    graph_variance_scale: float = 0.1     # Variance scale for graph perturbation
    perturbation_type: str = 'none'       # 'none', 'exponential', 'squared', 'squared-null-grad'
