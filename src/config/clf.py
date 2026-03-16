from dataclasses import dataclass
from src.config.shared import SharedArgs


@dataclass
class ClfArgs(SharedArgs):
    # CLF parameters
    lambda_x: float = 10.0    # Exponential decay parameter for CLF
    lambda_xy: float = 10.0   # Exponential decay parameter for CLF for xy phase
    chirality_factor: float = 0.0  # Weight for chirality term
    lr_duals: float = 3e-4  # Learning rate for dual variables (duals_learner)
    barrier: float = 1.0      # Barrier strength for dual norm constraints (duals_learner)
    eigenvalue_estimation_method: str = 'separate'  # How to estimate eigenvalue targets: 'separate' (independent x/y EMA), 'average' (shared averaged EMA), 'two_sided' (bi-orthogonal complex division)
    normalize_eigenvalue_targets: bool = False  # Normalize Rayleigh quotient targets by state norms before EMA update
    use_sg_ip: bool = True  # Apply stop_gradient to smaller-indexed eigenvector in bi-orthogonality inner products to break symmetry
