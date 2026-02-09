"""
Exact gradient dynamics for learning complex eigenvectors.

This implements the exact gradient formulas from Complex_ALLO.pdf:

For left eigenvector x_i:
  g_{x_i} = Re(<x̄_i, Ly_i>_ρ) * D*L*ȳ_i - Σ_{k=1}^{i} (α_{x,ik} + iβ_{x,ik} - b*h_{x,ik}) * D*ȳ_k

For right eigenvector y_i:
  g_{y_i} = Re(<x̄_i, Ly_i>_ρ) * L^T*D*x̄_i - Σ_{k=1}^{i} (α_{y,ik} + iβ_{y,ik} - b*h_{y,ik}) * D*x̄_k

For dual variables:
  g_{α_{x,jk}} = Re(h_{x,jk})
  g_{α_{y,jk}} = Re(h_{y,jk})
  g_{β_{x,jk}} = Im(h_{x,jk})
  g_{β_{y,jk}} = Im(h_{y,jk})

Update rule: Θ[t+1] = Θ[t] - α * g_Θ(Θ[t])
(Duals use negative gradient = gradient ascent)
"""

import os
import sys
import random
import time
import json
import pickle
from typing import Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import jax
import jax.numpy as jnp
import tyro
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.envs.gridworld import GridWorldEnv
from src.envs.env import create_environment_from_text, EXAMPLE_ENVIRONMENTS
from src.envs.door_gridworld import (
    create_door_gridworld_from_base,
    create_random_doors,
)
from src.data_collection import collect_transition_counts_and_episodes
from src.utils.plotting import (
    visualize_multiple_eigenvectors,
)
from src.utils.laplacian import compute_eigendecomposition


def get_transition_matrix(
    transition_counts: jnp.ndarray,
    make_stochastic: bool = True,
) -> jnp.ndarray:
    """Build transition matrix from counts (non-symmetrized)."""
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)
    else:
        transition_matrix = transition_counts

    if make_stochastic:
        row_sums = jnp.sum(transition_matrix.clip(1e-8), axis=1, keepdims=True)
        transition_matrix = transition_matrix.clip(1e-8) / row_sums

    return transition_matrix


def compute_successor_representation(
    transition_matrix: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    """Compute the successor representation SR_γ = (I - γP)^(-1)."""
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)
    sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix)
    return sr_matrix


def compute_sampling_distribution(
    transition_counts: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the empirical sampling distribution from transition counts."""
    if len(transition_counts.shape) == 3:
        state_visit_counts = jnp.sum(transition_counts, axis=(1, 2))
    else:
        state_visit_counts = jnp.sum(transition_counts, axis=1)

    total_visits = jnp.sum(state_visit_counts)
    sampling_probs = state_visit_counts / total_visits
    return sampling_probs


def compute_nonsymmetric_laplacian(
    transition_matrix: jnp.ndarray,
    gamma: float,
    delta: float = 0.0,
) -> jnp.ndarray:
    """Compute the non-symmetric Laplacian L = (1+δ)I - (1-γ)P·SR_γ."""
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)
    sr_matrix = compute_successor_representation(transition_matrix, gamma)
    laplacian = (1 + delta) * identity - (1 - gamma) * (transition_matrix @ sr_matrix)
    return laplacian


@dataclass
class Args:
    # Environment
    env_type: str = "room4"
    env_file: str | None = None
    env_file_name: str | None = None
    max_episode_length: int = 1000

    # Data collection
    num_envs: int = 1000
    num_steps: int = 1000

    # Irreversible doors
    use_doors: bool = False
    num_doors: int = 5
    door_seed: int = 42

    # Model
    num_eigenvectors: int = 1  # Start with just 1 to verify convergence

    # Training
    learning_rate: float = 0.01
    num_gradient_steps: int = 10000
    gamma: float = 0.2
    delta: float = 0.1

    # Augmented Lagrangian parameters
    barrier_coef: float = 4.0  # b > 2 for stability (theory says b > 2)

    # Logging and saving
    log_freq: int = 100
    plot_freq: int = 1000
    save_model: bool = True
    plot_during_training: bool = True
    results_dir: str = "./results"

    # Misc
    seed: int = 42
    exp_name: str | None = None
    exp_number: int = 0


def get_canonical_free_states(env):
    """Get the canonical set of free (non-obstacle) states from the environment."""
    width = env.width
    height = env.height
    all_states = set(range(width * height))

    obstacle_states = set()
    if env.has_obstacles:
        for obs in env.obstacles:
            obs_x, obs_y = int(obs[0]), int(obs[1])
            if 0 <= obs_x < width and 0 <= obs_y < height:
                state_idx = obs_y * width + obs_x
                obstacle_states.add(state_idx)

    free_states = sorted(all_states - obstacle_states)
    return jnp.array(free_states, dtype=jnp.int32)


def create_gridworld_env(args: Args):
    """Create a gridworld environment from text file or example."""
    if args.env_type == 'file':
        if args.env_file is None and args.env_file_name is None:
            raise ValueError("Must specify either env_file or env_file_name when env_type='file'")

        env = create_environment_from_text(
            file_path=args.env_file,
            file_name=args.env_file_name,
            max_steps=args.max_episode_length,
            precision=32,
        )
    else:
        if args.env_type not in EXAMPLE_ENVIRONMENTS:
            raise ValueError(f"Unknown env_type: {args.env_type}")

        text_content = EXAMPLE_ENVIRONMENTS[args.env_type]
        env = create_environment_from_text(
            text_content=text_content,
            max_steps=args.max_episode_length,
            precision=32,
        )

    print(f"Loaded environment: {args.env_type}")
    print(f"  Grid size: {env.width} x {env.height}")
    print(f"  Number of obstacles: {len(env.obstacles) if env.has_obstacles else 0}")

    return env


def compute_complex_inner_product(
    x_real: jnp.ndarray,
    x_imag: jnp.ndarray,
    y_real: jnp.ndarray,
    y_imag: jnp.ndarray,
    D: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute complex inner product <x̄, y>_ρ = x^T D [[y]] where [[y]] is conjugate of y.

    In the theory: h_{x,jk} = x_j^T D [[y_k]] - δ_{jk}
    So we need <x_j, [[y_k]]>_D = (x_j)^H D y_k* = conj(x_j)^T D conj(y_k)

    Actually from PDF: h_{x,jk} = <x̄_j, [[y_k]]>_ρ = x_j^T D [[y_k]]
    where x̄ denotes conjugate of x.

    Returns: (real_part, imag_part) of the inner product
    """
    # <x̄, [[y]]> = conj(x)^T D conj(y) = (x_real - i*x_imag)^T D (y_real - i*y_imag)
    # = (x_real^T D y_real + x_imag^T D y_imag) + i*(-x_real^T D y_imag + x_imag^T D y_real)

    # Apply D weighting
    Dy_real = D @ y_real
    Dy_imag = D @ y_imag

    # Real part: x_real^T D y_real + x_imag^T D y_imag
    real_part = x_real @ Dy_real + x_imag @ Dy_imag

    # Imag part: -x_real^T D y_imag + x_imag^T D y_real
    imag_part = -x_real @ Dy_imag + x_imag @ Dy_real

    return real_part, imag_part


def compute_Ly_product(
    L: jnp.ndarray,
    y_real: jnp.ndarray,
    y_imag: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute L @ y for complex y."""
    Ly_real = L @ y_real
    Ly_imag = L @ y_imag
    return Ly_real, Ly_imag


def compute_exact_gradients(
    X_real: jnp.ndarray,  # (n_states, d) - left eigenvector real parts
    X_imag: jnp.ndarray,  # (n_states, d) - left eigenvector imag parts
    Y_real: jnp.ndarray,  # (n_states, d) - right eigenvector real parts
    Y_imag: jnp.ndarray,  # (n_states, d) - right eigenvector imag parts
    alpha_x: jnp.ndarray,  # (d, d) lower triangular - duals for Re(h_x)
    alpha_y: jnp.ndarray,  # (d, d) lower triangular - duals for Re(h_y)
    beta_x: jnp.ndarray,   # (d, d) lower triangular - duals for Im(h_x)
    beta_y: jnp.ndarray,   # (d, d) lower triangular - duals for Im(h_y)
    L: jnp.ndarray,        # (n_states, n_states) - Laplacian matrix
    D: jnp.ndarray,        # (n_states,) - sampling distribution (diagonal)
    b: float,              # barrier coefficient
) -> Dict[str, jnp.ndarray]:
    """
    Compute exact gradients according to the theory.

    From PDF Section 1.2:
    g_{x_i} = Re(<x̄_i, Ly_i>_ρ) * D*L*ȳ_i - Σ_{k=1}^{i} (α_{x,ik} + iβ_{x,ik} - b*h_{x,ik}) * D*ȳ_k
    g_{y_i} = Re(<x̄_i, Ly_i>_ρ) * L^T*D*x̄_i - Σ_{k=1}^{i} (α_{y,ik} + iβ_{y,ik} - b*h_{y,ik}) * D*x̄_k

    g_{α_{x,jk}} = Re(h_{x,jk})
    g_{α_{y,jk}} = Re(h_{y,jk})
    g_{β_{x,jk}} = Im(h_{x,jk})
    g_{β_{y,jk}} = Im(h_{y,jk})
    """
    n_states, d = X_real.shape
    D_diag = jnp.diag(D)  # Convert to diagonal matrix

    # Initialize gradient arrays
    g_X_real = jnp.zeros_like(X_real)
    g_X_imag = jnp.zeros_like(X_imag)
    g_Y_real = jnp.zeros_like(Y_real)
    g_Y_imag = jnp.zeros_like(Y_imag)
    g_alpha_x = jnp.zeros_like(alpha_x)
    g_alpha_y = jnp.zeros_like(alpha_y)
    g_beta_x = jnp.zeros_like(beta_x)
    g_beta_y = jnp.zeros_like(beta_y)

    # Precompute L @ Y and L^T @ X for all eigenvectors
    LY_real = L @ Y_real  # (n_states, d)
    LY_imag = L @ Y_imag
    LTX_real = L.T @ X_real  # (n_states, d)
    LTX_imag = L.T @ X_imag

    # Precompute D @ Y and D @ X (conjugates)
    # ȳ = y_real - i*y_imag, x̄ = x_real - i*x_imag
    DY_real = D_diag @ Y_real  # D * y_real
    DY_imag = D_diag @ Y_imag  # D * y_imag
    DX_real = D_diag @ X_real
    DX_imag = D_diag @ X_imag

    # Compute all constraint errors h_{x,jk} and h_{y,jk}
    # h_{x,jk} = x_j^T D [[y_k]] - δ_{jk} = <x̄_j, [[y_k]]>_ρ - δ_{jk}
    # where [[y]] = y_real - i*y_imag (conjugate)

    # For h_x: need x_j^T D conj(y_k)
    # Real: x_real^T D y_real + x_imag^T D y_imag
    # Imag: -x_real^T D y_imag + x_imag^T D y_real
    h_x_real = X_real.T @ DY_real + X_imag.T @ DY_imag  # (d, d)
    h_x_imag = -X_real.T @ DY_imag + X_imag.T @ DY_real  # (d, d)
    h_x_real = h_x_real - jnp.eye(d)  # Subtract identity for constraint

    # For h_y: need conj(x_k)^T D y_j = [[x_k]]^T D y_j
    # Real: x_real^T D y_real + x_imag^T D y_imag
    # Imag: x_real^T D y_imag - x_imag^T D y_real
    h_y_real = X_real.T @ DY_real + X_imag.T @ DY_imag  # (d, d)
    h_y_imag = X_real.T @ DY_imag - X_imag.T @ DY_real  # (d, d)
    h_y_real = h_y_real - jnp.eye(d)

    # Apply lower triangular mask (we only constrain j >= k)
    tril_mask = jnp.tril(jnp.ones((d, d)))
    h_x_real = h_x_real * tril_mask
    h_x_imag = h_x_imag * tril_mask
    h_y_real = h_y_real * tril_mask
    h_y_imag = h_y_imag * tril_mask

    # Dual gradients are just the constraint errors
    g_alpha_x = h_x_real
    g_alpha_y = h_y_real
    g_beta_x = h_x_imag
    g_beta_y = h_y_imag

    # Compute gradients for each eigenvector
    for i in range(d):
        x_i_real = X_real[:, i]
        x_i_imag = X_imag[:, i]
        y_i_real = Y_real[:, i]
        y_i_imag = Y_imag[:, i]

        # Compute Re(<x̄_i, Ly_i>_ρ)
        # <x̄_i, Ly_i> = conj(x_i)^T D (L @ y_i)
        Ly_i_real = LY_real[:, i]
        Ly_i_imag = LY_imag[:, i]

        # Apply D
        D_Ly_i_real = D * Ly_i_real
        D_Ly_i_imag = D * Ly_i_imag

        # Inner product with conjugate of x_i
        # Real part: x_real^T D Ly_real + x_imag^T D Ly_imag
        spectral_product_real = x_i_real @ D_Ly_i_real + x_i_imag @ D_Ly_i_imag

        # First term of gradient for x_i: Re(<x̄_i, Ly_i>_ρ) * D*L*ȳ_i
        # ȳ_i = y_i_real - i*y_i_imag
        # D*L*ȳ_i = D @ L @ ȳ_i
        DL = D_diag @ L
        DL_ybar_i_real = DL @ y_i_real   # Real part of D*L*ȳ_i
        DL_ybar_i_imag = -DL @ y_i_imag  # Imag part of D*L*ȳ_i (conjugate)

        g_xi_real = spectral_product_real * DL_ybar_i_real
        g_xi_imag = spectral_product_real * DL_ybar_i_imag

        # First term of gradient for y_i: Re(<x̄_i, Ly_i>_ρ) * L^T*D*x̄_i
        # x̄_i = x_i_real - i*x_i_imag
        LTD = L.T @ D_diag
        LTD_xbar_i_real = LTD @ x_i_real   # Real part
        LTD_xbar_i_imag = -LTD @ x_i_imag  # Imag part (conjugate)

        g_yi_real = spectral_product_real * LTD_xbar_i_real
        g_yi_imag = spectral_product_real * LTD_xbar_i_imag

        # Second term: sum over k from 1 to i
        # For x_i: -Σ_{k=1}^{i} (α_{x,ik} + iβ_{x,ik} - b*h_{x,ik}) * D*ȳ_k
        for k in range(i + 1):
            # Complex coefficient: α_{x,ik} + iβ_{x,ik} - b*h_{x,ik}
            # h_{x,ik} = h_x_real[i,k] + i*h_x_imag[i,k]
            coef_real = alpha_x[i, k] - b * h_x_real[i, k]
            coef_imag = beta_x[i, k] - b * h_x_imag[i, k]

            # D*ȳ_k = D @ (y_k_real - i*y_k_imag)
            D_ybar_k_real = DY_real[:, k]
            D_ybar_k_imag = -DY_imag[:, k]  # Conjugate

            # Complex multiplication: (coef_real + i*coef_imag) * (D_ybar_k_real + i*D_ybar_k_imag)
            term_real = coef_real * D_ybar_k_real - coef_imag * D_ybar_k_imag
            term_imag = coef_real * D_ybar_k_imag + coef_imag * D_ybar_k_real

            g_xi_real = g_xi_real - term_real
            g_xi_imag = g_xi_imag - term_imag

        # For y_i: -Σ_{k=1}^{i} (α_{y,ik} + iβ_{y,ik} - b*h_{y,ik}) * D*x̄_k
        for k in range(i + 1):
            coef_real = alpha_y[i, k] - b * h_y_real[i, k]
            coef_imag = beta_y[i, k] - b * h_y_imag[i, k]

            D_xbar_k_real = DX_real[:, k]
            D_xbar_k_imag = -DX_imag[:, k]  # Conjugate

            term_real = coef_real * D_xbar_k_real - coef_imag * D_xbar_k_imag
            term_imag = coef_real * D_xbar_k_imag + coef_imag * D_xbar_k_real

            g_yi_real = g_yi_real - term_real
            g_yi_imag = g_yi_imag - term_imag

        g_X_real = g_X_real.at[:, i].set(g_xi_real)
        g_X_imag = g_X_imag.at[:, i].set(g_xi_imag)
        g_Y_real = g_Y_real.at[:, i].set(g_yi_real)
        g_Y_imag = g_Y_imag.at[:, i].set(g_yi_imag)

    return {
        'g_X_real': g_X_real,
        'g_X_imag': g_X_imag,
        'g_Y_real': g_Y_real,
        'g_Y_imag': g_Y_imag,
        'g_alpha_x': g_alpha_x,
        'g_alpha_y': g_alpha_y,
        'g_beta_x': g_beta_x,
        'g_beta_y': g_beta_y,
        'h_x_real': h_x_real,
        'h_x_imag': h_x_imag,
        'h_y_real': h_y_real,
        'h_y_imag': h_y_imag,
    }


def compute_cosine_similarity(
    learned_real: jnp.ndarray,
    learned_imag: jnp.ndarray,
    gt_real: jnp.ndarray,
    gt_imag: jnp.ndarray,
    D: jnp.ndarray = None,
) -> float:
    """Compute cosine similarity between learned and ground truth complex eigenvectors."""
    # Normalize learned eigenvector by largest magnitude component
    magnitudes = jnp.sqrt(learned_real**2 + learned_imag**2)
    max_idx = jnp.argmax(magnitudes)
    scale = learned_real[max_idx] + 1j * learned_imag[max_idx]
    scale_norm = jnp.abs(scale)

    # Divide by scale to normalize phase
    learned_normalized = (learned_real + 1j * learned_imag) / (scale + 1e-10)
    learned_norm_real = jnp.real(learned_normalized)
    learned_norm_imag = jnp.imag(learned_normalized)

    # Apply weighting if provided
    if D is not None:
        sqrt_D = jnp.sqrt(D)
        learned_norm_real = learned_norm_real * sqrt_D
        learned_norm_imag = learned_norm_imag * sqrt_D
        gt_real = gt_real * sqrt_D
        gt_imag = gt_imag * sqrt_D

    # Complex inner product magnitude
    inner_real = jnp.dot(learned_norm_real, gt_real) + jnp.dot(learned_norm_imag, gt_imag)
    inner_imag = jnp.dot(learned_norm_real, gt_imag) - jnp.dot(learned_norm_imag, gt_real)

    learned_norm = jnp.sqrt(jnp.dot(learned_norm_real, learned_norm_real) +
                           jnp.dot(learned_norm_imag, learned_norm_imag))
    gt_norm = jnp.sqrt(jnp.dot(gt_real, gt_real) + jnp.dot(gt_imag, gt_imag))

    cos_sim = jnp.abs(inner_real) / (learned_norm * gt_norm + 1e-10)
    return float(cos_sim)


def main(args: Args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Create environment
    env = create_gridworld_env(args)

    # Get canonical (free) states from base environment
    canonical_states = get_canonical_free_states(env)
    num_canonical = len(canonical_states)
    num_states = env.width * env.height
    print(f"Number of free states: {num_canonical} (out of {num_states} total)")

    # Add doors if requested
    data_env = env
    door_config = None

    if args.use_doors:
        door_rng = np.random.default_rng(args.door_seed)
        door_config = create_random_doors(
            env=env,
            num_doors=args.num_doors,
            rng=door_rng,
        )
        data_env = create_door_gridworld_from_base(env, door_config)
        print(f"\nCreated {len(door_config['doors'])} irreversible doors")

    # Collect transition data
    print("\nCollecting transition data...")
    transition_counts_full, episodes, metrics = collect_transition_counts_and_episodes(
        env=data_env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_states=num_states,
        seed=args.seed,
    )
    print(f"Collected {metrics['total_transitions']} transitions.")

    # Extract canonical state subspace
    transition_counts = transition_counts_full[jnp.ix_(canonical_states, jnp.arange(env.action_space), canonical_states)]

    # Build transition matrix and Laplacian
    print("\nBuilding transition matrix and Laplacian...")
    transition_matrix = get_transition_matrix(transition_counts)

    # Compute sampling distribution
    sampling_probs = compute_sampling_distribution(transition_counts)
    D = sampling_probs  # Diagonal of sampling distribution matrix

    print(f"  Sampling prob range: [{D.min():.6f}, {D.max():.6f}]")

    # Compute Laplacian
    print(f"Computing non-symmetric Laplacian with gamma={args.gamma}, delta={args.delta}...")
    L = compute_nonsymmetric_laplacian(transition_matrix, gamma=args.gamma, delta=args.delta)

    # Compute ground truth eigendecomposition
    eigendecomp = compute_eigendecomposition(L, k=args.num_eigenvectors, ascending=True)

    gt_eigenvalues_real = eigendecomp['eigenvalues_real']
    gt_eigenvalues_imag = eigendecomp['eigenvalues_imag']
    gt_left_real = eigendecomp['left_eigenvectors_real']
    gt_left_imag = eigendecomp['left_eigenvectors_imag']
    gt_right_real = eigendecomp['right_eigenvectors_real']
    gt_right_imag = eigendecomp['right_eigenvectors_imag']

    print(f"\nGround truth eigenvalues:")
    for i in range(args.num_eigenvectors):
        print(f"  λ_{i}: {gt_eigenvalues_real[i]:.6f} + {gt_eigenvalues_imag[i]:.6f}i")

    # Setup results directory
    if args.exp_name is None:
        exp_name = f"{args.env_type}__allo_exact__{args.exp_number}__{args.seed}__{int(time.time())}"
    else:
        exp_name = args.exp_name

    results_dir = Path(args.results_dir) / args.env_type / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"\nResults directory: {results_dir}")

    # Save args
    with open(results_dir / "args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Initialize parameters
    print(f"\nInitializing parameters for {args.num_eigenvectors} eigenvector(s)...")
    d = args.num_eigenvectors
    n = num_canonical

    init_scale = 0.1
    key, *subkeys = jax.random.split(key, 5)

    # Eigenvector matrices (n_states, d)
    X_real = init_scale * jax.random.normal(subkeys[0], (n, d))
    X_imag = init_scale * jax.random.normal(subkeys[1], (n, d))
    Y_real = init_scale * jax.random.normal(subkeys[2], (n, d))
    Y_imag = init_scale * jax.random.normal(subkeys[3], (n, d))

    # Dual variables (d, d) lower triangular
    alpha_x = jnp.zeros((d, d))
    alpha_y = jnp.zeros((d, d))
    beta_x = jnp.zeros((d, d))
    beta_y = jnp.zeros((d, d))

    print(f"  X, Y shape: ({n}, {d})")
    print(f"  Dual shape: ({d}, {d})")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Barrier coefficient b: {args.barrier_coef}")

    # Training loop
    print("\nStarting exact gradient descent...")
    metrics_history = []
    start_time = time.time()

    for step in tqdm(range(args.num_gradient_steps)):
        # Compute exact gradients
        grads = compute_exact_gradients(
            X_real, X_imag, Y_real, Y_imag,
            alpha_x, alpha_y, beta_x, beta_y,
            L, D, args.barrier_coef
        )

        # Update primal variables (gradient descent: minimize)
        X_real = X_real - args.learning_rate * grads['g_X_real']
        X_imag = X_imag - args.learning_rate * grads['g_X_imag']
        Y_real = Y_real - args.learning_rate * grads['g_Y_real']
        Y_imag = Y_imag - args.learning_rate * grads['g_Y_imag']

        # Update dual variables (gradient ascent: maximize)
        # g_Θ has negative sign for duals in the combined system
        alpha_x = alpha_x + args.learning_rate * grads['g_alpha_x']
        alpha_y = alpha_y + args.learning_rate * grads['g_alpha_y']
        beta_x = beta_x + args.learning_rate * grads['g_beta_x']
        beta_y = beta_y + args.learning_rate * grads['g_beta_y']

        # Apply lower triangular mask to duals
        tril_mask = jnp.tril(jnp.ones((d, d)))
        alpha_x = alpha_x * tril_mask
        alpha_y = alpha_y * tril_mask
        beta_x = beta_x * tril_mask
        beta_y = beta_y * tril_mask

        # Log metrics
        if step % args.log_freq == 0:
            # Compute constraint errors
            h_x_error = jnp.abs(grads['h_x_real']).sum() + jnp.abs(grads['h_x_imag']).sum()
            h_y_error = jnp.abs(grads['h_y_real']).sum() + jnp.abs(grads['h_y_imag']).sum()
            total_constraint_error = h_x_error + h_y_error

            # Compute gradient norms
            grad_X_norm = jnp.sqrt((grads['g_X_real']**2).sum() + (grads['g_X_imag']**2).sum())
            grad_Y_norm = jnp.sqrt((grads['g_Y_real']**2).sum() + (grads['g_Y_imag']**2).sum())

            # Compute cosine similarities
            left_sims = []
            right_sims = []
            for i in range(d):
                left_sim = compute_cosine_similarity(
                    X_real[:, i], X_imag[:, i],
                    gt_left_real[:, i], gt_left_imag[:, i],
                    D
                )
                right_sim = compute_cosine_similarity(
                    Y_real[:, i], Y_imag[:, i],
                    gt_right_real[:, i], gt_right_imag[:, i],
                    None
                )
                left_sims.append(left_sim)
                right_sims.append(right_sim)

            avg_left_sim = np.mean(left_sims)
            avg_right_sim = np.mean(right_sims)

            # Eigenvalue estimates from diagonal duals
            # At equilibrium: α_{x,ii} + iβ_{x,ii} ≈ eigenvalue
            eigenvalue_est_real = -0.5 * (jnp.diag(alpha_x) + jnp.diag(alpha_y))
            eigenvalue_est_imag = -0.5 * (jnp.diag(beta_x) + jnp.diag(beta_y))

            metrics_dict = {
                'step': step,
                'constraint_error': float(total_constraint_error),
                'h_x_error': float(h_x_error),
                'h_y_error': float(h_y_error),
                'grad_X_norm': float(grad_X_norm),
                'grad_Y_norm': float(grad_Y_norm),
                'left_cosine_sim': avg_left_sim,
                'right_cosine_sim': avg_right_sim,
            }

            for i in range(d):
                metrics_dict[f'eigenvalue_est_real_{i}'] = float(eigenvalue_est_real[i])
                metrics_dict[f'eigenvalue_est_imag_{i}'] = float(eigenvalue_est_imag[i])
                metrics_dict[f'left_sim_{i}'] = left_sims[i]
                metrics_dict[f'right_sim_{i}'] = right_sims[i]

            metrics_history.append(metrics_dict)

            tqdm.write(
                f"Step {step}: constraint_err={total_constraint_error:.6f}, "
                f"grad_X={grad_X_norm:.4f}, grad_Y={grad_Y_norm:.4f}, "
                f"left_sim={avg_left_sim:.4f}, right_sim={avg_right_sim:.4f}"
            )

            if d == 1:
                tqdm.write(
                    f"  λ_est: {eigenvalue_est_real[0]:.4f} + {eigenvalue_est_imag[0]:.4f}i, "
                    f"λ_gt: {gt_eigenvalues_real[0]:.4f} + {gt_eigenvalues_imag[0]:.4f}i"
                )

        # Plot during training
        if args.plot_during_training and step % args.plot_freq == 0 and step > 0:
            # Plot constraint error evolution
            if len(metrics_history) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                steps = [m['step'] for m in metrics_history]

                # Constraint error
                ax = axes[0, 0]
                ax.semilogy(steps, [m['constraint_error'] for m in metrics_history])
                ax.set_xlabel('Step')
                ax.set_ylabel('Constraint Error')
                ax.set_title('Biorthogonality Constraint Error')
                ax.grid(True)

                # Gradient norms
                ax = axes[0, 1]
                ax.semilogy(steps, [m['grad_X_norm'] for m in metrics_history], label='grad_X')
                ax.semilogy(steps, [m['grad_Y_norm'] for m in metrics_history], label='grad_Y')
                ax.set_xlabel('Step')
                ax.set_ylabel('Gradient Norm')
                ax.set_title('Gradient Norms')
                ax.legend()
                ax.grid(True)

                # Cosine similarities
                ax = axes[1, 0]
                ax.plot(steps, [m['left_cosine_sim'] for m in metrics_history], label='Left')
                ax.plot(steps, [m['right_cosine_sim'] for m in metrics_history], label='Right')
                ax.set_xlabel('Step')
                ax.set_ylabel('Cosine Similarity')
                ax.set_title('Eigenvector Similarity to Ground Truth')
                ax.legend()
                ax.grid(True)
                ax.set_ylim([0, 1])

                # Eigenvalue estimates
                ax = axes[1, 1]
                for i in range(d):
                    est_real = [m[f'eigenvalue_est_real_{i}'] for m in metrics_history]
                    ax.plot(steps, est_real, label=f'λ_{i} est')
                    ax.axhline(y=gt_eigenvalues_real[i], linestyle='--', alpha=0.5)
                ax.set_xlabel('Step')
                ax.set_ylabel('Eigenvalue (Real)')
                ax.set_title('Eigenvalue Estimates')
                ax.legend()
                ax.grid(True)

                plt.tight_layout()
                plt.savefig(plots_dir / f"training_step_{step}.png", dpi=150)
                plt.close()

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.2f}s")

    # Save final results
    if args.save_model:
        np.save(results_dir / "final_X_real.npy", np.array(X_real))
        np.save(results_dir / "final_X_imag.npy", np.array(X_imag))
        np.save(results_dir / "final_Y_real.npy", np.array(Y_real))
        np.save(results_dir / "final_Y_imag.npy", np.array(Y_imag))
        np.save(results_dir / "alpha_x.npy", np.array(alpha_x))
        np.save(results_dir / "alpha_y.npy", np.array(alpha_y))
        np.save(results_dir / "beta_x.npy", np.array(beta_x))
        np.save(results_dir / "beta_y.npy", np.array(beta_y))
        np.save(results_dir / "gt_eigenvalues_real.npy", np.array(gt_eigenvalues_real))
        np.save(results_dir / "gt_eigenvalues_imag.npy", np.array(gt_eigenvalues_imag))
        np.save(results_dir / "gt_left_real.npy", np.array(gt_left_real))
        np.save(results_dir / "gt_left_imag.npy", np.array(gt_left_imag))
        np.save(results_dir / "gt_right_real.npy", np.array(gt_right_real))
        np.save(results_dir / "gt_right_imag.npy", np.array(gt_right_imag))

        with open(results_dir / "metrics_history.json", 'w') as f:
            json.dump(metrics_history, f, indent=2)

    # Final visualization
    if args.plot_during_training:
        # Create final training curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        steps = [m['step'] for m in metrics_history]

        ax = axes[0, 0]
        ax.semilogy(steps, [m['constraint_error'] for m in metrics_history])
        ax.set_xlabel('Step')
        ax.set_ylabel('Constraint Error')
        ax.set_title('Biorthogonality Constraint Error')
        ax.grid(True)

        ax = axes[0, 1]
        ax.semilogy(steps, [m['grad_X_norm'] for m in metrics_history], label='grad_X')
        ax.semilogy(steps, [m['grad_Y_norm'] for m in metrics_history], label='grad_Y')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Norms')
        ax.legend()
        ax.grid(True)

        ax = axes[1, 0]
        ax.plot(steps, [m['left_cosine_sim'] for m in metrics_history], label='Left')
        ax.plot(steps, [m['right_cosine_sim'] for m in metrics_history], label='Right')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Eigenvector Similarity to Ground Truth')
        ax.legend()
        ax.grid(True)
        ax.set_ylim([0, 1])

        ax = axes[1, 1]
        for i in range(d):
            est_real = [m[f'eigenvalue_est_real_{i}'] for m in metrics_history]
            ax.plot(steps, est_real, label=f'λ_{i} est')
            ax.axhline(y=gt_eigenvalues_real[i], linestyle='--', alpha=0.5,
                      label=f'λ_{i} gt' if i == 0 else None)
        ax.set_xlabel('Step')
        ax.set_ylabel('Eigenvalue (Real)')
        ax.set_title('Eigenvalue Estimates')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / "final_training_curves.png", dpi=150)
        plt.close()

    print(f"\nFinal Results:")
    print(f"  Constraint error: {metrics_history[-1]['constraint_error']:.6f}")
    print(f"  Left cosine sim: {metrics_history[-1]['left_cosine_sim']:.4f}")
    print(f"  Right cosine sim: {metrics_history[-1]['right_cosine_sim']:.4f}")

    for i in range(d):
        print(f"  Eigenvector {i}:")
        print(f"    λ_est: {metrics_history[-1][f'eigenvalue_est_real_{i}']:.4f} + {metrics_history[-1][f'eigenvalue_est_imag_{i}']:.4f}i")
        print(f"    λ_gt:  {gt_eigenvalues_real[i]:.4f} + {gt_eigenvalues_imag[i]:.4f}i")

    return results_dir


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
