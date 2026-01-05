"""
Test script to verify that the sampling distribution from the replay buffer
correctly reconstructs the transition matrix P_π * SR_γ.

This script:
1. Creates the same environment and replay buffer used in allo_complex.py
2. Samples a large number of transitions
3. Counts transition frequencies to reconstruct empirical transition matrix
4. Computes eigendecomposition of the corresponding Laplacian
5. Compares reconstructed eigenvectors with ground truth eigenvectors
"""

import sys
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import tyro

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from exp_alcl.allo_complex import (
    Args,
    create_gridworld_env,
    collect_data_and_compute_eigenvectors,
)
from exp_complex_basis.eigendecomposition import compute_eigendecomposition


@dataclass
class TestArgs:
    """Arguments for the sampling test."""
    # Number of samples to collect for estimating transition matrix
    num_samples: int = 1000000  # 1M samples should give good estimates

    # Environment and data collection parameters (inherit from allo_complex)
    env_type: str = "room4"  # 'room4', 'maze', 'spiral', 'obstacles', 'empty', or 'file'
    env_file: str | None = None  # Path to environment text file (if env_type='file')
    env_file_name: str | None = None  # Name of environment file in src/envs/txt/ (e.g., 'GridRoom-4')
    max_episode_length: int = 1000
    num_envs: int = 1000
    num_steps: int = 1000
    gamma: float = 0.2

    # Irreversible doors
    use_doors: bool = False
    num_doors: int = 5
    door_seed: int = 42

    # Number of eigenvectors to compare (None = all available)
    num_eigenvectors: int | None = None

    # Misc
    seed: int = 42
    batch_size: int = 256  # For sampling efficiency

    # Save results
    save_dir: str = "./sampling_test_results"


def test_sampling_distribution(test_args: TestArgs):
    """
    Test that sampling from replay buffer recovers the correct transition distribution.
    """
    print("="*80)
    print("TESTING SAMPLING DISTRIBUTION")
    print("="*80)

    # Create environment and collect data using the same process as allo_complex.py
    # Convert TestArgs to Args for compatibility
    # Determine how many eigenvectors to compute
    # If None specified, we'll compute ALL (set to a large number initially, then trim)
    if test_args.num_eigenvectors is None:
        # We'll determine the actual number after creating the environment
        num_eig_initial = None  # Signal to compute all
    else:
        num_eig_initial = test_args.num_eigenvectors

    args = Args(
        env_type=test_args.env_type,
        env_file=test_args.env_file,
        env_file_name=test_args.env_file_name,
        max_episode_length=test_args.max_episode_length,
        num_envs=test_args.num_envs,
        num_steps=test_args.num_steps,
        gamma=test_args.gamma,
        use_doors=test_args.use_doors,
        num_doors=test_args.num_doors,
        door_seed=test_args.door_seed,
        seed=test_args.seed,
        num_eigenvectors=10,  # Temporary, will update after we know num_states
    )

    print("\n1. Creating environment and collecting data...")
    env = create_gridworld_env(args)

    laplacian_matrix, eigendecomp_temp, state_coords, canonical_states, sampling_probs, door_config, data_env, replay_buffer = \
        collect_data_and_compute_eigenvectors(env, args)

    num_states = len(canonical_states)
    print(f"   Number of states: {num_states}")

    # Determine how many eigenvectors to compare
    num_eigenvectors_to_compare = test_args.num_eigenvectors if test_args.num_eigenvectors is not None else num_states
    num_eigenvectors_to_compare = min(num_eigenvectors_to_compare, num_states)

    print(f"   Comparing {num_eigenvectors_to_compare} eigenvectors (out of {num_states} total states)")

    # Recompute eigendecomposition with the correct number if needed
    if num_eigenvectors_to_compare > 10:
        print(f"   Recomputing ground truth eigendecomposition with {num_eigenvectors_to_compare} eigenvectors...")
        eigendecomp = compute_eigendecomposition(
            laplacian_matrix,
            k=num_eigenvectors_to_compare,
            sort_by_magnitude=True,
            ascending=True,
        )
    else:
        eigendecomp = eigendecomp_temp

    print(f"   Ground truth eigenvalues (first 10):")
    print(f"   Real: {eigendecomp['eigenvalues_real'][:10]}")
    print(f"   Imag: {eigendecomp['eigenvalues_imag'][:10]}")

    # Sample transitions from replay buffer
    print(f"\n2. Sampling {test_args.num_samples} transitions from replay buffer...")
    # Use JAX array for efficient vectorized counting
    transition_counts = jnp.zeros((num_states, num_states), dtype=jnp.int32)

    num_batches = (test_args.num_samples + test_args.batch_size - 1) // test_args.batch_size
    total_samples = 0

    for _ in tqdm(range(num_batches), desc="Sampling batches"):
        # Sample a batch of transitions
        batch = replay_buffer.sample(test_args.batch_size, discount=test_args.gamma)

        # Extract state indices
        state_indices = jnp.array(batch.obs)
        next_state_indices = jnp.array(batch.next_obs)

        # Vectorized counting (like in src/data_collection.py)
        transition_counts = transition_counts.at[state_indices, next_state_indices].add(1)
        total_samples += len(state_indices)

    # Convert back to numpy for remaining operations
    transition_counts = np.array(transition_counts)

    print(f"   Total samples collected: {total_samples}")
    print(f"   Transition matrix sparsity: {np.sum(transition_counts > 0) / (num_states * num_states):.4f}")

    # Normalize to get empirical transition matrix
    print("\n3. Reconstructing empirical transition matrix...")
    # Each row should sum to the number of times that state was visited as source
    row_sums = transition_counts.sum(axis=1, keepdims=True)

    # Avoid division by zero
    empirical_transition_matrix = np.zeros_like(transition_counts, dtype=np.float64)
    nonzero_rows = row_sums.flatten() > 0
    empirical_transition_matrix[nonzero_rows] = (
        transition_counts[nonzero_rows] / row_sums[nonzero_rows]
    )

    # Check if any states were never sampled as source
    unsampled_states = np.where(~nonzero_rows)[0]
    if len(unsampled_states) > 0:
        print(f"   WARNING: {len(unsampled_states)} states never sampled as source states")
        print(f"   Unsampled state indices: {unsampled_states[:10]}..." if len(unsampled_states) > 10 else f"   Unsampled state indices: {unsampled_states}")

    # Verify row sums (should be 1 or 0)
    row_sums_check = empirical_transition_matrix.sum(axis=1)
    print(f"   Row sums min: {row_sums_check.min():.6f}, max: {row_sums_check.max():.6f}")
    print(f"   Row sums (non-zero rows) should be ≈1.0")

    # Compute Laplacian from empirical transition matrix
    # IMPORTANT: The empirical transition matrix already contains the (1-γ) factor!
    # Sampling recovers (1-γ)*P*SR_γ, not P*SR_γ, due to the geometric distribution normalization
    # Therefore: L = I - empirical_matrix (NOT I - (1-γ)*empirical_matrix)
    print("\n4. Computing Laplacian from empirical transition matrix...")
    print("   NOTE: Empirical matrix represents (1-γ)*P*SR_γ (includes the (1-γ) factor)")
    gamma = test_args.gamma
    empirical_laplacian = np.eye(num_states) - empirical_transition_matrix

    # Compute eigendecomposition of empirical Laplacian
    print("\n5. Computing eigendecomposition of empirical Laplacian...")
    empirical_eigendecomp = compute_eigendecomposition(
        empirical_laplacian,
        k=num_eigenvectors_to_compare,
        sort_by_magnitude=True,
        ascending=True,  # Smallest eigenvalues first
    )

    print(f"   Empirical eigenvalues (first 10):")
    print(f"   Real: {empirical_eigendecomp['eigenvalues_real'][:10]}")
    print(f"   Imag: {empirical_eigendecomp['eigenvalues_imag'][:10]}")

    # Compare eigenvalues
    print("\n6. Comparing eigenvalues...")
    gt_eigenvalues_real = eigendecomp['eigenvalues_real'][:num_eigenvectors_to_compare]
    gt_eigenvalues_imag = eigendecomp['eigenvalues_imag'][:num_eigenvectors_to_compare]
    emp_eigenvalues_real = empirical_eigendecomp['eigenvalues_real'][:num_eigenvectors_to_compare]
    emp_eigenvalues_imag = empirical_eigendecomp['eigenvalues_imag'][:num_eigenvectors_to_compare]

    eigenvalue_error_real = np.abs(gt_eigenvalues_real - emp_eigenvalues_real)
    eigenvalue_error_imag = np.abs(gt_eigenvalues_imag - emp_eigenvalues_imag)
    eigenvalue_error_magnitude = np.sqrt(eigenvalue_error_real**2 + eigenvalue_error_imag**2)

    print(f"   Eigenvalue error (magnitude) for first 10:")
    for i in range(min(10, num_eigenvectors_to_compare)):
        print(f"      Eigenvector {i}: {eigenvalue_error_magnitude[i]:.6f}")
    print(f"   Mean eigenvalue error (all {num_eigenvectors_to_compare}): {eigenvalue_error_magnitude.mean():.6f}")
    print(f"   Max eigenvalue error: {eigenvalue_error_magnitude.max():.6f}")

    # Compare eigenvectors
    print("\n7. Comparing eigenvectors...")

    # Extract ground truth eigenvectors
    gt_left_real = eigendecomp['left_eigenvectors_real'][:, :num_eigenvectors_to_compare]
    gt_left_imag = eigendecomp['left_eigenvectors_imag'][:, :num_eigenvectors_to_compare]
    gt_right_real = eigendecomp['right_eigenvectors_real'][:, :num_eigenvectors_to_compare]
    gt_right_imag = eigendecomp['right_eigenvectors_imag'][:, :num_eigenvectors_to_compare]

    # Extract empirical eigenvectors
    emp_left_real = empirical_eigendecomp['left_eigenvectors_real'][:, :num_eigenvectors_to_compare]
    emp_left_imag = empirical_eigendecomp['left_eigenvectors_imag'][:, :num_eigenvectors_to_compare]
    emp_right_real = empirical_eigendecomp['right_eigenvectors_real'][:, :num_eigenvectors_to_compare]
    emp_right_imag = empirical_eigendecomp['right_eigenvectors_imag'][:, :num_eigenvectors_to_compare]

    # Compute cosine similarities for each eigenvector
    # For complex eigenvectors, we need to account for both real and imaginary parts
    print(f"\n   Right eigenvector similarities (first 10 of {num_eigenvectors_to_compare}):")
    for i in range(min(10, num_eigenvectors_to_compare)):
        # Flatten to vectors
        gt_real_i = gt_right_real[:, i]
        gt_imag_i = gt_right_imag[:, i]
        emp_real_i = emp_right_real[:, i]
        emp_imag_i = emp_right_imag[:, i]

        # Compute cosine similarity for real and imaginary parts separately
        # (accounting for potential sign/phase differences)
        cos_sim_real = np.abs(np.dot(gt_real_i, emp_real_i)) / (
            np.linalg.norm(gt_real_i) * np.linalg.norm(emp_real_i) + 1e-10
        )
        cos_sim_imag = np.abs(np.dot(gt_imag_i, emp_imag_i)) / (
            np.linalg.norm(gt_imag_i) * np.linalg.norm(emp_imag_i) + 1e-10
        ) if np.linalg.norm(gt_imag_i) > 1e-10 else 1.0

        # L2 error
        error_real = np.linalg.norm(gt_real_i - emp_real_i)
        error_imag = np.linalg.norm(gt_imag_i - emp_imag_i)

        # Also try with sign flip
        error_real_flipped = np.linalg.norm(gt_real_i + emp_real_i)
        error_imag_flipped = np.linalg.norm(gt_imag_i + emp_imag_i)

        print(f"      Eigenvector {i}:")
        print(f"         Real part - cos_sim: {cos_sim_real:.6f}, L2 error: {error_real:.6f}, L2 error (flipped): {error_real_flipped:.6f}")
        if np.linalg.norm(gt_imag_i) > 1e-10:
            print(f"         Imag part - cos_sim: {cos_sim_imag:.6f}, L2 error: {error_imag:.6f}, L2 error (flipped): {error_imag_flipped:.6f}")

    print(f"\n   Left eigenvector similarities (first 10 of {num_eigenvectors_to_compare}):")
    for i in range(min(10, num_eigenvectors_to_compare)):
        # Flatten to vectors
        gt_real_i = gt_left_real[:, i]
        gt_imag_i = gt_left_imag[:, i]
        emp_real_i = emp_left_real[:, i]
        emp_imag_i = emp_left_imag[:, i]

        # Compute cosine similarity
        cos_sim_real = np.abs(np.dot(gt_real_i, emp_real_i)) / (
            np.linalg.norm(gt_real_i) * np.linalg.norm(emp_real_i) + 1e-10
        )
        cos_sim_imag = np.abs(np.dot(gt_imag_i, emp_imag_i)) / (
            np.linalg.norm(gt_imag_i) * np.linalg.norm(emp_imag_i) + 1e-10
        ) if np.linalg.norm(gt_imag_i) > 1e-10 else 1.0

        # L2 error
        error_real = np.linalg.norm(gt_real_i - emp_real_i)
        error_imag = np.linalg.norm(gt_imag_i - emp_imag_i)

        # Also try with sign flip
        error_real_flipped = np.linalg.norm(gt_real_i + emp_real_i)
        error_imag_flipped = np.linalg.norm(gt_imag_i + emp_imag_i)

        print(f"      Eigenvector {i}:")
        print(f"         Real part - cos_sim: {cos_sim_real:.6f}, L2 error: {error_real:.6f}, L2 error (flipped): {error_real_flipped:.6f}")
        if np.linalg.norm(gt_imag_i) > 1e-10:
            print(f"         Imag part - cos_sim: {cos_sim_imag:.6f}, L2 error: {error_imag:.6f}, L2 error (flipped): {error_imag_flipped:.6f}")

    # Compute overall Frobenius norm for the eigenvector matrices
    print(f"\n   Frobenius norm comparisons:")
    print(f"   Context: Frobenius norm measures total element-wise difference across the matrix")
    print(f"            For a {num_states}x{num_eigenvectors_to_compare} matrix, random vectors would have ||diff|| ≈ {np.sqrt(num_states * num_eigenvectors_to_compare):.1f}")
    frob_right_real = np.linalg.norm(gt_right_real - emp_right_real, 'fro')
    frob_right_real_flipped = np.linalg.norm(gt_right_real + emp_right_real, 'fro')
    frob_left_real = np.linalg.norm(gt_left_real - emp_left_real, 'fro')
    frob_left_real_flipped = np.linalg.norm(gt_left_real + emp_left_real, 'fro')
    print(f"   Right eigenvectors (real): {frob_right_real:.6f} (flipped: {frob_right_real_flipped:.6f})")
    print(f"   Left eigenvectors (real):  {frob_left_real:.6f} (flipped: {frob_left_real_flipped:.6f})")

    # Save results
    print(f"\n8. Saving results to {test_args.save_dir}...")
    save_dir = Path(test_args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save matrices
    np.save(save_dir / "transition_counts.npy", transition_counts)
    np.save(save_dir / "empirical_transition_matrix.npy", empirical_transition_matrix)
    np.save(save_dir / "empirical_laplacian.npy", empirical_laplacian)
    np.save(save_dir / "ground_truth_laplacian.npy", laplacian_matrix)

    # Save eigendecompositions
    np.save(save_dir / "gt_eigenvalues_real.npy", gt_eigenvalues_real)
    np.save(save_dir / "gt_eigenvalues_imag.npy", gt_eigenvalues_imag)
    np.save(save_dir / "emp_eigenvalues_real.npy", emp_eigenvalues_real)
    np.save(save_dir / "emp_eigenvalues_imag.npy", emp_eigenvalues_imag)

    np.save(save_dir / "gt_left_real.npy", gt_left_real)
    np.save(save_dir / "gt_left_imag.npy", gt_left_imag)
    np.save(save_dir / "gt_right_real.npy", gt_right_real)
    np.save(save_dir / "gt_right_imag.npy", gt_right_imag)

    np.save(save_dir / "emp_left_real.npy", emp_left_real)
    np.save(save_dir / "emp_left_imag.npy", emp_left_imag)
    np.save(save_dir / "emp_right_real.npy", emp_right_real)
    np.save(save_dir / "emp_right_imag.npy", emp_right_imag)

    # Create visualizations
    print("\n9. Creating visualizations...")

    # Plot eigenvalue comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Real parts
    axes[0].scatter(gt_eigenvalues_real, emp_eigenvalues_real, alpha=0.6)
    axes[0].plot([gt_eigenvalues_real.min(), gt_eigenvalues_real.max()],
                 [gt_eigenvalues_real.min(), gt_eigenvalues_real.max()],
                 'r--', alpha=0.5, label='y=x')
    axes[0].set_xlabel('Ground Truth Eigenvalues (Real)')
    axes[0].set_ylabel('Empirical Eigenvalues (Real)')
    axes[0].set_title('Eigenvalue Comparison - Real Parts')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Imaginary parts
    axes[1].scatter(gt_eigenvalues_imag, emp_eigenvalues_imag, alpha=0.6)
    max_val = max(abs(gt_eigenvalues_imag).max(), abs(emp_eigenvalues_imag).max())
    axes[1].plot([-max_val, max_val], [-max_val, max_val], 'r--', alpha=0.5, label='y=x')
    axes[1].set_xlabel('Ground Truth Eigenvalues (Imag)')
    axes[1].set_ylabel('Empirical Eigenvalues (Imag)')
    axes[1].set_title('Eigenvalue Comparison - Imaginary Parts')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "eigenvalue_comparison.png", dpi=150, bbox_inches='tight')
    print(f"   Saved eigenvalue comparison to {save_dir / 'eigenvalue_comparison.png'}")

    # Plot transition matrix heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Ground truth
    gt_transition_matrix = np.eye(num_states) - laplacian_matrix
    im0 = axes[0].imshow(gt_transition_matrix, cmap='viridis', aspect='auto')
    axes[0].set_title('Ground Truth Transition Matrix')
    axes[0].set_xlabel('Next State')
    axes[0].set_ylabel('Current State')
    plt.colorbar(im0, ax=axes[0])

    # Empirical
    im1 = axes[1].imshow(empirical_transition_matrix, cmap='viridis', aspect='auto')
    axes[1].set_title('Empirical Transition Matrix')
    axes[1].set_xlabel('Next State')
    axes[1].set_ylabel('Current State')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_dir / "transition_matrix_comparison.png", dpi=150, bbox_inches='tight')
    print(f"   Saved transition matrix comparison to {save_dir / 'transition_matrix_comparison.png'}")

    # Plot spatial visualizations of eigenvectors on the grid
    print("\n   Creating spatial visualizations of eigenvectors...")
    num_to_visualize = min(6, num_eigenvectors_to_compare)  # Visualize first 6 eigenvectors

    # Create grid positions from state_coords
    grid_height = env.height
    grid_width = env.width

    for evec_idx in range(num_to_visualize):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Eigenvector {evec_idx} Spatial Visualization (λ ≈ {gt_eigenvalues_real[evec_idx]:.6f})', fontsize=14)

        # Create grids for visualization
        gt_right_grid = np.full((grid_height, grid_width), np.nan)
        emp_right_grid = np.full((grid_height, grid_width), np.nan)
        gt_left_grid = np.full((grid_height, grid_width), np.nan)
        emp_left_grid = np.full((grid_height, grid_width), np.nan)

        # Fill in the values for each canonical state
        for state_idx, canon_state in enumerate(canonical_states):
            y = canon_state // grid_width
            x = canon_state % grid_width
            gt_right_grid[y, x] = gt_right_real[state_idx, evec_idx]
            emp_right_grid[y, x] = emp_right_real[state_idx, evec_idx]
            gt_left_grid[y, x] = gt_left_real[state_idx, evec_idx]
            emp_left_grid[y, x] = emp_left_real[state_idx, evec_idx]

        # Determine color scale (use same scale for GT and empirical for comparison)
        vmin_right = min(np.nanmin(gt_right_grid), np.nanmin(emp_right_grid))
        vmax_right = max(np.nanmax(gt_right_grid), np.nanmax(emp_right_grid))
        vmin_left = min(np.nanmin(gt_left_grid), np.nanmin(emp_left_grid))
        vmax_left = max(np.nanmax(gt_left_grid), np.nanmax(emp_left_grid))

        # Plot ground truth right eigenvector
        im0 = axes[0, 0].imshow(gt_right_grid, cmap='RdBu_r', vmin=vmin_right, vmax=vmax_right)
        axes[0, 0].set_title('Ground Truth Right Eigenvector (Real)')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[0, 0])

        # Plot empirical right eigenvector
        im1 = axes[0, 1].imshow(emp_right_grid, cmap='RdBu_r', vmin=vmin_right, vmax=vmax_right)
        axes[0, 1].set_title('Empirical Right Eigenvector (Real)')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[0, 1])

        # Plot ground truth left eigenvector
        im2 = axes[1, 0].imshow(gt_left_grid, cmap='RdBu_r', vmin=vmin_left, vmax=vmax_left)
        axes[1, 0].set_title('Ground Truth Left Eigenvector (Real)')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[1, 0])

        # Plot empirical left eigenvector
        im3 = axes[1, 1].imshow(emp_left_grid, cmap='RdBu_r', vmin=vmin_left, vmax=vmax_left)
        axes[1, 1].set_title('Empirical Left Eigenvector (Real)')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(save_dir / f"eigenvector_{evec_idx}_spatial.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   Saved spatial visualizations for {num_to_visualize} eigenvectors to {save_dir}")

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

    # Return summary statistics
    return {
        'num_states': num_states,
        'num_samples': total_samples,
        'eigenvalue_error_mean': eigenvalue_error_magnitude.mean(),
        'eigenvalue_error_std': eigenvalue_error_magnitude.std(),
        'unsampled_states': len(unsampled_states),
    }


if __name__ == "__main__":
    test_args = tyro.cli(TestArgs)
    results = test_sampling_distribution(test_args)

    print("\nSummary:")
    print(f"  Number of states: {results['num_states']}")
    print(f"  Number of samples: {results['num_samples']}")
    print(f"  Mean eigenvalue error: {results['eigenvalue_error_mean']:.6f}")
    print(f"  Std eigenvalue error: {results['eigenvalue_error_std']:.6f}")
    print(f"  Unsampled states: {results['unsampled_states']}")
