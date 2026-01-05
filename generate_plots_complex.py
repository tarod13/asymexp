#!/usr/bin/env python3
"""
Generate plots from exported ALLO Complex training data.

This script can be run while training is ongoing or after training completes.
It reads exported numpy arrays and JSON files to create all visualizations
for complex eigenvectors (left/right, real/imag).

Usage:
    python generate_plots_complex.py <results_dir>
    python generate_plots_complex.py results/room4/room4__allo_complex__0__42__1234567890

The script generates:
- Ground truth eigenvectors (left and right, real and imaginary parts)
- Learned eigenvectors at each checkpoint (all 4 components)
- Learning curves
- Dual variable evolution
- Cosine similarity evolution (left and right separately)
- Sampling distribution visualization
- Final comparison plots
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from exp_alcl.allo_complex import (
    plot_learning_curves,
    plot_dual_variable_evolution,
    plot_cosine_similarity_evolution,
    plot_sampling_distribution,
    plot_all_duals_evolution
)
from exp_complex_basis.eigenvector_visualization import (
    visualize_multiple_eigenvectors,
    visualize_eigenvector_on_grid,
)


def load_data(results_dir):
    """Load all exported data from results directory."""
    results_dir = Path(results_dir)

    # Load visualization metadata
    with open(results_dir / "viz_metadata.pkl", 'rb') as f:
        viz_metadata = pickle.load(f)

    # Load ground truth eigenvalues (complex)
    gt_eigenvalues_real = np.load(results_dir / "gt_eigenvalues_real.npy")
    gt_eigenvalues_imag = np.load(results_dir / "gt_eigenvalues_imag.npy")
    gt_eigenvalues = gt_eigenvalues_real + 1j * gt_eigenvalues_imag

    # Load ground truth eigenvectors (complex, left and right)
    gt_left_real = np.load(results_dir / "gt_left_real.npy")
    gt_left_imag = np.load(results_dir / "gt_left_imag.npy")
    gt_right_real = np.load(results_dir / "gt_right_real.npy")
    gt_right_imag = np.load(results_dir / "gt_right_imag.npy")

    # Load sampling distribution if available
    sampling_dist_file = results_dir / "sampling_distribution.npy"
    if sampling_dist_file.exists():
        sampling_probs = np.load(sampling_dist_file)
    else:
        sampling_probs = None

    # Load metrics history (may not exist if training is still running)
    metrics_file = results_dir / "metrics_history.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics_history = json.load(f)
    else:
        metrics_history = None
        print("Warning: metrics_history.json not found (training may still be running)")

    # Load latest learned eigenvectors (4 separate files)
    latest_left_real_file = results_dir / "latest_learned_left_real.npy"
    latest_left_imag_file = results_dir / "latest_learned_left_imag.npy"
    latest_right_real_file = results_dir / "latest_learned_right_real.npy"
    latest_right_imag_file = results_dir / "latest_learned_right_imag.npy"

    if (latest_left_real_file.exists() and latest_left_imag_file.exists() and
        latest_right_real_file.exists() and latest_right_imag_file.exists()):
        latest_learned = {
            'left_real': np.load(latest_left_real_file),
            'left_imag': np.load(latest_left_imag_file),
            'right_real': np.load(latest_right_real_file),
            'right_imag': np.load(latest_right_imag_file),
        }
    else:
        latest_learned = None
        print("Warning: latest_learned_*.npy files not found")

    # Load final learned eigenvectors (4 separate files)
    final_left_real_file = results_dir / "final_learned_left_real.npy"
    final_left_imag_file = results_dir / "final_learned_left_imag.npy"
    final_right_real_file = results_dir / "final_learned_right_real.npy"
    final_right_imag_file = results_dir / "final_learned_right_imag.npy"

    if (final_left_real_file.exists() and final_left_imag_file.exists() and
        final_right_real_file.exists() and final_right_imag_file.exists()):
        final_learned = {
            'left_real': np.load(final_left_real_file),
            'left_imag': np.load(final_left_imag_file),
            'right_real': np.load(final_right_real_file),
            'right_imag': np.load(final_right_imag_file),
        }
    else:
        final_learned = None
        print("Warning: final_learned_*.npy files not found")

    return {
        'viz_metadata': viz_metadata,
        'gt_eigenvalues': gt_eigenvalues,
        'gt_eigenvalues_real': gt_eigenvalues_real,
        'gt_eigenvalues_imag': gt_eigenvalues_imag,
        'gt_left_real': gt_left_real,
        'gt_left_imag': gt_left_imag,
        'gt_right_real': gt_right_real,
        'gt_right_imag': gt_right_imag,
        'sampling_probs': sampling_probs,
        'metrics_history': metrics_history,
        'latest_learned': latest_learned,
        'final_learned': final_learned,
        'results_dir': results_dir,
    }


def plot_ground_truth(data, plots_dir):
    """Generate ground truth eigenvector plots for all components."""
    viz_meta = data['viz_metadata']
    num_eigs = data['gt_right_real'].shape[1]

    eigendecomp = {
        'eigenvalues': data['gt_eigenvalues'],
        'eigenvalues_real': data['gt_eigenvalues_real'],
        'eigenvalues_imag': data['gt_eigenvalues_imag'],
        'right_eigenvectors_real': data['gt_right_real'],
        'right_eigenvectors_imag': data['gt_right_imag'],
        'left_eigenvectors_real': data['gt_left_real'],
        'left_eigenvectors_imag': data['gt_left_imag'],
    }

    # Plot right eigenvectors (real parts)
    print("Plotting ground truth right eigenvectors (real parts)...")
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='right',
        component='real',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "ground_truth_right_eigenvectors_real.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot right eigenvectors (imaginary parts)
    print("Plotting ground truth right eigenvectors (imaginary parts)...")
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='right',
        component='imag',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "ground_truth_right_eigenvectors_imag.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot left eigenvectors (real parts)
    print("Plotting ground truth left eigenvectors (real parts)...")
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='left',
        component='real',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "ground_truth_left_eigenvectors_real.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot left eigenvectors (imaginary parts)
    print("Plotting ground truth left eigenvectors (imaginary parts)...")
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='left',
        component='imag',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "ground_truth_left_eigenvectors_imag.png"),
        shared_colorbar=True
    )
    plt.close()


def plot_latest_learned(data, plots_dir):
    """Generate learned eigenvector plots for the latest checkpoint."""
    if data['latest_learned'] is None:
        print("Skipping latest learned eigenvectors plot (data not available)")
        return

    print("Plotting latest learned eigenvectors...")

    viz_meta = data['viz_metadata']
    num_eigs = viz_meta['num_eigenvectors']
    latest = data['latest_learned']

    # Create eigendecomposition dict
    learned_eigendecomp = {
        'eigenvalues': np.zeros(num_eigs, dtype=np.complex64),
        'eigenvalues_real': np.zeros(num_eigs),
        'eigenvalues_imag': np.zeros(num_eigs),
        'right_eigenvectors_real': latest['right_real'],
        'right_eigenvectors_imag': latest['right_imag'],
        'left_eigenvectors_real': latest['left_real'],
        'left_eigenvectors_imag': latest['left_imag'],
    }

    # Plot right eigenvectors (real parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='right',
        component='real',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_right_eigenvectors_real_latest.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot right eigenvectors (imaginary parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='right',
        component='imag',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_right_eigenvectors_imag_latest.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot left eigenvectors (real parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='left',
        component='real',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_left_eigenvectors_real_latest.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot left eigenvectors (imaginary parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='left',
        component='imag',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_left_eigenvectors_imag_latest.png"),
        shared_colorbar=True
    )
    plt.close()


def plot_final_learned(data, plots_dir):
    """Generate final learned eigenvector plots."""
    if data['final_learned'] is None:
        print("Skipping final learned eigenvectors plot (data not available)")
        return

    print("Plotting final learned eigenvectors...")

    viz_meta = data['viz_metadata']
    num_eigs = viz_meta['num_eigenvectors']
    final = data['final_learned']

    # Create eigendecomposition dict
    learned_eigendecomp = {
        'eigenvalues': np.zeros(num_eigs, dtype=np.complex64),
        'eigenvalues_real': np.zeros(num_eigs),
        'eigenvalues_imag': np.zeros(num_eigs),
        'right_eigenvectors_real': final['right_real'],
        'right_eigenvectors_imag': final['right_imag'],
        'left_eigenvectors_real': final['left_real'],
        'left_eigenvectors_imag': final['left_imag'],
    }

    # Plot right eigenvectors (real parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='right',
        component='real',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_right_eigenvectors_real_final.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot right eigenvectors (imaginary parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='right',
        component='imag',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_right_eigenvectors_imag_final.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot left eigenvectors (real parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='left',
        component='real',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_left_eigenvectors_real_final.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot left eigenvectors (imaginary parts)
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(num_eigs)),
        eigendecomposition=learned_eigendecomp,
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        eigenvector_type='left',
        component='imag',
        ncols=min(4, num_eigs),
        wall_color='gray',
        save_path=str(plots_dir / "learned_left_eigenvectors_imag_final.png"),
        shared_colorbar=True
    )
    plt.close()


def plot_learning_metrics(data, plots_dir):
    """Generate learning curves, dual evolution, and cosine similarity plots."""
    if data['metrics_history'] is None:
        print("Skipping metrics plots (metrics_history.json not available)")
        print("  This file is created when training completes.")
        return

    print("Plotting learning curves...")
    plot_learning_curves(
        data['metrics_history'],
        str(plots_dir / "learning_curves.png")
    )

    print("Plotting dual variable evolution...")
    # Use 'gamma' if 'geometric_gamma' is not present (for newer runs)
    gamma = data['viz_metadata'].get('gamma', data['viz_metadata'].get('geometric_gamma', 0.99))
    plot_dual_variable_evolution(
        data['metrics_history'],
        data['gt_eigenvalues'],  # Use full complex eigenvalues
        gamma,
        str(plots_dir / "dual_variable_evolution.png"),
        num_eigenvectors=data['viz_metadata']['num_eigenvectors']
    )

    print("Plotting cosine similarity evolution...")
    plot_cosine_similarity_evolution(
        data['metrics_history'],
        str(plots_dir / "cosine_similarity_evolution.png"),
        num_eigenvectors=data['viz_metadata']['num_eigenvectors']
    )

    print("Plotting all duals evolution...")
    plot_all_duals_evolution(
        data['metrics_history'],
        str(plots_dir / "all_duals_evolution.png"),
        num_eigenvectors=min(6, data['viz_metadata']['num_eigenvectors'])
    )


def plot_sampling_dist(data, plots_dir):
    """Generate sampling distribution visualization."""
    if data['sampling_probs'] is None:
        print("Skipping sampling distribution plot (data not available)")
        return

    print("Plotting sampling distribution...")

    viz_meta = data['viz_metadata']

    plot_sampling_distribution(
        sampling_probs=data['sampling_probs'],
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        save_path=str(plots_dir / "sampling_distribution.png"),
        portals=viz_meta['door_markers'] if viz_meta.get('door_markers') else None
    )


def plot_final_comparison(data, plots_dir):
    """Generate final comparison plots (ground truth vs learned)."""
    if data['final_learned'] is None:
        print("Skipping final comparison (final learned data not available)")
        return

    print("Plotting final comparison...")

    viz_meta = data['viz_metadata']

    # Compare right eigenvectors (real parts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Plot first ground truth right eigenvector (real part, skip constant eigenvector 0)
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=data['gt_right_real'][:, 1],
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        title='Ground Truth Right Eigenvector 1 (Real)',
        ax=axes[0],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    # Plot first learned right eigenvector (real part)
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=data['final_learned']['right_real'][:, 1],
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        title='Learned Right Feature 1 (Real)',
        ax=axes[1],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    plt.tight_layout()
    plt.savefig(plots_dir / "final_comparison_right_real.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Compare left eigenvectors (real parts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Plot first ground truth left eigenvector (real part)
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=data['gt_left_real'][:, 1],
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        title='Ground Truth Left Eigenvector 1 (Real)',
        ax=axes[0],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    # Plot first learned left eigenvector (real part)
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=data['final_learned']['left_real'][:, 1],
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        title='Learned Left Feature 1 (Real)',
        ax=axes[1],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    plt.tight_layout()
    plt.savefig(plots_dir / "final_comparison_left_real.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from ALLO Complex training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to results directory (e.g., results/room4/room4__allo_complex__0__42__1234567890)'
    )
    parser.add_argument(
        '--skip-ground-truth',
        action='store_true',
        help='Skip plotting ground truth eigenvectors'
    )
    parser.add_argument(
        '--skip-learned',
        action='store_true',
        help='Skip plotting latest learned eigenvectors'
    )
    parser.add_argument(
        '--include-latest',
        action='store_true',
        help='Include latest learned plots (identical to final after training completes)'
    )
    parser.add_argument(
        '--skip-final',
        action='store_true',
        help='Skip plotting final learned eigenvectors'
    )
    parser.add_argument(
        '--skip-metrics',
        action='store_true',
        help='Skip plotting learning curves, dual evolution, and cosine similarity'
    )
    parser.add_argument(
        '--skip-comparison',
        action='store_true',
        help='Skip plotting final comparison'
    )
    parser.add_argument(
        '--skip-sampling-dist',
        action='store_true',
        help='Skip plotting sampling distribution'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    # Create plots directory
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print(f"Loading data from {results_dir}...")
    try:
        data = load_data(results_dir)
    except FileNotFoundError as e:
        print(f"Error: Required data file not found: {e}")
        print("Make sure training has exported data files.")
        return 1

    # Generate plots
    if not args.skip_ground_truth:
        plot_ground_truth(data, plots_dir)

    # Note: "latest" and "final" are identical after training completes
    # Only plot "latest" if explicitly requested (useful during training)
    if not args.skip_learned and args.include_latest:
        plot_latest_learned(data, plots_dir)

    if not args.skip_final:
        plot_final_learned(data, plots_dir)

    if not args.skip_metrics:
        plot_learning_metrics(data, plots_dir)

    if not args.skip_sampling_dist:
        plot_sampling_dist(data, plots_dir)

    if not args.skip_comparison:
        plot_final_comparison(data, plots_dir)

    print(f"\n✓ All plots saved to {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
