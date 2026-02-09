#!/usr/bin/env python3
"""
Generate plots from exported ALLO training data.

This script can be run while training is ongoing or after training completes.
It reads exported numpy arrays and JSON files to create all visualizations.

Usage:
    python generate_plots.py <results_dir>
    python generate_plots.py results/room4/room4__allo__0__42__1234567890

The script generates:
- Ground truth eigenvectors
- Learned eigenvectors at each checkpoint
- Learning curves
- Dual variable evolution (with comparison to simple Laplacian if available)
- Sampling distribution visualization
- Final comparison plot
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

from exp_alcl.allo import plot_learning_curves, plot_dual_variable_evolution, plot_cosine_similarity_evolution, plot_sampling_distribution
from src.utils.plotting import (
    visualize_multiple_eigenvectors,
    visualize_eigenvector_on_grid,
)


def load_data(results_dir):
    """Load all exported data from results directory."""
    results_dir = Path(results_dir)

    # Load visualization metadata
    with open(results_dir / "viz_metadata.pkl", 'rb') as f:
        viz_metadata = pickle.load(f)

    # Load ground truth
    gt_eigenvalues = np.load(results_dir / "gt_eigenvalues.npy")
    gt_eigenvectors = np.load(results_dir / "gt_eigenvectors.npy")

    # Load simple Laplacian eigenvalues if available
    simple_eigenvalues_file = results_dir / "gt_eigenvalues_simple.npy"
    if simple_eigenvalues_file.exists():
        gt_eigenvalues_simple = np.load(simple_eigenvalues_file)
    else:
        gt_eigenvalues_simple = None

    # Load weighted Laplacian eigenvalues if available
    weighted_eigenvalues_file = results_dir / "gt_eigenvalues_weighted.npy"
    if weighted_eigenvalues_file.exists():
        gt_eigenvalues_weighted = np.load(weighted_eigenvalues_file)
    else:
        gt_eigenvalues_weighted = None

    # Load inverse-weighted Laplacian eigenvalues if available
    invweighted_eigenvalues_file = results_dir / "gt_eigenvalues_invweighted.npy"
    if invweighted_eigenvalues_file.exists():
        gt_eigenvalues_invweighted = np.load(invweighted_eigenvalues_file)
    else:
        gt_eigenvalues_invweighted = None

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

    # Load latest learned eigenvectors (single file, overwritten during training)
    latest_eigenvectors_file = results_dir / "latest_learned_eigenvectors.npy"
    if latest_eigenvectors_file.exists():
        latest_eigenvectors = np.load(latest_eigenvectors_file)
    else:
        latest_eigenvectors = None
        print("Warning: latest_learned_eigenvectors.npy not found")

    return {
        'viz_metadata': viz_metadata,
        'gt_eigenvalues': gt_eigenvalues,
        'gt_eigenvectors': gt_eigenvectors,
        'gt_eigenvalues_simple': gt_eigenvalues_simple,
        'gt_eigenvalues_weighted': gt_eigenvalues_weighted,
        'gt_eigenvalues_invweighted': gt_eigenvalues_invweighted,
        'sampling_probs': sampling_probs,
        'metrics_history': metrics_history,
        'latest_eigenvectors': latest_eigenvectors,
        'results_dir': results_dir,
    }


def plot_ground_truth(data, plots_dir):
    """Generate ground truth eigenvector plots for all available Laplacian formulations."""
    viz_meta = data['viz_metadata']

    # Plot inverse-weighted Laplacian eigenvectors (main baseline)
    print("Plotting inverse-weighted Laplacian ground truth eigenvectors...")
    gt_eigenvectors = data['gt_eigenvectors']
    num_eigs = gt_eigenvectors.shape[1]
    eigendecomp = {
        'eigenvalues': data['gt_eigenvalues'].astype(np.complex64),
        'eigenvalues_real': data['gt_eigenvalues'],
        'eigenvalues_imag': np.zeros_like(data['gt_eigenvalues']),
        'right_eigenvectors_real': gt_eigenvectors,
        'right_eigenvectors_imag': np.zeros_like(gt_eigenvectors),
        'left_eigenvectors_real': gt_eigenvectors,
        'left_eigenvectors_imag': np.zeros_like(gt_eigenvectors),
    }

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
        save_path=str(plots_dir / "ground_truth_eigenvectors_invweighted.png"),
        shared_colorbar=True
    )
    plt.close()

    # Plot simple Laplacian eigenvectors if available
    if data['gt_eigenvalues_simple'] is not None:
        print("Plotting simple Laplacian ground truth eigenvectors...")
        # Load the simple eigenvectors
        simple_eigenvectors_file = data['results_dir'] / "gt_eigenvectors_simple.npy"
        if simple_eigenvectors_file.exists():
            gt_eigenvectors_simple = np.load(simple_eigenvectors_file)

            eigendecomp_simple = {
                'eigenvalues': data['gt_eigenvalues_simple'].astype(np.complex64),
                'eigenvalues_real': data['gt_eigenvalues_simple'],
                'eigenvalues_imag': np.zeros_like(data['gt_eigenvalues_simple']),
                'right_eigenvectors_real': gt_eigenvectors_simple,
                'right_eigenvectors_imag': np.zeros_like(gt_eigenvectors_simple),
                'left_eigenvectors_real': gt_eigenvectors_simple,
                'left_eigenvectors_imag': np.zeros_like(gt_eigenvectors_simple),
            }

            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(num_eigs)),
                eigendecomposition=eigendecomp_simple,
                canonical_states=viz_meta['canonical_states'],
                grid_width=viz_meta['grid_width'],
                grid_height=viz_meta['grid_height'],
                portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
                eigenvector_type='right',
                component='real',
                ncols=min(4, num_eigs),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_eigenvectors_simple.png"),
                shared_colorbar=True
            )
            plt.close()

    # Plot weighted Laplacian eigenvectors if available
    if data['gt_eigenvalues_weighted'] is not None:
        print("Plotting weighted Laplacian ground truth eigenvectors...")
        # Load the weighted eigenvectors
        weighted_eigenvectors_file = data['results_dir'] / "gt_eigenvectors_weighted.npy"
        if weighted_eigenvectors_file.exists():
            gt_eigenvectors_weighted = np.load(weighted_eigenvectors_file)

            eigendecomp_weighted = {
                'eigenvalues': data['gt_eigenvalues_weighted'].astype(np.complex64),
                'eigenvalues_real': data['gt_eigenvalues_weighted'],
                'eigenvalues_imag': np.zeros_like(data['gt_eigenvalues_weighted']),
                'right_eigenvectors_real': gt_eigenvectors_weighted,
                'right_eigenvectors_imag': np.zeros_like(gt_eigenvectors_weighted),
                'left_eigenvectors_real': gt_eigenvectors_weighted,
                'left_eigenvectors_imag': np.zeros_like(gt_eigenvectors_weighted),
            }

            visualize_multiple_eigenvectors(
                eigenvector_indices=list(range(num_eigs)),
                eigendecomposition=eigendecomp_weighted,
                canonical_states=viz_meta['canonical_states'],
                grid_width=viz_meta['grid_width'],
                grid_height=viz_meta['grid_height'],
                portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
                eigenvector_type='right',
                component='real',
                ncols=min(4, num_eigs),
                wall_color='gray',
                save_path=str(plots_dir / "ground_truth_eigenvectors_weighted.png"),
                shared_colorbar=True
            )
            plt.close()


def plot_latest_learned(data, plots_dir):
    """Generate learned eigenvector plots for the latest checkpoint."""
    if data['latest_eigenvectors'] is None:
        print("Skipping latest learned eigenvectors plot (data not available)")
        return

    print("Plotting latest learned eigenvectors...")

    viz_meta = data['viz_metadata']
    num_eigs = viz_meta['num_eigenvectors']
    learned_features = data['latest_eigenvectors']

    # Create eigendecomposition dict
    learned_eigendecomp = {
        'eigenvalues': np.zeros(num_eigs, dtype=np.complex64),
        'eigenvalues_real': np.zeros(num_eigs),
        'eigenvalues_imag': np.zeros(num_eigs),
        'right_eigenvectors_real': learned_features,
        'right_eigenvectors_imag': np.zeros_like(learned_features),
        'left_eigenvectors_real': learned_features,
        'left_eigenvectors_imag': np.zeros_like(learned_features),
    }

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
        save_path=str(plots_dir / "learned_eigenvectors_latest.png"),
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
        data['gt_eigenvalues'],
        gamma,
        str(plots_dir / "dual_variable_evolution.png"),
        num_eigenvectors=data['viz_metadata']['num_eigenvectors'],
        ground_truth_eigenvalues_simple=data.get('gt_eigenvalues_simple'),
        ground_truth_eigenvalues_weighted=data.get('gt_eigenvalues_weighted')
    )

    print("Plotting cosine similarity evolution...")
    plot_cosine_similarity_evolution(
        data['metrics_history'],
        str(plots_dir / "cosine_similarity_evolution.png"),
        num_eigenvectors=data['viz_metadata']['num_eigenvectors']
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
    """Generate final comparison plot."""
    print("Plotting final comparison...")

    viz_meta = data['viz_metadata']

    # Load final learned features
    final_file = data['results_dir'] / "final_learned_eigenvectors.npy"
    if not final_file.exists():
        print("Warning: final_learned_eigenvectors.npy not found, skipping final comparison")
        return

    final_learned_features = np.load(final_file)
    gt_eigenvectors = data['gt_eigenvectors']

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    # Plot first ground truth eigenvector (skip constant eigenvector 0)
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=gt_eigenvectors[:, 1],
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        title='Ground Truth Eigenvector 1',
        ax=axes[0],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    # Plot first learned feature
    visualize_eigenvector_on_grid(
        eigenvector_idx=1,
        eigenvector_values=final_learned_features[:, 1],
        canonical_states=viz_meta['canonical_states'],
        grid_width=viz_meta['grid_width'],
        grid_height=viz_meta['grid_height'],
        portals=viz_meta['door_markers'] if viz_meta['door_markers'] else None,
        title='Learned Feature 1',
        ax=axes[1],
        cmap='RdBu_r',
        show_colorbar=True,
        wall_color='gray'
    )

    plt.tight_layout()
    plt.savefig(plots_dir / "final_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from ALLO training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'results_dir',
        type=str,
        help='Path to results directory (e.g., results/room4/room4__allo__0__42__1234567890)'
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

    if not args.skip_learned:
        plot_latest_learned(data, plots_dir)

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
