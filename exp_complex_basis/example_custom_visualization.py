"""
Example script demonstrating custom eigenvector visualization options.

This script shows how to use the new features:
1. Plotting left and right eigenvectors in the same figure
2. Customizing the number of rows and columns
3. Changing the wall color
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import jax.numpy as jnp
import numpy as np

from exp_complex_basis import (
    visualize_multiple_eigenvectors,
    visualize_left_right_eigenvectors,
)


def example_custom_layout():
    """Example: Custom layout with more eigenvectors and different wall color."""

    # Load previously saved results
    import pickle
    results_file = Path("exp_complex_basis/results/eigendecomposition_results_batched.pkl")

    if not results_file.exists():
        print("Please run run_analysis.py first to generate results.")
        return

    with open(results_file, "rb") as f:
        results = pickle.load(f)

    # Extract data for first environment
    eigendecomp = results["batched_eigendecomposition"]
    first_env_eigendecomp = {key: value[0] for key, value in eigendecomp.items()}
    metadata = results["metadata"]
    canonical_states = metadata["canonical_states"]

    # Reconstruct portals
    portals = {}
    if "first_env_portals" in metadata:
        for portal in metadata["first_env_portals"]:
            if len(portal) == 3:
                source_canonical, action, dest_canonical = portal
                if source_canonical >= 0 and dest_canonical >= 0:
                    source_full = int(canonical_states[source_canonical])
                    dest_full = int(canonical_states[dest_canonical])
                    portals[(source_full, action)] = dest_full

    output_dir = Path("exp_complex_basis/results/custom_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Example 1: Plot 12 eigenvectors in a 3x4 grid with pink walls
    print("\nExample 1: 12 eigenvectors in 3 rows x 4 cols with pink walls")
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(min(12, first_env_eigendecomp['eigenvalues'].shape[0]))),
        eigendecomposition=first_env_eigendecomp,
        canonical_states=canonical_states,
        grid_width=metadata["grid_width"],
        grid_height=metadata["grid_height"],
        portals=portals,
        eigenvector_type='right',
        component='real',
        nrows=3,
        ncols=4,
        wall_color='pink',
        save_path=output_dir / "example1_12eigvecs_3x4_pink.png"
    )
    print(f"  Saved to {output_dir / 'example1_12eigvecs_3x4_pink.png'}")

    # Example 2: Plot 8 eigenvectors in a 2x4 grid with light blue walls
    print("\nExample 2: 8 eigenvectors in 2 rows x 4 cols with light blue walls")
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(min(8, first_env_eigendecomp['eigenvalues'].shape[0]))),
        eigendecomposition=first_env_eigendecomp,
        canonical_states=canonical_states,
        grid_width=metadata["grid_width"],
        grid_height=metadata["grid_height"],
        portals=portals,
        eigenvector_type='left',
        component='real',
        nrows=2,
        ncols=4,
        wall_color='lightblue',
        save_path=output_dir / "example2_8eigvecs_2x4_lightblue.png"
    )
    print(f"  Saved to {output_dir / 'example2_8eigvecs_2x4_lightblue.png'}")

    # Example 3: Left-right comparison with 8 eigenvectors and yellow walls
    print("\nExample 3: Left vs Right comparison for 8 eigenvectors with yellow walls")
    visualize_left_right_eigenvectors(
        eigenvector_indices=list(range(min(8, first_env_eigendecomp['eigenvalues'].shape[0]))),
        eigendecomposition=first_env_eigendecomp,
        canonical_states=canonical_states,
        grid_width=metadata["grid_width"],
        grid_height=metadata["grid_height"],
        portals=portals,
        component='real',
        wall_color='lightyellow',
        save_path=output_dir / "example3_leftright_8eigvecs_yellow.png"
    )
    print(f"  Saved to {output_dir / 'example3_leftright_8eigvecs_yellow.png'}")

    # Example 4: Compact view - 15 eigenvectors in 3x5 grid (default gray walls)
    print("\nExample 4: 15 eigenvectors in 3 rows x 5 cols with default gray walls")
    visualize_multiple_eigenvectors(
        eigenvector_indices=list(range(min(15, first_env_eigendecomp['eigenvalues'].shape[0]))),
        eigendecomposition=first_env_eigendecomp,
        canonical_states=canonical_states,
        grid_width=metadata["grid_width"],
        grid_height=metadata["grid_height"],
        portals=portals,
        eigenvector_type='right',
        component='imag',
        nrows=3,
        ncols=5,
        # wall_color defaults to 'gray'
        save_path=output_dir / "example4_15eigvecs_3x5_gray.png"
    )
    print(f"  Saved to {output_dir / 'example4_15eigvecs_3x5_gray.png'}")

    print(f"\n✓ All custom visualizations saved to {output_dir}")


if __name__ == "__main__":
    example_custom_layout()
