"""
Visualization utilities for hitting time analysis.

Core visualization functions (visualize_hitting_time_on_grid, visualize_source_vs_target_hitting_times)
have been moved to src/utils/plotting.py. They are re-exported here for backward compatibility.
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from src.utils.plotting import (
    visualize_hitting_time_on_grid,
    visualize_source_vs_target_hitting_times,
)


def create_hitting_time_visualization_report(
    hitting_time_matrix: jnp.ndarray,
    canonical_states: jnp.ndarray,
    grid_width: int,
    grid_height: int,
    portals: Optional[Dict[Tuple[int, int], int]] = None,
    output_dir: str = "exp_complex_basis/results/visualizations",
    num_targets: int = 6,
    target_indices: Optional[List[int]] = None,
    ncols: int = 6,
    wall_color: str = 'gray',
    log_scale: bool = False,
):
    """
    Create a complete visualization report for hitting times.
    Generates two versions: one with shared scale (limits compared) and one with independent scales.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating hitting time visualizations...")

    num_states = hitting_time_matrix.shape[0]

    if target_indices is None:
        target_indices = np.linspace(0, num_states - 1, num_targets, dtype=int).tolist()
    
    print(f"  Visualizing source/target comparisons for {len(target_indices)} states...")
    
    # 1. Shared Scale Version (Limits compared)
    filename_shared = f"hitting_times_asymmetry{'_log' if log_scale else ''}_shared_scale.png"
    visualize_source_vs_target_hitting_times(
        state_indices=target_indices,
        hitting_time_matrix=hitting_time_matrix,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        ncols=ncols,
        wall_color=wall_color,
        save_path=output_path / filename_shared,
        log_scale=log_scale,
        shared_colorbar=True
    )
    plt.close()

    # 2. Independent Scale Version
    filename_independent = f"hitting_times_asymmetry{'_log' if log_scale else ''}_independent_scale.png"
    visualize_source_vs_target_hitting_times(
        state_indices=target_indices,
        hitting_time_matrix=hitting_time_matrix,
        canonical_states=canonical_states,
        grid_width=grid_width,
        grid_height=grid_height,
        portals=portals,
        ncols=ncols,
        wall_color=wall_color,
        save_path=output_path / filename_independent,
        log_scale=log_scale,
        shared_colorbar=False
    )
    plt.close()

    print(f"\nHitting time visualization reports saved to {output_dir}")