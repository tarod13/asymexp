# Eigendecomposition Analysis for Non-Symmetrized Dynamics Matrices

This package provides tools to analyze the eigendecomposition of non-symmetrized transition/dynamics matrices and compare distances in eigenspace with actual environment distances.

## Overview

The analysis focuses on:

1. **Eigendecomposition of non-symmetrized matrices**: Unlike SVD or symmetrized eigendecomposition, this preserves the directional nature of transitions and can produce complex eigenvalues/eigenvectors.

2. **Distance analysis in eigenspace**: Computes pairwise distances between states using:
   - Real components of eigenvectors
   - Imaginary components of eigenvectors
   - Combined real and imaginary components

3. **Comparison with environment distances**: Compares eigenspace distances with actual distances in the environment (Euclidean, Manhattan, shortest path).

## Files

- `eigendecomposition.py`: Core functions for computing eigendecomposition
- `distance_analysis.py`: Distance computation and comparison utilities
- `run_analysis.py`: Main script to run the complete analysis
- `visualization.py`: Visualization utilities for results
- `__init__.py`: Package initialization

## Quick Start

### Running the Analysis

```bash
# Run with default parameters (generates new data)
python exp_complex_basis/run_analysis.py --generate-new

# Run with custom parameters
python exp_complex_basis/run_analysis.py \
    --generate-new \
    --num-envs 20 \
    --num-episodes 200 \
    --k 30 \
    --grid-width 13

# Load existing transition data
python exp_complex_basis/run_analysis.py \
    --data-path path/to/transition_data.pkl
```

### Command-line Arguments

- `--data-path`: Path to saved transition data (pickle file)
- `--generate-new`: Generate new transition data instead of loading
- `--num-envs`: Number of environments for data generation (default: 10)
- `--num-episodes`: Number of episodes per environment (default: 100)
- `--max-steps`: Maximum steps per episode (default: 100)
- `--k`: Number of eigenvectors to compute (default: 20)
- `--grid-width`: Width of the grid environment (default: 13)
- `--output-dir`: Output directory for results (default: exp_complex_basis/results)
- `--seed`: Random seed (default: 42)

## Using as a Library

```python
import jax.numpy as jnp
from exp_complex_basis import (
    compute_eigendecomposition,
    compute_eigenspace_distances,
    compute_environment_distances,
    compare_distances,
)

# Compute eigendecomposition
eigendecomp = compute_eigendecomposition(transition_matrix, k=20)

# Analyze eigenspace distances
eigen_dists = compute_eigenspace_distances(
    eigendecomp,
    metric="euclidean",
    use_real=True,
    use_imag=True,
    k=10
)

# Compute environment distances
states = jnp.arange(num_states)
env_dists = compute_environment_distances(
    states,
    grid_width=13,
    transition_matrix=transition_matrix
)

# Compare distances
comparison = compare_distances(
    eigen_dists["distances_combined"],
    env_dists["euclidean"]
)
print(f"Correlation: {comparison['correlation']:.4f}")
```

## Output

The analysis produces:

1. **Results file** (`eigendecomposition_results.pkl`): Complete analysis results including:
   - Transition matrix
   - Eigenvalues and eigenvectors (complex)
   - Real and imaginary components separated
   - Distance matrices (eigenspace and environment)
   - Correlation metrics

2. **Summary file** (`analysis_summary.txt`): Text summary of key findings

3. **Visualizations** (if using `visualization.py`):
   - Eigenvalue spectrum plots
   - Eigenvector heatmaps
   - Distance matrix heatmaps
   - Scatter plots comparing eigenspace vs environment distances

## Understanding the Results

### Eigenvalue Spectrum

- **Real eigenvalues**: Correspond to non-oscillatory dynamics
- **Complex eigenvalues**: Indicate oscillatory/rotational dynamics in state space
- Magnitude indicates the importance/scale of each mode

### Distance Correlations

High positive correlation between eigenspace and environment distances suggests that:
- The eigenspace representation preserves geometric structure
- States close in eigenspace are also close in the environment

Different correlations for real vs imaginary components can reveal:
- Real components may capture symmetric/diffusive dynamics
- Imaginary components may capture asymmetric/directional dynamics

### Interpretation

- **k values**: Different numbers of eigenvectors capture different levels of detail
  - Small k: Captures only dominant modes
  - Large k: Captures finer-grained structure

- **Real vs Imaginary**:
  - Real components often dominate for nearly symmetric dynamics
  - Imaginary components become significant when transitions are highly directional

## Example Analysis Workflow

```python
from exp_complex_basis.run_analysis import run_eigendecomposition_analysis
from exp_complex_basis.visualization import create_full_analysis_report

# Generate and analyze data
transition_counts = generate_transition_data(
    num_envs=20,
    num_episodes=200,
    max_steps=100
)

# Run analysis
results = run_eigendecomposition_analysis(
    transition_counts=transition_counts,
    grid_width=13,
    k=30,
    k_values_to_analyze=[5, 10, 20, 30],
    output_dir="exp_complex_basis/results"
)

# Generate visualizations
create_full_analysis_report(
    results,
    output_dir="exp_complex_basis/results",
    max_k_plot=30
)
```

## Theory Background

For a non-symmetric matrix M, the eigendecomposition is:

```
M v = λ v
```

where:
- λ can be complex (λ = a + bi)
- v can be complex (v = u + iw)

For complex eigenvectors:
- Real part (u): Represents one phase of oscillation
- Imaginary part (w): Represents the quadrature phase

The distance between states in eigenspace uses both components to capture the full structure of the dynamics.

## Dependencies

- JAX (for numerical computations)
- NumPy
- Matplotlib (for visualizations)
- Seaborn (for visualizations)
- Standard library (pickle, argparse, pathlib)

## Notes

- The transition matrix is **not symmetrized** (unlike standard SVD approaches)
- This preserves directional information in state transitions
- Complex eigenvalues/vectors are expected and informative
- All computations use JAX for efficiency and GPU compatibility
