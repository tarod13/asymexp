# Experiment 01: Complex Eigendecomposition of Portal Environments

Tests whether the complex eigendecomposition of non-symmetrized transition
matrices yields meaningful representations. Specifically, it checks if
eigenspace distances (real and imaginary components) correlate with actual
environment distances (Euclidean, Manhattan, shortest path) in grid
environments with random portals (teleportations).

## What it does

1. Creates multiple grid environments with random portals (asymmetric transitions)
2. Collects transition data via random rollouts
3. Computes eigendecomposition of the non-symmetrized transition matrices (batched)
4. Computes hitting times from the eigendecomposition
5. Analyzes distances in eigenspace vs. environment distances
6. Generates visualizations of eigenvectors, hitting times, and distance comparisons

## Usage

```bash
python experiments/01/run_analysis.py --generate-new --num-envs 10 --num-portals 10 --k 20
```

## Key finding

Complex eigenvectors of asymmetric transition matrices encode directional
information that is lost by symmetrization. The imaginary components capture
asymmetry introduced by portals.
