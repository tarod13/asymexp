# Results Storage and Plotting Guide for allo_complex.py

## Results Directory Structure

When you run `allo_complex.py`, results are saved to:
```
./results/<env_type>/<env_type>__allo_complex__<exp_number>__<seed>__<timestamp>/
```

For example:
```
./results/room4/room4__allo_complex__0__42__1735502400/
├── plots/                          # Visualizations (if plot_during_training=True)
├── models/                         # Model checkpoints
├── args.json                       # Training configuration
├── metrics_history.json            # All training metrics
├── gt_eigenvalues*.npy            # Ground truth eigenvalues (3 versions)
├── gt_eigenvectors*.npy           # Ground truth eigenvectors (3 versions)
├── state_coords.npy               # Normalized (x,y) coordinates
├── sampling_distribution.npy      # Empirical state visitation
├── viz_metadata.pkl               # Grid dimensions, canonical states
├── door_config.pkl                # Door configuration (if use_doors=True)
├── latest_learned_*.npy           # Latest learned eigenvectors (4 files)
└── final_learned_*.npy            # Final learned eigenvectors (4 files)
```

## Saved Files for Complex Eigenvectors

### Ground Truth (computed once at start)

**Three Laplacian versions** are computed for comparison:

1. **Simple Laplacian**: `L = I - (1-γ)(SR_γ + SR_γ^T)/2`
   - `gt_eigenvalues_simple.npy`
   - `gt_eigenvectors_simple.npy`

2. **Weighted Laplacian**: `L = D - (1-γ)(DSR_γ + SR_γ^T D^T)/2`
   - `gt_eigenvalues_weighted.npy`
   - `gt_eigenvectors_weighted.npy`

3. **Inverse-Weighted Laplacian** (main baseline): `L = I - (1-γ)(SR_γ + D^{-1}SR_γ^T D)/2`
   - `gt_eigenvalues.npy` (or `gt_eigenvalues_invweighted.npy`)
   - `gt_eigenvectors.npy` (or `gt_eigenvectors_invweighted.npy`)

### Learned Features (Complex Eigenvectors)

**Latest snapshots** (updated every `plot_freq` steps):
- `latest_learned_left_real.npy` - Left eigenvectors φ, real part
- `latest_learned_left_imag.npy` - Left eigenvectors φ, imaginary part
- `latest_learned_right_real.npy` - Right eigenvectors ψ, real part
- `latest_learned_right_imag.npy` - Right eigenvectors ψ, imaginary part

**Final results** (saved at end of training):
- `final_learned_left_real.npy` - Final left eigenvectors φ, real part
- `final_learned_left_imag.npy` - Final left eigenvectors φ, imaginary part
- `final_learned_right_real.npy` - Final right eigenvectors ψ, real part
- `final_learned_right_imag.npy` - Final right eigenvectors ψ, imaginary part

Each file has shape `[num_canonical_states, num_eigenvectors]`.

### Metrics

**`metrics_history.json`** contains:
- `gradient_step`: Training iteration
- `allo`: Total loss value
- `graph_loss`: Graph drawing loss
- `dual_loss`, `dual_loss_neg`: Dual variable losses
- `barrier_loss`: Barrier penalty
- `imag_part_penalty`: Penalty for Im(φ^T ψ) ≠ 0
- `total_error`: Sum of biorthogonality errors
- `cosine_sim_*_invweighted`: Cosine similarities vs inverse-weighted Laplacian
- `cosine_sim_*_simple`: Cosine similarities vs simple Laplacian
- `cosine_sim_*_weighted`: Cosine similarities vs weighted Laplacian
- `dual_0`, `dual_1`, ...: Learned eigenvalue estimates

## Plotting Modes

### Mode 1: Live Plotting (Slow)
Set `--plot_during_training=True` to generate plots during training.

**Plots created:**
1. **Ground truth eigenvectors** (created once at start):
   - `plots/ground_truth_eigenvectors_simple.png`
   - `plots/ground_truth_eigenvectors_weighted.png`
   - `plots/ground_truth_eigenvectors_invweighted.png`
   - `plots/sampling_distribution.png`

2. **Learning progress** (updated every `plot_freq` steps):
   - `plots/learned_eigenvectors_latest.png` - Current learned eigenvectors
   - `plots/learning_curves.png` - Loss curves over time
   - `plots/dual_variable_evolution.png` - Eigenvalue estimates vs ground truth
   - `plots/cosine_similarity_evolution.png` - Similarity metrics over time

3. **Final comparison** (created at end):
   - `plots/final_comparison.png` - Side-by-side GT vs learned

**Visualization details:**
- Uses `visualize_multiple_eigenvectors()` from `exp_complex_basis.eigenvector_visualization`
- Displays both real and imaginary components
- For complex eigenvectors, can show:
  - `eigenvector_type='right'` or `'left'`
  - `component='real'` or `'imag'`

### Mode 2: Data Export Only (Fast, Default)
Set `--plot_during_training=False` (default).

**Behavior:**
- Saves all `.npy` files and `metrics_history.json`
- Does NOT create plots during training
- Much faster - avoids matplotlib overhead
- Prints: `"Data exported. Use generate_plots.py to create visualizations."`

**To create plots later:**
You can use a separate plotting script (see below).

## Creating Plots After Training

### Option 1: Use generate_plots_complex.py
```bash
python generate_plots_complex.py ./results/room4/room4__allo_complex__0__42__1735502400
```

This script will generate all plots from the saved data:
- Ground truth eigenvectors (left/right, real/imag)
- Latest learned eigenvectors (all 4 components)
- Final learned eigenvectors (all 4 components)
- Learning curves
- Dual variable evolution
- Cosine similarity evolution (left and right separately)
- Sampling distribution
- Final comparison plots

### Option 2: Manual plotting with saved data

Here's a simple example to plot the learned eigenvectors:

```python
import numpy as np
import matplotlib.pyplot as plt
from exp_complex_basis.eigenvector_visualization import visualize_multiple_eigenvectors

# Load learned eigenvectors
results_dir = "./results/room4/room4__allo_complex__0__42__1735502400"

left_real = np.load(f"{results_dir}/final_learned_left_real.npy")
left_imag = np.load(f"{results_dir}/final_learned_left_imag.npy")
right_real = np.load(f"{results_dir}/final_learned_right_real.npy")
right_imag = np.load(f"{results_dir}/final_learned_right_imag.npy")

# Load metadata
import pickle
with open(f"{results_dir}/viz_metadata.pkl", 'rb') as f:
    viz_metadata = pickle.load(f)

# Create eigendecomposition dict for visualization
eigendecomp = {
    'eigenvalues': np.zeros(right_real.shape[1], dtype=np.complex64),
    'eigenvalues_real': np.zeros(right_real.shape[1]),
    'eigenvalues_imag': np.zeros(right_real.shape[1]),
    'right_eigenvectors_real': right_real,
    'right_eigenvectors_imag': right_imag,
    'left_eigenvectors_real': left_real,
    'left_eigenvectors_imag': left_imag,
}

# Visualize right eigenvectors (real parts)
visualize_multiple_eigenvectors(
    eigenvector_indices=list(range(10)),
    eigendecomposition=eigendecomp,
    canonical_states=viz_metadata['canonical_states'],
    grid_width=viz_metadata['grid_width'],
    grid_height=viz_metadata['grid_height'],
    eigenvector_type='right',
    component='real',
    ncols=4,
    save_path='learned_right_real.png',
)
plt.show()

# Visualize right eigenvectors (imaginary parts)
visualize_multiple_eigenvectors(
    eigenvector_indices=list(range(10)),
    eigendecomposition=eigendecomp,
    canonical_states=viz_metadata['canonical_states'],
    grid_width=viz_metadata['grid_width'],
    grid_height=viz_metadata['grid_height'],
    eigenvector_type='right',
    component='imag',
    ncols=4,
    save_path='learned_right_imag.png',
)
plt.show()
```

## Key Differences from Original allo.py

### Saved Files
- **allo.py**: Saves single `final_learned_eigenvectors.npy`
- **allo_complex.py**: Saves 4 separate files (left/right × real/imag)

### Visualization
- **allo.py**: Uses symmetric eigenvectors (left = right, imag = 0)
- **allo_complex.py**: Can visualize all 4 components separately

### Metrics
- **allo.py**: No `imag_part_penalty` in metrics
- **allo_complex.py**: Tracks `imag_part_penalty` for Im(φ^T ψ)

## Tips

1. **For quick iteration**: Use `--plot_during_training=False` to avoid plotting overhead
2. **For monitoring**: Use `--plot_during_training=True` to watch learning in real-time
3. **For analysis**: Load `.npy` files and create custom plots
4. **Resume training**: Use `--resume_from=<results_dir>` to continue from checkpoint

## Example Usage

```bash
# Fast training (no plots)
python exp_alcl/allo_complex.py \
    --env_type room4 \
    --num_eigenvectors 10 \
    --num_gradient_steps 20000 \
    --plot_during_training=False

# Training with live plots
python exp_alcl/allo_complex.py \
    --env_type room4 \
    --num_eigenvectors 10 \
    --num_gradient_steps 20000 \
    --plot_during_training=True \
    --plot_freq 500
```
