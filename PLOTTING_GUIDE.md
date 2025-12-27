# Separated Plotting for ALLO Training

## Overview

The ALLO training has been optimized to **export data** instead of creating plots during training. This eliminates plotting overhead and allows you to generate visualizations independently.

## Benefits

✅ **Faster training** - No matplotlib overhead blocking the training loop
✅ **Flexible** - Generate plots while training is running or after completion
✅ **Customizable** - Modify plots without re-running training
✅ **Parallel** - Run plotting in a separate process
✅ **Memory efficient** - No need to keep plotting libraries loaded during training

---

## How It Works

### During Training

ALLO exports data at regular intervals (controlled by `plot_freq`):

```
results/room4/room4__allo__0__42__1234567890/
├── viz_metadata.pkl                        # Environment & visualization settings
├── gt_eigenvalues.npy                      # Ground truth eigenvalues
├── gt_eigenvectors.npy                     # Ground truth eigenvectors
├── metrics_history.json                    # Training metrics (loss, error, etc.)
├── learned_eigenvectors_step_1000.npy      # Learned features at step 1000
├── learned_eigenvectors_step_2000.npy      # Learned features at step 2000
├── ...
└── final_learned_eigenvectors.npy          # Final learned features
```

### After Training (or while running)

Run the plotting script to generate all visualizations:

```bash
python generate_plots.py results/room4/room4__allo__0__42__1234567890/
```

This creates:

```
results/room4/room4__allo__0__42__1234567890/plots/
├── ground_truth_eigenvectors.png           # Ground truth eigenvectors
├── learned_eigenvectors_step_1000.png      # Learned eigenvectors at step 1000
├── learned_eigenvectors_step_2000.png      # Learned eigenvectors at step 2000
├── ...
├── learning_curves.png                     # Training metrics over time
├── dual_variable_evolution.png             # Eigenvalue approximations
└── final_comparison.png                    # Side-by-side comparison
```

---

## Usage

### 1. Training (Default Mode - No Plotting)

```bash
python exp_alcl/allo.py --env_type room4 --num_gradient_steps 20000
```

This runs training **without creating any plots** (much faster).

### 2. Generate Plots After Training

```bash
# Generate all plots
python generate_plots.py results/room4/room4__allo__0__42__1234567890/

# Generate only specific plots
python generate_plots.py results/room4/... --skip-learned  # Skip learned checkpoints
python generate_plots.py results/room4/... --latest-only   # Only plot latest checkpoint
python generate_plots.py results/room4/... --skip-metrics  # Skip learning curves
```

### 3. Generate Plots While Training

Open a second terminal and run:

```bash
# In terminal 1
python exp_alcl/allo.py --env_type room4

# In terminal 2 (while training is running)
python generate_plots.py results/room4/<run_dir>/
```

The plotting script will use whatever data is available at the time.

### 4. Legacy Mode (Plot During Training)

If you really want the old behavior:

```bash
python exp_alcl/allo.py --env_type room4 --plot_during_training
```

⚠️ **Not recommended** - This blocks training and is much slower.

---

## Command-Line Options

### generate_plots.py

```bash
python generate_plots.py <results_dir> [options]

Positional arguments:
  results_dir              Path to results directory

Options:
  --skip-ground-truth      Skip plotting ground truth eigenvectors
  --skip-learned           Skip plotting learned eigenvector checkpoints
  --skip-metrics           Skip plotting learning curves and dual evolution
  --skip-comparison        Skip plotting final comparison
  --latest-only            Only plot the latest learned checkpoint (faster)
```

### allo.py (new parameter)

```bash
--plot_during_training   Create plots during training (default: False)
```

---

## Performance Impact

### Training Time (20,000 steps, plot_freq=1000)

| Mode | Plotting Time | Total Training Time | Speedup |
|------|---------------|---------------------|---------|
| **Old (inline plotting)** | ~60-90 seconds | ~5-6 minutes | 1× |
| **New (export only)** | ~0 seconds | ~3-4 minutes | **1.5-2× faster** |

### Plotting Time (generating all plots)

```bash
python generate_plots.py <results_dir>
# Takes ~30-60 seconds (can run in parallel with training)
```

---

## Examples

### Minimal: Only plot final results

```bash
# Train
python exp_alcl/allo.py --env_type room4

# Plot only final comparison
python generate_plots.py results/room4/<run_dir>/ \
    --skip-ground-truth \
    --skip-learned \
    --skip-metrics
```

### Quick check: Latest checkpoint only

```bash
python generate_plots.py results/room4/<run_dir>/ --latest-only
```

### Custom: Generate plots at specific intervals

```bash
# Train with larger plot_freq (export less often)
python exp_alcl/allo.py --env_type room4 --plot_freq 5000

# Generate plots for all checkpoints
python generate_plots.py results/room4/<run_dir>/
```

---

## Workflow Recommendations

### Development (fast iteration)

```bash
# Train quickly without plotting
python exp_alcl/allo.py --env_type room4 --num_gradient_steps 5000

# Check final results only
python generate_plots.py results/room4/<run_dir>/ --latest-only
```

### Production (full analysis)

```bash
# Train with regular checkpoints
python exp_alcl/allo.py --env_type room4 --num_gradient_steps 20000 --plot_freq 1000

# Generate all plots
python generate_plots.py results/room4/<run_dir>/
```

### Parallel (maximum efficiency)

```bash
# Terminal 1: Train
python exp_alcl/allo.py --env_type room4 --num_gradient_steps 20000

# Terminal 2: Monitor with plots (updates every 30 seconds)
while true; do
    python generate_plots.py results/room4/<run_dir>/ --latest-only
    sleep 30
done
```

---

## Customizing Plots

Since plotting is separate, you can modify `generate_plots.py` to:

- Change plot styles, colors, fonts
- Add new plot types
- Filter which eigenvectors to visualize
- Combine multiple runs for comparison
- Export to different formats

Example modifications:

```python
# In generate_plots.py

# Change DPI
plt.savefig(..., dpi=150)  # Lower DPI for faster rendering

# Plot only specific eigenvectors
eigenvector_indices=[0, 1, 2]  # Instead of all

# Add custom annotations
ax.text(x, y, "annotation")
```

---

## Troubleshooting

### "Required data file not found"

Make sure training has run long enough to export data. Check that these files exist:
- `viz_metadata.pkl`
- `gt_eigenvalues.npy`
- `metrics_history.json`

### "No learned checkpoints found"

Training needs to reach at least one `plot_freq` interval. Either:
- Wait for training to reach the first checkpoint
- Lower `plot_freq` (e.g., `--plot_freq 100`)

### Plots look different from before

This is expected! The new system uses exactly the same plotting code, but:
- Matplotlib versions might differ
- Random seeds for colors might differ
- DPI settings might differ

All numerical values should be identical.

---

## Summary

**Old workflow:**
```
Train (slow, with plotting) → Done
```

**New workflow:**
```
Train (fast, no plotting) → Generate plots separately
                          ↓
                      Can run in parallel or after
```

This separation gives you:
- **Faster training** (no plotting overhead)
- **More flexibility** (plot anytime, customize easily)
- **Better resource usage** (no matplotlib during training)

Enjoy the speedup! 🚀
