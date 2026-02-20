# Analysis Script Examples

This file contains common usage examples for the `scripts/run_analysis.sh` script.

## Portal Environment Analysis (Experiment 01)

### Basic run with default parameters
```bash
./scripts/run_analysis.sh portal
```

### Generate more environments and eigenvectors
```bash
./scripts/run_analysis.sh portal --num-envs 50 --k 40 --num-portals 15
```

### Custom output directory
```bash
./scripts/run_analysis.sh portal --output-dir ./results/portal_custom --seed 123
```

### Full parameter example
```bash
./scripts/run_analysis.sh portal \
    --base-env GridRoom-4 \
    --num-envs 100 \
    --num-portals 20 \
    --num-rollouts 200 \
    --num-steps 150 \
    --k 50 \
    --output-dir ./results/portal_large \
    --seed 42 \
    --num-eigenvectors 10 \
    --wall-color gray
```

## Door Environment Analysis (Experiment 02)

### Basic run with default parameters
```bash
./scripts/run_analysis.sh door
```

### More doors and environments
```bash
./scripts/run_analysis.sh door --num-doors 20 --num-envs 100
```

### Enable logarithmic scale for hitting time visualization
```bash
./scripts/run_analysis.sh door --num-doors 10 --log-scale
```

### Full parameter example
```bash
./scripts/run_analysis.sh door \
    --base-env GridRoom-4 \
    --num-envs 50 \
    --num-doors 15 \
    --num-rollouts 200 \
    --num-steps 150 \
    --k 40 \
    --output-dir ./results/door_large \
    --seed 42 \
    --num-eigenvectors 10 \
    --num-targets 8 \
    --log-scale
```

## Sweep Analysis

### Analyze a specific sweep
```bash
./scripts/run_analysis.sh sweep \
    --exp_name batch_lr_sweep \
    --results_dir ./results/sweeps \
    --env_type file
```

### Analyze with custom output directory
```bash
./scripts/run_analysis.sh sweep \
    --exp_name my_sweep \
    --results_dir ./results/sweeps \
    --output_dir ./results/sweeps/analysis/my_sweep_v2
```

## Using with SLURM

When running analysis from SLURM jobs, set the `ASYMEXP_ROOT` environment variable to point to your project directory:

```bash
#!/bin/bash
#SBATCH --job-name=sweep_analysis
#SBATCH --output=analysis_%j.log

export ASYMEXP_ROOT=/home/user/asymexp

$ASYMEXP_ROOT/scripts/run_analysis.sh sweep \
    --exp_name batch_lr_sweep \
    --results_dir $ASYMEXP_ROOT/results/sweeps
```

Or export it in your `.bashrc` or job submission script:
```bash
export ASYMEXP_ROOT=/home/user/asymexp
sbatch your_analysis_job.sh
```

## Tips

1. **View all options**: Run `./scripts/run_analysis.sh --help` to see all available options

2. **Default values**: If you don't specify parameters, sensible defaults are used:
   - num-envs: 10
   - k: 20
   - num-portals/num-doors: 10/5
   - num-rollouts: 100
   - num-steps: 100
   - seed: 42

3. **SLURM usage**: Always set `ASYMEXP_ROOT` when running from SLURM jobs, as the script gets copied to a temporary job directory

4. **Output location**: Results are saved to:
   - Portal: `experiments/01/results/` (or custom --output-dir)
   - Door: `experiments/02/results/` (or custom --output-dir)
   - Sweep: `results/sweeps/analysis/{exp_name}/` (or custom --output-dir)

5. **Visualizations**: All analysis scripts generate:
   - Eigenvalue/eigenvector visualizations
   - Summary statistics
   - Plots saved as PNG files in the output directory

6. **Progress tracking**: The script shows colored output:
   - Green: Success messages
   - Yellow: Progress indicators
   - Red: Error messages
