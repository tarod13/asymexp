# Running Analysis on Compute Canada

## Overview

There are two ways to run analysis scripts:

1. **Local/Interactive**: `run_analysis.sh` (for quick tests on login node)
2. **SLURM Jobs**: `submit_analysis.sh` (for production runs on compute nodes)

## SLURM Submission (Recommended)

Use `submit_analysis.sh` to run analysis as a SLURM job to avoid OOM errors:

```bash
# Portal analysis (experiment 01)
sbatch scripts/submit_analysis.sh portal --num-envs 20 --k 30

# Door analysis (experiment 02)
sbatch scripts/submit_analysis.sh door --num-doors 10 --num-envs 50

# Sweep analysis
sbatch scripts/submit_analysis.sh sweep --exp_name batch_lr_sweep
```

### Custom Resources

Override default resources as needed:

```bash
# More memory for large sweeps
sbatch --mem=32G scripts/submit_analysis.sh sweep --exp_name batch_lr_sweep

# Longer time limit
sbatch --time=8:00:00 scripts/submit_analysis.sh portal --num-envs 100

# More CPUs for parallel processing
sbatch --cpus-per-task=8 scripts/submit_analysis.sh door --num-envs 100
```

### Default Resources

- **Time**: 4 hours
- **Memory**: 16GB
- **CPUs**: 4
- **Account**: aip-machado

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f logs/analysis_<job_id>.out
tail -f logs/analysis_<job_id>.err

# Cancel a job
scancel <job_id>
```

## Local/Interactive Runs

For quick tests on the login node (use sparingly):

```bash
# Portal analysis
./scripts/run_analysis.sh portal --num-envs 5 --k 10

# Door analysis
./scripts/run_analysis.sh door --num-doors 3 --num-envs 10

# Sweep analysis
./scripts/run_analysis.sh sweep --exp_name test_sweep
```

⚠️ **Warning**: Don't run large analyses on login nodes - they may OOM or violate usage policies.

## Analysis Types

### Portal Analysis (Experiment 01)

Eigendecomposition analysis on portal environments.

```bash
sbatch scripts/submit_analysis.sh portal \
    --num-envs 20 \
    --num-portals 10 \
    --k 30 \
    --output-dir ./results/portal_analysis
```

**Options**:
- `--num-envs`: Number of environments (default: 10)
- `--num-portals`: Portals per environment (default: 10)
- `--k`: Number of eigenvectors (default: 20)
- `--num-rollouts`: Rollouts per environment (default: 100)
- `--num-steps`: Steps per rollout (default: 100)
- `--seed`: Random seed (default: 42)

### Door Analysis (Experiment 02)

Eigendecomposition analysis on environments with irreversible doors.

```bash
sbatch scripts/submit_analysis.sh door \
    --num-envs 50 \
    --num-doors 10 \
    --k 30 \
    --log-scale
```

**Options**:
- `--num-envs`: Number of environments (default: 10)
- `--num-doors`: Doors per environment (default: 5)
- `--k`: Number of eigenvectors (default: 20)
- `--log-scale`: Use logarithmic scale for hitting times

### Sweep Analysis

Analyze batch size × learning rate sweep results.

```bash
sbatch scripts/submit_analysis.sh sweep \
    --exp_name batch_lr_sweep \
    --results_dir ./results/sweeps
```

**Options**:
- `--exp_name`: Experiment name (default: batch_lr_sweep)
- `--results_dir`: Results directory (default: ./results/sweeps)
- `--env_type`: Environment type (default: file)

## Dependencies

Analysis scripts require:
- **Modules**: `StdEnv/2023`, `gcc/14.3`, `python/3.11`, `scipy-stack/2024b`
- **Virtual env**: `~/ENV` with JAX, numpy, matplotlib

These are automatically loaded by both `run_analysis.sh` and `submit_analysis.sh`.

## Troubleshooting

### OOM Errors

**Problem**: Out of memory errors when running analysis.

**Solution**: Use `submit_analysis.sh` with more memory:
```bash
sbatch --mem=32G scripts/submit_analysis.sh <analysis_type> <options>
```

### Import Errors (JAX, numpy, etc.)

**Problem**: `ModuleNotFoundError: No module named 'jax'`

**Solution**: Ensure virtual environment is set up:
```bash
bash scripts/setup_remote.sh  # or hard_setup_remote.sh
```

### Path Errors

**Problem**: `Cannot find project root`

**Solution**: Set `ASYMEXP_ROOT`:
```bash
export ASYMEXP_ROOT=$HOME/asymexp
sbatch scripts/submit_analysis.sh <analysis_type> <options>
```

(This is set automatically in `submit_analysis.sh`)

## Examples

```bash
# Quick portal test (small, on login node)
./scripts/run_analysis.sh portal --num-envs 5 --k 10

# Production portal run (large, on compute node)
sbatch --mem=24G scripts/submit_analysis.sh portal --num-envs 100 --k 50

# Sweep analysis with longer time limit
sbatch --time=12:00:00 scripts/submit_analysis.sh sweep --exp_name batch_lr_sweep

# Multiple door analysis jobs
for n in 5 10 15 20; do
    sbatch scripts/submit_analysis.sh door --num-doors $n --num-envs 50
done
```
