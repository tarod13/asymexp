# Compute Canada Setup Scripts

Quick reference for setting up and using the RL environment on Compute Canada.

## Files

- **`hard_setup_remote.sh`** - Fresh install (clears existing environment)
- **`setup_remote.sh`** - Regular setup (preserves existing environment if present)
- **`activate_remote.sh`** - Activates the environment for each session
- **`submit_job_XYZ.sh`** - SLURM job submission script

## Usage

### First Time Setup

```bash
bash hard_setup_remote.sh
```

Run this once to create the environment from scratch. Takes a few seconds.

### Regular Setup (if needed)

```bash
bash setup_remote.sh
```

Use this to reinstall packages, but keeping the existing environment.

### Daily Use - Activating the Environment

**IMPORTANT:** Always use `source` or `.`, not `bash`:

```bash
source activate_remote.sh
```

or

```bash
. activate_remote.sh
```

Run this every time Python/JAX/Brax is needed.

## Quick Check

After sourcing the activation script, the following is a simple check to verify it worked:

```bash
which python        # Should show: ~/ENV/bin/python
python -c "import brax; print('✓ Brax loaded')"
```

## Common Mistake

❌ `bash activate_remote.sh` - Doesn't work (runs in subshell)  
✓ `source activate_remote.sh` - Works correctly

## Typical Workflows

```bash
# SSH into Compute Canada
ssh your-username@cluster.computecanada.ca

# Activate environment
source activate_remote.sh

# Run code
python rl_script.py
```

```bash
# SSH into Compute Canada
ssh vulcan

# Activate environment
. activate_remote.sh

# Run code
python rl_script.py
```

### Batch Job Submission

```bash
sbatch submit_job_XYZ.sh

# Check job status
squeue -u $USER

# View logs (while running or after completion)
tail -f logs/XYZ.out
```