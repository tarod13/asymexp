#!/bin/bash
# Setup script for RL library on Compute Canada

# Force clear all modules and load required system modules
module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 scipy-stack/2024b  mujoco/3.3.0

# Create environment only if it doesn't exist
if [ ! -d ~/ENV ]; then
    virtualenv --no-download ~/ENV
fi

source ~/ENV/bin/activate

# Update pip and install core ML packages
pip install --upgrade pip

# Install packages
pip install jax flax optax tensorboard

# Install brax with --no-deps to avoid mujoco version conflicts, then install missing deps manually
pip install --no-deps brax

# Install brax dependencies manually (excluding mujoco which we have from module)
pip install etils flask flask-cors jaxopt jinja2 ml-collections mujoco-mjx==3.3.0 tensorboardx trimesh flashbax

# Install additional utility packages
pip install tqdm tyro

# Prevent JAX from grabbing all GPU memory at once
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Verify libraries are working
python -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')"
python -c "import jax; print(f'JAX version: {jax.__version__}, JAX backend: {jax.default_backend()}')"
python -c "import brax; print(f'Brax version: {brax.__version__}')"

echo "Setup complete!"