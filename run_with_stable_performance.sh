#!/bin/bash
# Script to run training with stable CPU performance settings

# Force XLA to use consistent compilation settings for CPU
export XLA_FLAGS="
--xla_force_host_platform_device_count=1
--xla_cpu_multi_thread_eigen=false
"

# Set consistent CPU thread usage (adjust based on your CPU cores)
# Use 4-8 threads for best performance/consistency tradeoff
export XLA_CPU_JIT_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# Disable CPU frequency scaling (requires root, comment out if not available)
# sudo cpupower frequency-set --governor performance

# Pin to performance cores if hybrid architecture
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Clear JAX compilation cache to start fresh
echo "Clearing JAX cache..."
rm -rf ~/.cache/jax/* 2>/dev/null || true

# Print settings
echo "CPU Performance settings:"
echo "  XLA_FLAGS: $XLA_FLAGS"
echo "  Thread counts: $OMP_NUM_THREADS"
echo "  CPU affinity: $OMP_PROC_BIND, $OMP_PLACES"
echo ""

# Run the training script with provided arguments
python -m exp_alcl.allo "$@"

