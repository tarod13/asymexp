#!/bin/bash

module --force purge
module load StdEnv/2023 gcc/14.3 python/3.11 cuda/12.9 mujoco/3.3.0
source ~/ENV/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Environment activated and ready to use!"