# Experiment 02: Eigendecomposition of Irreversible Door Environments

Tests asymmetric transition dynamics created by one-way passages (irreversible
doors) in grid environments. Doors block the reverse transition between adjacent
cells, creating asymmetry in a more physically interpretable way than portals.

## What it does

1. Creates multiple grid environments with randomly placed irreversible doors
2. Collects transition data (the agent experiences blocked transitions)
3. Computes eigendecomposition of the non-symmetrized transition matrices
4. Computes hitting times and analyzes asymmetry
5. Visualizes eigenvectors and hitting time asymmetry with door markers

## How doors differ from portals

| Property          | Doors                           | Portals                     |
|-------------------|---------------------------------|-----------------------------|
| Affected states   | Adjacent cells only             | Any pair of cells           |
| Mechanism         | Block reverse transition        | Create new teleportation    |
| Physical analogy  | One-way valve                   | Wormhole                    |
| Interpretability  | High (local, intuitive)         | Lower (non-local)           |

## Usage

```bash
python experiments/02/run_analysis.py --num-envs 10 --num-doors 5 --k 20
```

## Key finding

Irreversible doors create asymmetric transition matrices where the hitting time
from A to B differs from B to A. The complex eigenvectors of these matrices
capture this directional information through their imaginary components.
