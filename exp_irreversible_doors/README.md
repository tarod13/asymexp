# Irreversible Doors Experiment

This experiment studies eigendecomposition of asymmetric transition matrices created by **irreversible doors** - one-way passages in grid environments.

## Concept

### What are Irreversible Doors?

An irreversible door is a one-way passage between adjacent grid cells:
- **Forward direction (allowed)**: From state `s`, action `a` leads to state `s'` with probability 1
- **Reverse direction (blocked)**: From state `s'`, the reverse action `a'` cannot lead back to `s`

### Mathematical Definition

For a door at `(s, a) -> s'`:
- Normal transition: `p(s' | s, a) = 1`
- Blocked reverse: `p(s | s', a_reverse) = 0`

Where `a_reverse` is the opposite action (e.g., if `a=up`, then `a_reverse=down`).

### Difference from Portals

| Feature | Portals | Doors |
|---------|---------|-------|
| Target | Arbitrary state | Adjacent state only |
| Physics | Teleportation | Normal movement |
| Asymmetry | Creates new connections | Blocks existing connections |
| Interpretability | Less intuitive | More intuitive |

## Why This Matters

Irreversible doors create **asymmetric transition matrices** in a more interpretable way than portals:
1. **Local asymmetry**: Only affects neighboring states
2. **Physical intuition**: Like real one-way doors
3. **Clearer eigenvectors**: Easier to understand left vs right eigenvector differences

## Usage

```bash
# Basic usage with 5 doors per environment
python exp_irreversible_doors/run_analysis.py --num-doors 5

# Custom configuration
python exp_irreversible_doors/run_analysis.py \
    --num-envs 10 \
    --num-doors 10 \
    --num-rollouts 100 \
    --k 15 \
    --num-eigenvectors 12 \
    --nrows 3 \
    --ncols 4 \
    --wall-color lightblue
```

## Output

The analysis generates:
- `results/door_locations.png` - Map showing door locations and directions
- `results/visualizations/` - Eigenvector heatmaps overlaid on grid
  - Left and right eigenvectors (real and imaginary components)
  - Combined left-right comparison views
- `results/door_eigendecomposition_results.pkl` - Full analysis results
- `results/analysis_summary.txt` - Summary statistics

## Implementation Details

1. **Finding reversible transitions**: Identifies all `(s, a) -> s'` pairs where reverse transition exists
2. **Random selection**: Randomly selects `num_doors` transitions to make irreversible
3. **Transition modification**: Sets `transition_counts[s', a_reverse, s] = 0` for selected doors
4. **Eigendecomposition**: Computes both left and right eigenvectors of asymmetric matrix
5. **Visualization**: Shows eigenvector values on grid with door markers

## Example

For a GridRoom-4 environment with 5 doors:
```
State s: (2, 3)
Action a: right
State s': (3, 3)

Door effect:
✓ Can go right from (2,3) to (3,3)
✗ Cannot go left from (3,3) to (2,3)
```

This creates a one-way passage that contributes to matrix asymmetry.
