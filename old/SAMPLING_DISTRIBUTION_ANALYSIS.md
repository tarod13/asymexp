# Sampling Distribution Analysis

## Question
Does the effective state distribution take into account terminal states?

## How the Sampling Distribution is Computed

### 1. Transition Count Collection

In `src/data_collection.py:289-478`, the function `collect_transition_counts_and_episodes()` collects transition counts by:

1. Running parallel environments for a fixed number of steps
2. At each step:
   - Taking a random action from the current state
   - Recording the transition: `(state, action, next_state)`
   - Checking if `next_state` is terminal (`done=True`)
   - If terminal, **immediately resetting** the environment to initial state

**Key code** (`data_collection.py:354-359`):
```python
next_env_states, state_indices, next_state_indices, rewards, dones, _ = vmap_step(num_envs)(
    rng_step, env_states, action_array
)

# Accumulate transition counts for all parallel environments
transition_counts = transition_counts.at[state_indices, action_array, next_state_indices].add(1.0)
```

Then the environment is conditionally reset (`line 423`):
```python
reset_env_states = vmap_reset(num_envs)(rng, next_env_states, dones)
```

The `vmap_reset` function (`data_collection.py:31-42`) resets the environment **if and only if** `done=True`.

### 2. Sampling Distribution Computation

From the collected `transition_counts` array of shape `[num_states, num_actions, num_states]`:
- Axis 0: source state
- Axis 1: action
- Axis 2: destination state

The sampling distribution is computed (`allo_complex.py:167-194`):
```python
def compute_sampling_distribution(transition_counts):
    # Sum over actions and next states to get visit counts per state
    state_visit_counts = jnp.sum(transition_counts, axis=(1, 2))

    # Normalize to get probabilities
    total_visits = jnp.sum(state_visit_counts)
    sampling_probs = state_visit_counts / total_visits

    # Create diagonal matrix
    D = jnp.diag(sampling_probs)
    return D
```

**Important**: `state_visit_counts[s]` = **number of times we STARTED from state `s`** (not ended in state `s`).

## Terminal States Behavior

### When Terminal States Occur

From `src/envs/gridworld.py:169-179`, terminal states occur when:
1. The agent reaches a goal position (if `has_goal=True`)
2. The maximum number of steps is reached (`max_steps`)

### What Happens at Terminal States

When `terminal=True` in the environment state (`gridworld.py:162-167`):
```python
# Either take action or keep same state if terminal
position, steps = jax.lax.cond(
    terminal,
    lambda args: args,  # If terminal, stay in place
    _step_impl,         # If not terminal, apply step function
    (position, steps)
)
```

The agent **stays in place** (self-loop) when trying to take actions from terminal states.

### How Terminal States Are Handled in Data Collection

**Critical observation**: Terminal states trigger **immediate reset**, so:

1. ✅ **Transitions TO terminal states are recorded**
   - Example: `(state_42, action_3, terminal_state_7)` is recorded

2. ❌ **Transitions FROM terminal states are NOT recorded**
   - The environment resets immediately after reaching terminal
   - Next step starts from the initial state, not the terminal state

3. **Result**: Terminal states have **zero outgoing transitions**
   - `state_visit_counts[terminal_state] = 0`
   - `sampling_probs[terminal_state] = 0`

## Impact on Eigenvector Normalization

### The Mathematical Issue

The learned left eigenvectors are eigenvectors of the adjoint with respect to the weighted inner product:
```
<u, v>_D = u^T D v
```

To convert adjoint eigenvectors to true left eigenvectors, we scale by `D` (the sampling distribution diagonal matrix).

**However**, if `D[terminal_state, terminal_state] = 0`, then:
- Multiplying by `D` would zero out the terminal state components
- This could be problematic if terminal states should have non-zero left eigenvector components

### Which Environments Have Terminal States?

Looking at `src/envs/env.py:6-95`, the example environments include:

**Environments WITH goals** (have terminal states):
- `"empty"`: Has goal 'G'
- `"obstacles"`: Has goal 'G'
- `"maze"`: Has goal 'G'
- `"spiral"`: Has goal 'G'

**Environments WITHOUT goals** (no terminal states except max_steps timeout):
- `"room4"`: No goal marker
- `"room4_doors"`: No goal marker
- `"asymmetric_maze"`: No goal marker

### Current Implementation Status

For environments **without goals** (like `room4`):
- Terminal states only occur at `max_steps` timeout
- These are artificial boundaries, not part of the MDP structure
- The sampling distribution correctly excludes these artificial terminal states
- ✅ **This is appropriate behavior**

For environments **with goals**:
- Goal states ARE part of the MDP structure
- They should have non-zero stationary distribution
- BUT: immediate reset means they get `sampling_probs = 0`
- ❌ **This could be problematic**

## Recommendations

### 1. For Goal-Free Environments (Current Use Case)
The current implementation is **correct**:
- No structural terminal states
- Only max_steps timeout creates artificial terminals
- These should NOT be included in the stationary distribution
- ✅ **No changes needed**

### 2. For Environments With Goals (Future Work)
If using environments with goal states, consider:

**Option A**: Modify data collection to include absorbing state behavior
- Don't reset immediately at terminal
- Allow transitions from terminal to itself
- This gives terminal states non-zero sampling probability

**Option B**: Compute stationary distribution analytically
- Use transition matrix to compute true stationary distribution
- Don't rely on empirical sampling distribution
- This requires knowing the MDP structure

**Option C**: Use continuing tasks (remove terminal states)
- Replace goals with rewards
- Make all states non-terminal
- This is more standard for representation learning

### 3. Verification
To check if your current runs are affected:
```python
# Check if any states have zero sampling probability
zero_prob_states = jnp.where(sampling_probs == 0)[0]
if len(zero_prob_states) > 0:
    print(f"Warning: {len(zero_prob_states)} states have zero sampling probability")
    print(f"These states: {zero_prob_states}")
```

## Conclusion

**For your current use case** (appears to be `room4` or similar goal-free environments):
- ✅ Terminal states (from max_steps) are correctly **excluded** from sampling distribution
- ✅ The normalization procedure is appropriate
- ✅ No changes needed

**For future use** with goal-based environments:
- ⚠️ May need to reconsider how terminal/absorbing states are handled
- The current implementation would give goal states zero probability
- This might not align with the theoretical stationary distribution
