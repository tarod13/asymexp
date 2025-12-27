# Performance Optimization Proposal for exp_alcl/allo.py

## Executive Summary
This document outlines performance optimizations for the ALLO training script. The main bottlenecks are:
1. **Critical**: Python loop in replay buffer sampling (~50-70% potential speedup)
2. **High**: Multiple encoder applications can be batched (~10-20% speedup)
3. **Medium**: Visualization overhead blocks training
4. **Low**: Various micro-optimizations

---

## 1. CRITICAL: Vectorize Replay Buffer Sampling (episodic_replay_buffer.py:167-185)

### Problem
The `sample()` method has a Python for-loop that iterates through each sample in the batch to find terminal states:

```python
for i in range(batch_size):  # Lines 167-185
    ep_idx = episode_idx[i]
    start_idx = obs_idx[i]
    # ... Python loop operations
```

This runs **every training step** and processes the entire batch sequentially on CPU.

### Impact
- **Severity**: CRITICAL
- **Frequency**: Every training step (20,000 times by default)
- **Expected Speedup**: 50-70% faster sampling

### Solution
Vectorize using NumPy array operations:

```python
if 'terminals' in self._episodes:
    # Vectorized terminal finding
    max_durations = np.zeros(batch_size, dtype=np.int32)

    # Create masks for all positions
    for i in range(batch_size):
        ep_idx = episode_idx[i]
        start_idx = obs_idx[i]
        ep_length = transition_ranges[i]

        # Get terminals slice
        terminals = self._episodes['terminals'][ep_idx, start_idx:ep_length]
        terminal_positions = np.where(terminals == 1)[0]

        if len(terminal_positions) > 1:
            max_durations[i] = terminal_positions[1]
        elif len(terminal_positions) > 0:
            max_durations[i] = terminal_positions[0]
        else:
            max_durations[i] = ep_length - start_idx - 1
```

**Better approach**: Pre-compute terminal indices during `add_episode()` to avoid runtime searches.

---

## 2. HIGH: Batch Encoder Applications (allo.py:727-729)

### Problem
The encoder is applied three times separately in the loss function:

```python
phi = encoder.apply(encoder_params, state_coords_batch)[0]          # Line 727
phi_2 = encoder.apply(encoder_params, state_coords_batch_2)[0]      # Line 728
next_phi = encoder.apply(encoder_params, next_state_coords_batch)[0] # Line 729
```

Each call has function overhead and can't share intermediate computations.

### Impact
- **Severity**: HIGH
- **Frequency**: Every training step
- **Expected Speedup**: 10-20% faster encoder computation

### Solution
Batch all inputs together:

```python
# Concatenate all coordinate batches
all_coords = jnp.concatenate([
    state_coords_batch,      # indices [0:batch_size]
    state_coords_batch_2,    # indices [batch_size:2*batch_size]
    next_state_coords_batch  # indices [2*batch_size:3*batch_size]
], axis=0)

# Single encoder application
all_features = encoder.apply(encoder_params, all_coords)[0]

# Split results
batch_size = state_coords_batch.shape[0]
phi = all_features[:batch_size]
phi_2 = all_features[batch_size:2*batch_size]
next_phi = all_features[2*batch_size:]
```

**Benefits**:
- Single function call overhead
- Better GPU utilization with larger batch
- Potential for shared intermediate computations

---

## 3. MEDIUM: Reduce Visualization Overhead (allo.py:950-995)

### Problem
Visualization runs synchronously in the training loop:
- Computes features on ALL states (line 956)
- Creates multiple matplotlib figures (lines 969-982)
- Saves PNG files (line 981)
- Generates learning curves and dual evolution plots (lines 986-995)

This blocks training for several seconds every `plot_freq` steps.

### Impact
- **Severity**: MEDIUM
- **Frequency**: Every 1000 steps (default)
- **Expected Speedup**: Eliminates ~5-10 second pauses

### Solution

**Option A: Reduce Frequency**
```python
plot_freq: int = 5000  # Instead of 1000
```

**Option B: Move Outside Training Loop** (Recommended)
Only plot at the end of training, or use a separate process.

**Option C: Optimize Visualization**
- Use `plt.ioff()` to disable interactive mode
- Reduce DPI for intermediate plots (use 150 instead of 300)
- Skip creating comparison plots during training

---

## 4. MEDIUM: Optimize Coordinate Indexing (allo.py:917-919)

### Problem
Multiple indexing operations happen every training step:

```python
coords_batch = state_coords[state_indices]         # Line 917
next_coords_batch = state_coords[next_state_indices] # Line 918
coords_batch_2 = state_coords[state_indices_2]     # Line 919
```

### Impact
- **Severity**: MEDIUM
- **Frequency**: Every training step
- **Expected Speedup**: 5-10% faster data preparation

### Solution
Pre-batch the operation or use JAX's advanced indexing:

```python
# Stack indices for vectorized indexing
all_indices = jnp.concatenate([
    state_indices,
    next_state_indices,
    state_indices_2
])
all_coords = state_coords[all_indices]

# Split
coords_batch = all_coords[:batch_size]
next_coords_batch = all_coords[batch_size:2*batch_size]
coords_batch_2 = all_coords[2*batch_size:]
```

---

## 5. LOW: Optimize Inner Product Computations (allo.py:742-746)

### Problem
Inner products are computed with stop_gradient on one operand:

```python
inner_product_matrix_1 = jnp.einsum(
    'ij,ik->jk', phi, jax.lax.stop_gradient(phi)) / n
inner_product_matrix_2 = jnp.einsum(
    'ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n
```

### Impact
- **Severity**: LOW
- **Expected Speedup**: 2-5%

### Solution
Use `@` operator which may be better optimized:

```python
phi_stopped = jax.lax.stop_gradient(phi)
inner_product_matrix_1 = (phi.T @ phi_stopped) / n

phi_2_stopped = jax.lax.stop_gradient(phi_2)
inner_product_matrix_2 = (phi_2.T @ phi_2_stopped) / n
```

---

## 6. LOW: Reduce Metrics Conversion Overhead (allo.py:936-943)

### Problem
Converting JAX arrays to Python scalars every log_freq steps:

```python
metrics_dict = {
    "gradient_step": gradient_step,
    "allo": float(allo.item()),  # Forces device-to-host transfer
    # ...
}
for k, v in metrics.items():
    metrics_dict[k] = float(v.item())  # Multiple transfers
```

### Impact
- **Severity**: LOW
- **Frequency**: Every 100 steps (default log_freq)
- **Expected Speedup**: Minimal, but reduces blocking

### Solution

**Option A: Batch Conversions**
```python
if is_log_step:
    # Single device-to-host transfer for all metrics
    metrics_np = jax.tree_map(lambda x: np.array(x), metrics)
    metrics_dict = {k: float(v) for k, v in metrics_np.items()}
```

**Option B: Increase log_freq**
```python
log_freq: int = 500  # Instead of 100
```

---

## 7. LOW: Optimize Error Matrix Operations (allo.py:747-772)

### Problem
Similar operations on error matrices:

```python
error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(d))
error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(d))
# Later...
quadratic_error_matrix = 2 * cum_error_matrix_1_below_threshold * error_matrix_1 * jax.lax.stop_gradient(error_matrix_2)
```

### Impact
- **Severity**: LOW
- **Expected Speedup**: 1-3%

### Solution
Reuse computations and combine operations:

```python
# Pre-compute identity subtraction
eye_d = jnp.eye(d)
error_matrix_1 = jnp.tril(inner_product_matrix_1 - eye_d)
error_matrix_2 = jnp.tril(inner_product_matrix_2 - eye_d)

# Combine multiplications
error_2_stopped = jax.lax.stop_gradient(error_matrix_2)
quadratic_error_matrix = 2 * (cum_error_matrix_1_below_threshold * error_matrix_1 * error_2_stopped)
```

---

## 8. MICRO: Use JIT-Compiled Helper Functions

### Problem
Some operations inside the JIT-compiled function could be pre-compiled separately.

### Solution
Move `check_previous_entries_below_threshold` outside and ensure it's JIT-compiled:

```python
@jax.jit
def check_previous_entries_below_threshold(matrix, threshold):
    # ... implementation
```

---

## Implementation Priority

### Phase 1 - Critical (Est. 60-80% total speedup)
1. ✅ Vectorize replay buffer sampling
2. ✅ Batch encoder applications

### Phase 2 - High Impact (Est. 10-15% additional speedup)
3. ✅ Optimize coordinate indexing
4. ✅ Reduce visualization frequency

### Phase 3 - Low Hanging Fruit (Est. 5-10% additional speedup)
5. ✅ Optimize inner product computations
6. ✅ Reduce metrics conversion overhead
7. ✅ Optimize error matrix operations

---

## Estimated Overall Speedup
Implementing all optimizations: **2-3x faster training** (20,000 steps could run in 1/2 to 1/3 the time)

Most critical improvements come from Phase 1 optimizations.

---

## Testing Recommendations

1. **Benchmark before optimization**: Run with current code and track:
   - Total training time
   - Steps per second (SPS)
   - Time per gradient step

2. **Benchmark after each optimization**: Verify improvements

3. **Verify correctness**: Check that:
   - Loss curves match original implementation
   - Final learned eigenvectors are similar
   - No numerical instabilities introduced

4. **Profile with JAX profiler**:
   ```python
   jax.profiler.start_trace("/tmp/jax-trace")
   # ... training code ...
   jax.profiler.stop_trace()
   ```

---

## Additional Considerations

### Memory vs Speed Trade-offs
- Batching encoder calls: Uses more memory but faster
- Pre-computing terminals: Uses more storage but faster sampling

### JAX-Specific Optimizations
- Ensure all operations stay on device (minimize device-to-host transfers)
- Use `jax.vmap` for parallel operations where applicable
- Consider using `jax.lax.scan` for sequential operations

### Hardware Considerations
- GPU: Batching operations will show larger speedups
- CPU: Vectorization will show larger speedups
- The optimizations benefit both, but different magnitudes
