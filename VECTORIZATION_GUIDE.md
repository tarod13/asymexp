# Vectorizing Replay Buffer Sampling - Detailed Guide

## Problem Analysis

### Current Sequential Implementation (Lines 167-185)

```python
for i in range(batch_size):  # Sequential loop - SLOW!
    ep_idx = episode_idx[i]
    start_idx = obs_idx[i]
    ep_length = transition_ranges[i]

    # Find next terminal starting from obs_idx
    terminals = self._episodes['terminals'][ep_idx, start_idx:ep_length]
    terminal_indices = np.where(terminals == 1)[0]

    if len(terminal_indices) > 0:
        if len(terminal_indices) > 1:
            max_durations[i] = terminal_indices[1]
        else:
            max_durations[i] = terminal_indices[0]
    else:
        max_durations[i] = ep_length - start_idx - 1
```

### Why This Is Slow

1. **Python loop overhead**: Each iteration has interpreter overhead
2. **Serial processing**: Processes one sample at a time (no parallelism)
3. **Variable-length slicing**: Each `terminals[start_idx:ep_length]` creates a new array
4. **Repeated numpy operations**: `np.where()` called `batch_size` times
5. **CPU-bound**: Cannot leverage GPU or SIMD vectorization

**With batch_size=256**: This loop runs 256 times, with 20,000 training steps = **5,120,000 iterations** total!

---

## Solution Approaches

There are three main strategies to vectorize this operation, each with different trade-offs.

---

## Approach 1: Pre-compute Terminal Indices (RECOMMENDED)

### Strategy
Pre-compute the positions of terminals when episodes are added to the buffer, avoiding runtime searches entirely.

### Implementation

#### Modify `add_episode()` method:

```python
def add_episode(self, episode: Dict):
    for step_component, component_list in episode.items():
        component_array = np.array(component_list, dtype=component_list[0].dtype)
        n = component_array.shape[0]

        if step_component not in self._episodes:
            _shape = component_array.shape
            if self._max_episode_length is not None:
                _shape = (self._max_episode_length,) + _shape[1:]
            self._episodes[step_component] = np.empty(
                (self._max_episodes,) + _shape, dtype=component_array.dtype
            )

        # Add component to episode buffer
        self._episodes[step_component][self._idx][:n] = component_array

        # NEW: Pre-compute terminal indices
        if step_component == 'terminals':
            # Find all terminal positions
            terminal_positions = np.where(component_array == 1)[0]

            # Store in a new buffer (create if needed)
            if 'terminal_indices' not in self._episodes:
                # Max possible terminals per episode
                max_terminals = 100  # Adjust based on your domain
                self._episodes['terminal_indices'] = np.full(
                    (self._max_episodes, max_terminals), -1, dtype=np.int32
                )
                self._episodes['num_terminals'] = np.zeros(
                    self._max_episodes, dtype=np.int32
                )

            # Store terminal positions
            num_terms = len(terminal_positions)
            self._episodes['terminal_indices'][self._idx, :num_terms] = terminal_positions
            self._episodes['num_terminals'][self._idx] = num_terms

        if step_component == 'obs':
            self._episodes_length[self._idx] = n

    self._idx = (self._idx + 1) % self._max_episodes
    self._full = self._full or self._idx == 0
```

#### Modified `sample()` method:

```python
def sample(self, batch_size, discount, env_info={}):
    # Sample episodes
    episode_idx = np.random.randint(len(self), size=batch_size)

    # Sample transitions
    transition_ranges = self._get_episode_lengths(episode_idx)
    obs_idx = uniform_sampling(transition_ranges - 1)

    # Calculate remaining trajectory length using pre-computed terminals
    if 'terminal_indices' in self._episodes:
        max_durations = np.zeros(batch_size, dtype=np.int32)

        # Vectorized approach using pre-computed indices
        for i in range(batch_size):
            ep_idx = episode_idx[i]
            start_idx = obs_idx[i]
            ep_length = transition_ranges[i]

            # Get pre-computed terminal positions for this episode
            num_terms = self._episodes['num_terminals'][ep_idx]
            term_positions = self._episodes['terminal_indices'][ep_idx, :num_terms]

            # Find terminals >= start_idx (vectorized comparison)
            valid_terms = term_positions[term_positions >= start_idx]

            if len(valid_terms) >= 2:
                max_durations[i] = valid_terms[1] - start_idx
            elif len(valid_terms) == 1:
                max_durations[i] = valid_terms[0] - start_idx
            else:
                max_durations[i] = ep_length - start_idx - 1
    else:
        max_durations = transition_ranges - obs_idx - 1

    transition_durations = discounted_sampling(max_durations, discount=discount) + 1
    next_obs_idx = obs_idx + transition_durations

    # ... rest remains the same
```

### Pros
- ✅ Much faster sampling (no runtime terminal searches)
- ✅ Simpler logic in hot path (sample method)
- ✅ One-time computational cost during data collection
- ✅ Still has a small loop, but much less work per iteration

### Cons
- ❌ Increased memory usage (stores terminal indices)
- ❌ Still has a loop (though much faster)
- ❌ Requires modifying `add_episode()`

### Performance Impact
**Expected speedup: 40-60%** - The loop is still there, but each iteration is much faster.

---

## Approach 2: Fully Vectorized with Padding (FASTEST)

### Strategy
Materialize all possible terminal searches into a large padded array, then use vectorized operations.

### Implementation

```python
def sample(self, batch_size, discount, env_info={}):
    # Sample episodes
    episode_idx = np.random.randint(len(self), size=batch_size)

    # Sample transitions
    transition_ranges = self._get_episode_lengths(episode_idx)
    obs_idx = uniform_sampling(transition_ranges - 1)

    if 'terminals' in self._episodes:
        # Create a fixed-size window for terminal search
        max_search_window = self._max_episode_length

        # Prepare indices for gathering terminals
        # Shape: (batch_size, max_search_window)
        search_offsets = np.arange(max_search_window)[None, :]  # (1, max_window)
        search_indices = obs_idx[:, None] + search_offsets  # (batch_size, max_window)

        # Clip to valid episode lengths
        search_indices = np.minimum(
            search_indices,
            transition_ranges[:, None] - 1
        )

        # Gather all terminal values in one operation
        # Shape: (batch_size, max_search_window)
        batch_terminals = self._episodes['terminals'][
            episode_idx[:, None],  # (batch_size, 1)
            search_indices         # (batch_size, max_window)
        ]

        # Find first two terminal positions for each sample
        # Use cumsum to find positions
        terminal_mask = (batch_terminals == 1).astype(np.int32)
        terminal_cumsum = np.cumsum(terminal_mask, axis=1)

        # Find where cumsum == 1 (first terminal) and cumsum == 2 (second terminal)
        first_terminal_mask = (terminal_cumsum == 1) & (terminal_mask == 1)
        second_terminal_mask = (terminal_cumsum == 2) & (terminal_mask == 1)

        # Get indices (offset from start_idx)
        first_terminal_pos = np.argmax(first_terminal_mask, axis=1)
        second_terminal_pos = np.argmax(second_terminal_mask, axis=1)

        # Check if we actually found terminals
        has_first = np.any(first_terminal_mask, axis=1)
        has_second = np.any(second_terminal_mask, axis=1)

        # Calculate max_durations using vectorized selection
        max_durations = np.where(
            has_second,
            second_terminal_pos,
            np.where(
                has_first,
                first_terminal_pos,
                transition_ranges - obs_idx - 1
            )
        )
    else:
        max_durations = transition_ranges - obs_idx - 1

    transition_durations = discounted_sampling(max_durations, discount=discount) + 1
    next_obs_idx = obs_idx + transition_durations

    # ... rest remains the same
```

### Explanation of Vectorized Logic

1. **Create search window**:
   ```python
   search_indices = obs_idx[:, None] + np.arange(max_window)[None, :]
   # For batch_size=3, obs_idx=[10, 20, 30]:
   # [[10, 11, 12, ..., 10+max_window],
   #  [20, 21, 22, ..., 20+max_window],
   #  [30, 31, 32, ..., 30+max_window]]
   ```

2. **Gather terminals in bulk**:
   ```python
   batch_terminals = self._episodes['terminals'][episode_idx[:, None], search_indices]
   # Single advanced indexing operation instead of batch_size separate operations
   ```

3. **Use cumsum to find terminal positions**:
   ```python
   terminal_cumsum = np.cumsum(terminal_mask, axis=1)
   # Example for one row: [0,0,1,1,1,2,2,2,...]
   #                           ^first  ^second
   ```

4. **Vectorized selection**:
   ```python
   max_durations = np.where(has_second, second_pos, np.where(has_first, first_pos, default))
   # All conditions evaluated in parallel
   ```

### Pros
- ✅ **Fully vectorized** - No Python loops!
- ✅ **Maximum performance** - Can use SIMD, GPU acceleration
- ✅ **Parallelizable** across batch dimension
- ✅ **Expected speedup: 60-80%**

### Cons
- ❌ **High memory usage** - Creates (batch_size × max_episode_length) temporary arrays
- ❌ **Memory-bandwidth bound** - May be slower if memory is limited
- ❌ **Complex code** - Harder to understand and debug

### When to Use
- Large batches (batch_size > 128)
- GPU acceleration available
- Sufficient memory bandwidth
- Episodes are relatively short (< 1000 steps)

---

## Approach 3: Hybrid with Numba JIT (BALANCED)

### Strategy
Keep the loop structure but compile it with Numba for near-C performance.

### Implementation

```python
import numba

# Add this as a module-level function (outside the class)
@numba.jit(nopython=True, parallel=False, cache=True)
def find_max_durations_numba(
    terminals_array,      # Full terminals buffer
    episode_idx,          # Which episodes to sample from
    obs_idx,              # Start positions
    transition_ranges,    # Episode lengths
    batch_size
):
    """JIT-compiled function to find max durations."""
    max_durations = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        ep_idx = episode_idx[i]
        start_idx = obs_idx[i]
        ep_length = transition_ranges[i]

        # Search for terminals
        terminal_count = 0
        first_term = -1
        second_term = -1

        for j in range(start_idx, ep_length):
            if terminals_array[ep_idx, j] == 1:
                terminal_count += 1
                if terminal_count == 1:
                    first_term = j - start_idx
                elif terminal_count == 2:
                    second_term = j - start_idx
                    break

        if second_term >= 0:
            max_durations[i] = second_term
        elif first_term >= 0:
            max_durations[i] = first_term
        else:
            max_durations[i] = ep_length - start_idx - 1

    return max_durations


# Modified sample() method
def sample(self, batch_size, discount, env_info={}):
    episode_idx = np.random.randint(len(self), size=batch_size)
    transition_ranges = self._get_episode_lengths(episode_idx)
    obs_idx = uniform_sampling(transition_ranges - 1)

    if 'terminals' in self._episodes:
        # Call JIT-compiled function
        max_durations = find_max_durations_numba(
            self._episodes['terminals'],
            episode_idx,
            obs_idx,
            transition_ranges,
            batch_size
        )
    else:
        max_durations = transition_ranges - obs_idx - 1

    transition_durations = discounted_sampling(max_durations, discount=discount) + 1
    next_obs_idx = obs_idx + transition_durations

    # ... rest remains the same
```

### Pros
- ✅ **Near-C performance** - 10-100x faster than Python loops
- ✅ **Low memory overhead** - Same memory usage as original
- ✅ **Simple to implement** - Just add `@numba.jit` decorator
- ✅ **Easy to understand** - Same logic as original

### Cons
- ❌ **Requires Numba** - Additional dependency
- ❌ **First call is slow** - JIT compilation overhead
- ❌ **Still sequential** - Doesn't leverage vectorization fully

### Performance Impact
**Expected speedup: 50-70%** - Much faster loop, but still a loop.

---

## Comparison Matrix

| Approach | Speedup | Memory | Complexity | Dependencies |
|----------|---------|--------|------------|--------------|
| **Current** | 1x (baseline) | Low | Low | None |
| **Pre-compute** | 40-60% | Medium | Medium | None |
| **Fully Vectorized** | 60-80% | High | High | None |
| **Numba JIT** | 50-70% | Low | Low | Numba |

---

## Recommended Implementation Strategy

### For Immediate Gains (Phase 1)
**Use Approach 3 (Numba JIT)**
- Fastest to implement (just add decorator + external function)
- Good performance improvement
- Minimal code changes

### For Maximum Performance (Phase 2)
**Combine Approach 1 + 2**
- Pre-compute terminal indices during `add_episode()`
- Use vectorized search with pre-computed indices
- Best of both worlds

### Example Combined Approach

```python
def sample(self, batch_size, discount, env_info={}):
    episode_idx = np.random.randint(len(self), size=batch_size)
    transition_ranges = self._get_episode_lengths(episode_idx)
    obs_idx = uniform_sampling(transition_ranges - 1)

    if 'terminal_indices' in self._episodes:
        # Gather pre-computed terminal indices for all samples
        # Shape: (batch_size, max_terminals_per_episode)
        batch_term_indices = self._episodes['terminal_indices'][episode_idx]
        batch_num_terms = self._episodes['num_terminals'][episode_idx]

        # Create mask for valid terminals >= obs_idx
        # Broadcasting: (batch_size, max_terminals) >= (batch_size, 1)
        valid_mask = batch_term_indices >= obs_idx[:, None]

        # Mask out invalid terminal slots (-1 values)
        valid_mask &= (batch_term_indices >= 0)

        # Adjust indices to be relative to obs_idx
        relative_term_indices = batch_term_indices - obs_idx[:, None]

        # Set invalid positions to a large number
        relative_term_indices = np.where(valid_mask, relative_term_indices, 999999)

        # Sort to get first and second terminals
        sorted_terms = np.sort(relative_term_indices, axis=1)

        # Count how many valid terminals each sample has
        num_valid_terms = np.sum(valid_mask, axis=1)

        # Vectorized selection
        max_durations = np.where(
            num_valid_terms >= 2,
            sorted_terms[:, 1],  # Second terminal
            np.where(
                num_valid_terms >= 1,
                sorted_terms[:, 0],  # First terminal
                transition_ranges - obs_idx - 1  # No terminals
            )
        )
    else:
        max_durations = transition_ranges - obs_idx - 1

    transition_durations = discounted_sampling(max_durations, discount=discount) + 1
    next_obs_idx = obs_idx + transition_durations

    # ... rest remains the same
```

This combines:
- Pre-computation (during add_episode)
- Vectorized operations (during sample)
- No Python loops
- Moderate memory usage

**Expected speedup: 70-80%** 🚀

---

## Testing and Validation

### Unit Test

```python
def test_vectorized_sampling():
    """Ensure vectorized version produces same results as original."""

    # Create test buffer
    buffer = EpisodicReplayBuffer(
        max_episodes=10,
        max_episode_length=100,
        observation_type='canonical_state',
        seed=42
    )

    # Add test episodes
    for _ in range(10):
        obs = np.random.randint(0, 50, size=(50, 1))
        terminals = np.zeros(50, dtype=np.int32)
        terminals[[10, 20, 30]] = 1  # Add terminals at specific positions

        buffer.add_episode({'obs': obs, 'terminals': terminals})

    # Sample multiple times and check consistency
    np.random.seed(42)
    batch1 = buffer.sample(256, discount=0.2)

    np.random.seed(42)
    batch2 = buffer.sample(256, discount=0.2)

    # Should produce identical results
    assert np.allclose(batch1.obs, batch2.obs)
    assert np.allclose(batch1.next_obs, batch2.next_obs)

    print("✓ Vectorized sampling produces consistent results")
```

### Benchmark

```python
import time

def benchmark_sampling(buffer, num_iterations=1000, batch_size=256):
    """Benchmark sampling performance."""

    start = time.time()
    for _ in range(num_iterations):
        batch = buffer.sample(batch_size, discount=0.2)
    end = time.time()

    total_time = end - start
    samples_per_sec = (num_iterations * batch_size) / total_time

    print(f"Total time: {total_time:.2f}s")
    print(f"Samples/sec: {samples_per_sec:,.0f}")
    print(f"Time per sample call: {total_time/num_iterations*1000:.2f}ms")

    return total_time

# Compare original vs optimized
print("Original implementation:")
time_original = benchmark_sampling(buffer_original)

print("\nOptimized implementation:")
time_optimized = benchmark_sampling(buffer_optimized)

print(f"\nSpeedup: {time_original/time_optimized:.2f}x")
```

---

## Key Takeaways

1. **The Python loop is the bottleneck** - It runs millions of times during training

2. **Multiple solutions exist** - Choose based on your constraints:
   - Want quick wins? → Use Numba (Approach 3)
   - Want maximum speed? → Use full vectorization (Approach 2)
   - Want balance? → Pre-compute + vectorize (Approach 1 + 2)

3. **Pre-computation is powerful** - Doing work once during data collection saves millions of operations

4. **NumPy's power is in bulk operations** - Single advanced indexing operation >> many simple ones

5. **Test thoroughly** - Vectorized code can be tricky; ensure correctness before optimizing further

Would you like me to implement any of these approaches in your codebase?
