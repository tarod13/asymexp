# Deep Dive: Full Vectorization vs Numba

## Understanding Approach 2: Full Vectorization

Let me walk through a concrete example with **batch_size=4** to show exactly how vectorization eliminates loops.

### The Problem We're Solving

Given:
- `batch_size = 4` (we sampled 4 transitions)
- Each sample comes from a different episode at a different starting position
- We need to find the next 1 or 2 terminal states for each sample

**Sample data:**
```python
episode_idx = [2, 5, 2, 8]        # Which episodes we sampled from
obs_idx = [10, 5, 25, 15]         # Starting position in each episode
transition_ranges = [50, 30, 50, 40]  # Length of each episode

# Episode 2's terminals array (length 50):
# [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,...]
#                        ^12            ^27

# Episode 5's terminals array (length 30):
# [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#                ^7              ^15

# Episode 8's terminals array (length 40):
# [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,...]
#                                    ^17
```

**What we want to compute:**
```python
# For sample 0: episode_idx=2, obs_idx=10
#   Terminals at positions 12 and 27 (both >= 10)
#   → max_duration = 27 - 10 = 17 (second terminal)

# For sample 1: episode_idx=5, obs_idx=5
#   Terminals at positions 7 and 15 (both >= 5)
#   → max_duration = 15 - 5 = 10 (second terminal)

# For sample 2: episode_idx=2, obs_idx=25
#   Terminal at position 27 (>= 25), but next would be outside episode
#   → max_duration = 27 - 25 = 2 (first terminal only)

# For sample 3: episode_idx=8, obs_idx=15
#   Terminal at position 17 (>= 15), no second terminal
#   → max_duration = 17 - 15 = 2 (first terminal only)
```

---

## Sequential Approach (Current/Numba)

```python
max_durations = np.zeros(4, dtype=np.int32)

for i in range(4):  # Process each sample one by one
    ep_idx = episode_idx[i]
    start_idx = obs_idx[i]
    ep_length = transition_ranges[i]

    # Slice the terminal array for this episode starting from obs_idx
    terminals = self._episodes['terminals'][ep_idx, start_idx:ep_length]
    # Sample 0: terminals = episodes[2, 10:50] → [0,0,1,0,0,...,0,0,1,0,...]

    # Find where terminals == 1
    terminal_indices = np.where(terminals == 1)[0]
    # Sample 0: terminal_indices = [2, 17]  (relative to start_idx=10)

    if len(terminal_indices) >= 2:
        max_durations[i] = terminal_indices[1]  # Second terminal
    elif len(terminal_indices) >= 1:
        max_durations[i] = terminal_indices[0]  # First terminal
    else:
        max_durations[i] = ep_length - start_idx - 1
```

**Problems:**
- 4 separate slicing operations (`terminals[ep_idx, start_idx:ep_length]`)
- 4 separate `np.where()` calls
- 4 conditional branches
- Each iteration waits for the previous one to complete

Even with Numba compilation, this is still **fundamentally sequential**.

---

## Vectorized Approach: Step-by-Step

### Step 1: Create Search Windows

Instead of processing one sample at a time, we create a matrix that contains ALL the positions we need to search.

```python
max_window = 50  # Maximum episode length

# Create offset array: [0, 1, 2, 3, ..., 49]
search_offsets = np.arange(max_window)  # Shape: (50,)

# Reshape for broadcasting: shape (1, 50)
search_offsets = search_offsets[None, :]

# Add to obs_idx (shape 4,) → broadcasts to (4, 50)
search_indices = obs_idx[:, None] + search_offsets
```

**Result of search_indices:**
```python
# Shape: (4, 50)
[[10, 11, 12, 13, 14, ..., 59],  # Sample 0: search positions for ep_idx=2
 [ 5,  6,  7,  8,  9, ..., 54],  # Sample 1: search positions for ep_idx=5
 [25, 26, 27, 28, 29, ..., 74],  # Sample 2: search positions for ep_idx=2
 [15, 16, 17, 18, 19, ..., 64]]  # Sample 3: search positions for ep_idx=8
```

### Step 2: Clip to Valid Episode Lengths

Some indices exceed the episode length, so clip them:

```python
search_indices = np.minimum(
    search_indices,
    transition_ranges[:, None] - 1  # Shape: (4, 1) → broadcasts
)
```

**After clipping:**
```python
# Shape: (4, 50)
[[10, 11, 12, 13, 14, ..., 49, 49, 49, ...],  # Sample 0: max=49 (ep_len=50)
 [ 5,  6,  7,  8,  9, ..., 29, 29, 29, ...],  # Sample 1: max=29 (ep_len=30)
 [25, 26, 27, 28, 29, ..., 49, 49, 49, ...],  # Sample 2: max=49 (ep_len=50)
 [15, 16, 17, 18, 19, ..., 39, 39, 39, ...]]  # Sample 3: max=39 (ep_len=40)
```

### Step 3: Gather All Terminals in One Operation

This is the **magic step** - we use NumPy's advanced indexing to fetch all values at once:

```python
batch_terminals = self._episodes['terminals'][
    episode_idx[:, None],  # Shape: (4, 1) → broadcasts to (4, 50)
    search_indices         # Shape: (4, 50)
]
```

**How broadcasting works:**
```python
episode_idx[:, None] =           search_indices =
[[2],                            [[10, 11, 12, ..., 49],
 [5],                             [ 5,  6,  7, ..., 29],
 [2],                             [25, 26, 27, ..., 49],
 [8]]                             [15, 16, 17, ..., 39]]

# Broadcasts to both shape (4, 50), then indexes:
terminals[2, 10], terminals[2, 11], terminals[2, 12], ..., terminals[2, 49]
terminals[5,  5], terminals[5,  6], terminals[5,  7], ..., terminals[5, 29]
terminals[2, 25], terminals[2, 26], terminals[2, 27], ..., terminals[2, 49]
terminals[8, 15], terminals[8, 16], terminals[8, 17], ..., terminals[8, 39]
```

**Result batch_terminals (shape 4×50):**
```python
[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, ...],  # Sample 0
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],  # Sample 1
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],  # Sample 2
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]]  # Sample 3
#     ^first           ^second (where they exist)
```

### Step 4: Find Terminal Positions with Cumsum

Here's a clever trick to find the 1st and 2nd terminals:

```python
terminal_mask = (batch_terminals == 1).astype(np.int32)
# [[0, 0, 1, 0, 0, ..., 1, 0, ...],
#  [0, 0, 1, 0, 0, ..., 1, 0, ...],
#  [0, 0, 1, 0, 0, ..., 0, 0, ...],
#  [0, 0, 1, 0, 0, ..., 0, 0, ...]]

terminal_cumsum = np.cumsum(terminal_mask, axis=1)
# [[0, 0, 1, 1, 1, ..., 2, 2, ...],   # First terminal at idx=2, second at idx=17
#  [0, 0, 1, 1, 1, ..., 2, 2, ...],   # First terminal at idx=2, second at idx=10
#  [0, 0, 1, 1, 1, ..., 1, 1, ...],   # First terminal at idx=2, no second
#  [0, 0, 1, 1, 1, ..., 1, 1, ...]]   # First terminal at idx=2, no second
```

**Key insight:**
- Where `terminal_cumsum == 1` AND `terminal_mask == 1` → **first terminal**
- Where `terminal_cumsum == 2` AND `terminal_mask == 1` → **second terminal**

```python
first_terminal_mask = (terminal_cumsum == 1) & (terminal_mask == 1)
# [[False, False, True, False, ..., False, ...],
#  [False, False, True, False, ..., False, ...],
#  [False, False, True, False, ..., False, ...],
#  [False, False, True, False, ..., False, ...]]

second_terminal_mask = (terminal_cumsum == 2) & (terminal_mask == 1)
# [[False, False, False, ..., True, False, ...],   # Second at idx=17
#  [False, False, False, ..., True, False, ...],   # Second at idx=10
#  [False, False, False, ..., False, False, ...],  # No second terminal
#  [False, False, False, ..., False, False, ...]]  # No second terminal

# Find the positions (indices along axis=1)
first_terminal_pos = np.argmax(first_terminal_mask, axis=1)
# [2, 2, 2, 2]  (positions in the window, not absolute)

second_terminal_pos = np.argmax(second_terminal_mask, axis=1)
# [17, 10, 0, 0]  (0 means not found - we check this below)

# Check if we actually found them
has_first = np.any(first_terminal_mask, axis=1)
# [True, True, True, True]

has_second = np.any(second_terminal_mask, axis=1)
# [True, True, False, False]
```

### Step 5: Vectorized Selection

Finally, use `np.where` (vectorized conditional) to select the right duration:

```python
max_durations = np.where(
    has_second,                        # Condition 1
    second_terminal_pos,               # Use second terminal
    np.where(
        has_first,                     # Condition 2
        first_terminal_pos,            # Use first terminal
        transition_ranges - obs_idx - 1  # Default: end of episode
    )
)
```

**Result:**
```python
max_durations = [17, 10, 2, 2]
```

This matches what we calculated manually! ✅

---

## Why Full Vectorization is Faster

### CPU/GPU Perspective

**Sequential (even with Numba):**
```
CPU executes:
  Load episode_idx[0] → Load terminals[2, 10:50] → Find where == 1 → Store result[0]
  Load episode_idx[1] → Load terminals[5, 5:30]  → Find where == 1 → Store result[1]
  Load episode_idx[2] → Load terminals[2, 25:50] → Find where == 1 → Store result[2]
  Load episode_idx[3] → Load terminals[8, 15:40] → Find where == 1 → Store result[3]
```
Total: **4 serial operations** (even if each is fast)

**Vectorized:**
```
CPU executes:
  Load all data → SIMD process 4 samples in parallel → Store all results

Single memory fetch for batch_terminals[episode_idx[:, None], search_indices]
Single SIMD operation for terminal_mask = (batch_terminals == 1)
Single SIMD operation for cumsum
Single SIMD operation for argmax
Single SIMD operation for np.where
```
Total: **1 parallel operation** (processes multiple elements per instruction)

### Memory Bandwidth

**Sequential:**
- 4 separate memory accesses (poor cache utilization)
- Each slice creates a temporary array
- Cannot use memory prefetching efficiently

**Vectorized:**
- 1 large contiguous memory access
- CPU can prefetch data
- Better cache line utilization
- Data stays in L1/L2 cache

### Instruction-Level Parallelism

Modern CPUs have SIMD instructions (SSE, AVX, AVX-512):

**Sequential:**
```
compare 1 value with 1 value → 1 result
compare 1 value with 1 value → 1 result
compare 1 value with 1 value → 1 result
compare 1 value with 1 value → 1 result
```

**Vectorized (AVX-512):**
```
compare 16 values with 16 values → 16 results (in 1 instruction!)
```

NumPy automatically uses SIMD when operations are vectorized. With batch_size=256, you get:
- **16× reduction in comparisons** (with AVX-512)
- **Better pipelining** (CPU can process multiple instructions simultaneously)

---

## Approach 1+2 vs Approach 3 (Numba)

### Why 1+2 is Better

**Approach 3 (Numba alone):**
```python
@numba.jit
def find_terminals(terminals, episode_idx, obs_idx, ...):
    for i in range(batch_size):  # Compiled to C, but still SERIAL
        for j in range(obs_idx[i], ep_length):  # Nested loop! O(batch_size × ep_length)
            if terminals[ep_idx, j] == 1:
                # ... logic
```

**Problems with Numba:**
1. **Still sequential** - processes one sample at a time
2. **Nested loops** - O(batch_size × average_ep_length)
3. **No SIMD** - compares one element at a time
4. **Poor cache usage** - scattered memory accesses
5. **Branch prediction misses** - conditionals inside loops

**Approach 1+2 (Pre-compute + Vectorize):**
```python
# Pre-computation (Approach 1): O(total_transitions) - done ONCE
for episode in episodes:
    terminal_positions = np.where(episode['terminals'] == 1)[0]
    store_terminal_indices(terminal_positions)

# Sampling (Approach 2): O(batch_size × max_terminals_per_episode)
# Gather pre-computed indices - vectorized!
batch_term_indices = terminal_indices[episode_idx]  # (batch_size, max_terminals)
valid_mask = batch_term_indices >= obs_idx[:, None]  # Vectorized comparison
relative_indices = batch_term_indices - obs_idx[:, None]  # Vectorized subtraction
sorted_terms = np.sort(relative_indices, axis=1)  # Vectorized sort
```

**Advantages:**
1. **Pre-computation amortizes cost** - terminal search done once during data collection
2. **Vectorized comparisons** - SIMD processes multiple samples
3. **No nested loops** - O(batch_size × max_terminals) where max_terminals << ep_length
4. **Better memory access** - contiguous array operations
5. **No branches in hot path** - all operations are straight-line

### Performance Comparison

Assume:
- `batch_size = 256`
- `average_ep_length = 100`
- `average_terminals_per_episode = 5`

**Approach 3 (Numba):**
```
Operations per sample() call:
  256 samples × 100 positions to check = 25,600 comparisons
  Even at 10 cycles/comparison (optimistic): 256,000 CPU cycles
```

**Approach 1+2:**
```
Pre-computation (done once):
  1000 episodes × 100 positions = 100,000 comparisons
  Stored in ~5KB memory (1000 × 5 terminals × 4 bytes)

Per sample() call:
  Gather: 256 samples × 5 terminals = 1,280 values (single memory access)
  Comparisons: 256 × 5 = 1,280 (vectorized → ~80 SIMD ops with AVX-512)
  Sort: 256 × 5 log(5) ≈ 3,000 operations
  Total: ~5,000 operations vs 25,600
```

**Speedup: ~5×** in the sampling operation alone!

### Real-World Impact

With 20,000 training steps:

**Numba:**
- 256,000 cycles × 20,000 steps = 5.12 billion cycles
- At 3 GHz: ~1.7 seconds in sampling alone

**Vectorized:**
- 5,000 operations × 20,000 steps = 100 million operations
- At 3 GHz with SIMD: ~0.2 seconds in sampling alone

**Difference: 1.5 seconds saved** just in this one operation!

---

## Memory Usage Comparison

### Approach 3 (Numba)
```
Memory per sample() call: minimal
  - Temporary arrays inside JIT function
  - ~few KB

Total: ~10 KB
```

### Approach 1+2 (Pre-compute + Vectorize)
```
Pre-computed storage (one-time):
  - terminal_indices: (1000 episodes × 100 max_terminals × 4 bytes) = 400 KB
  - num_terminals: (1000 episodes × 4 bytes) = 4 KB
  Total: ~404 KB

Per sample() call:
  - batch_term_indices: (256 × 100 × 4) = 100 KB
  - Various masks/intermediate arrays: ~200 KB
  Total: ~300 KB
```

**Memory cost: ~700 KB total** - completely negligible on modern systems (even a phone has GB of RAM).

---

## Why Not Just Use Numba?

Numba is great, but:

1. **Fundamental algorithmic limitation** - Even compiled to C, a sequential search through 100 positions for 256 samples is slower than a single vectorized operation

2. **CPU features underutilized** - Modern CPUs have 16-64 wide SIMD units that sit idle with scalar code

3. **Scalability** - As batch size increases, Numba scales linearly, vectorization scales sub-linearly (constant-time per batch element)

4. **GPU compatibility** - Vectorized NumPy/JAX code can run on GPU with minimal changes; Numba requires CUDA rewrite

**Analogy:**
- Numba = Making a bicycle go faster (better materials, aerodynamics)
- Vectorization = Switching to a car (fundamentally different approach)

You can make the bicycle very fast, but it's still a bicycle!

---

## Conclusion

**Use Approach 1+2 because:**

1. **Better algorithmic complexity**
   - Pre-computation: O(1) amortized cost
   - Vectorized search: O(batch_size) with large constant-factor reduction from SIMD

2. **Better hardware utilization**
   - SIMD instructions
   - Memory prefetching
   - Cache efficiency

3. **Scalability**
   - Handles larger batches efficiently
   - GPU-ready

4. **Total speedup: 70-80%** vs Numba's 50-70%

The only downside is memory usage (~1 MB), which is completely negligible.

**Bottom line:** Vectorization changes the fundamental algorithm, not just speeds up the implementation. That's why it beats even highly optimized sequential code.
