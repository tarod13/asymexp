import jax
import jax.numpy as jnp
import numpy as np
import chex
import flashbax as fbx
from typing import Tuple, Dict


# Define transition data structure (similar to TimeStep in PureJaxRL)
@chex.dataclass(frozen=True)
class TimeStep:
    state_idx: chex.Array  # Flattened state index
    action: chex.Array     # Action taken
    reward: chex.Array     # Reward received
    next_state_idx: chex.Array  # Next state index
    done: chex.Array       # Terminal flag
    is_valid: chex.Array  # Flag for valid high-level transition


# Create vectorized versions of environment functions
def make_env_fns(env, env_params=None):
    """Create vectorized reset and step functions."""
    # Vectorized initial reset
    def vmap_init_reset(n_envs):
        def _reset_fn(rng):
            return jax.vmap(env.reset, in_axes=(0, None))(
                jax.random.split(rng, n_envs), env_params
            )
        return _reset_fn

    # Vectorized reset
    def vmap_reset(n_envs):
        def _reset_one_env(rng, state, done):
            return jax.lax.cond(
                done,
                lambda: env.reset(rng, env_params),
                lambda: state,   # No reset needed
            )
        def _reset_fn(rng, states, dones):
            return jax.vmap(_reset_one_env)(
                jax.random.split(rng, n_envs), states, dones
            )
        return _reset_fn

    # Vectorized step
    def vmap_step(n_envs):
        def _step_fn(rng, states, actions):
            next_states, rewards, dones, infos = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(rng, n_envs), states, actions, env_params)

            # Get state representations (indices)
            state_indices = jax.vmap(env.get_state_representation)(states)
            next_state_indices = jax.vmap(env.get_state_representation)(next_states)

            return next_states, state_indices, next_state_indices, rewards, dones, infos
        return _step_fn

    return vmap_init_reset, vmap_reset, vmap_step


# Function to collect data following PureJaxRL style
def collect_data_purejaxrl_style(env, num_envs, num_steps, seed=42, precision=32):
    """
    Collect data from GridWorld environments using PureJaxRL style.

    Args:
        env: GridWorld environment
        num_envs: Number of parallel environments
        num_steps: Number of steps to collect
        seed: Random seed

    Returns:
        buffer_state: Final buffer state
    """
    # Determine data types based on precision
    if precision == 32:
        dtype = jnp.float32
        int_dtype = jnp.int32
    elif precision == 64:
        dtype = jnp.float64
        int_dtype = jnp.int64
    else:
        raise ValueError("Precision must be 32 or 64")

    # Get environment parameters
    env_params = env.get_params()

    # Create vectorized functions
    vmap_init_reset, vmap_reset, vmap_step = make_env_fns(env, env_params)

    # Initialize random key
    rng = jax.random.PRNGKey(seed)

    # Buffer configuration - similar to PureJaxRL
    buffer = fbx.make_flat_buffer(
        max_length=num_envs * num_steps,
        min_length=1,  # We'll sample later for training
        sample_batch_size=32,  # Default batch size, can be changed when sampling
        add_sequences=False,
        add_batch_size=num_envs,
    )

    # Jit-compile buffer functions
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    # Create a dummy timestep for buffer initialization
    rng, _rng = jax.random.split(rng)
    _action = jnp.array(0, dtype=int_dtype)
    _state_idx = jnp.array(0, dtype=int_dtype)
    _next_state_idx = jnp.array(0, dtype=int_dtype)
    _reward = jnp.array(0.0, dtype=dtype)
    _done = jnp.array(False, dtype=jnp.bool_)
    _is_valid = jnp.array(False, dtype=jnp.bool_)

    # Initialize buffer
    _timestep = TimeStep(
        state_idx=_state_idx,
        action=_action,
        reward=_reward,
        next_state_idx=_next_state_idx,
        done=_done,
        is_valid=_is_valid,
    )
    buffer_state = buffer.init(_timestep)

    # Define the single update step (similar to PureJaxRL)
    def _update_step(runner_state, unused):
        train_state, buffer_state, env_states, rng = runner_state

        # Split RNG for action selection and environment step
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Select random actions
        actions = jax.random.randint(
            rng_act,
            shape=(num_envs,),
            minval=0,
            maxval=env.action_space,
            dtype=int_dtype,
        )

        # Step environments
        next_env_states, state_indices, next_state_indices, rewards, dones, _ = vmap_step(num_envs)(
            rng_step, env_states, actions
        )

        # Update buffer with transitions
        timestep = TimeStep(
            state_idx=state_indices,
            action=actions,
            reward=rewards,
            next_state_idx=next_state_indices,
            done=dones,
            is_valid=jnp.ones_like(dones, dtype=jnp.bool_),   # All transitions are valid
        )
        buffer_state = buffer.add(buffer_state, timestep)

        # Reset environments if done
        reset_env_states = vmap_reset(num_envs)(rng, next_env_states, dones)

        # Update training state (in this case just a step counter)
        train_state = train_state + num_envs

        # Return updated state
        runner_state = (train_state, buffer_state, reset_env_states, rng)

        # Metrics
        metrics = {
            "timesteps": train_state,
            "buffer_size": jnp.prod(jnp.array(buffer_state['experience'].state_idx.shape)),
        }

        return runner_state, metrics

    # Initialize environments
    rng, reset_rng = jax.random.split(rng)
    env_states = vmap_init_reset(num_envs)(reset_rng)

    # Initialize runner state (similar to PureJaxRL)
    # train_state here is just a step counter
    train_state = jnp.array(0)
    runner_state = (train_state, buffer_state, env_states, rng)

    # Run collection loop using scan (just like PureJaxRL)
    steps_to_run = num_steps
    runner_state, metrics = jax.lax.scan(
        _update_step, runner_state, None, steps_to_run
    )

    # Extract final buffer state
    _, final_buffer_state, _, _ = runner_state

    return final_buffer_state, metrics


def collect_transition_counts(env, num_envs, num_steps, num_states, seed=42):
    """
    Collect transition counts directly without storing trajectories (memory efficient).

    This function runs num_envs parallel environments for num_steps and accumulates
    transition counts into a matrix [num_states, num_actions, num_states].

    Args:
        env: GridWorld environment
        num_envs: Number of parallel environments
        num_steps: Number of steps to collect per environment
        num_states: Total number of states in the environment
        seed: Random seed

    Returns:
        transition_counts: Array of shape [num_states, num_actions, num_states]
        metrics: Dictionary with collection metrics
    """
    # Get environment parameters
    env_params = env.get_params()
    num_actions = env.action_space

    # Create vectorized functions
    vmap_init_reset, vmap_reset, vmap_step = make_env_fns(env, env_params)

    # Initialize random key
    rng = jax.random.PRNGKey(seed)

    # Initialize transition count matrix
    transition_counts = jnp.zeros((num_states, num_actions, num_states), dtype=jnp.float32)

    # Define the single update step
    def _update_step(runner_state, unused):
        transition_counts, env_states, rng = runner_state

        # Split RNG for action selection and environment step
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Select random actions
        actions = jax.random.randint(
            rng_act,
            shape=(num_envs,),
            minval=0,
            maxval=num_actions,
        )

        # Step environments
        next_env_states, state_indices, next_state_indices, rewards, dones, _ = vmap_step(num_envs)(
            rng_step, env_states, actions
        )

        # Accumulate transition counts for all parallel environments
        # Directly update the count matrix using batched indexing
        transition_counts = transition_counts.at[state_indices, actions, next_state_indices].add(1.0)

        # Reset environments if done
        reset_env_states = vmap_reset(num_envs)(rng, next_env_states, dones)

        # Return updated state
        runner_state = (transition_counts, reset_env_states, rng)

        return runner_state, None

    # Initialize environments
    rng, reset_rng = jax.random.split(rng)
    env_states = vmap_init_reset(num_envs)(reset_rng)

    # Initialize runner state
    runner_state = (transition_counts, env_states, rng)

    # Run collection loop using scan
    runner_state, _ = jax.lax.scan(
        _update_step, runner_state, None, num_steps
    )

    # Extract final transition counts
    final_transition_counts, _, _ = runner_state

    # Metrics
    metrics = {
        "total_transitions": int(final_transition_counts.sum()),
        "num_envs": num_envs,
        "num_steps": num_steps,
    }

    return final_transition_counts, metrics


def collect_transition_counts_and_episodes(env, num_envs, num_steps, num_states, seed=42):
    """
    Collect both transition counts and episode trajectories efficiently.

    This function runs num_envs parallel environments for num_steps and accumulates
    both transition counts and full episode trajectories in a single pass.

    Returns data in OGBench-compatible format with proper handling of episode boundaries.

    Args:
        env: GridWorld environment
        num_envs: Number of parallel environments
        num_steps: Number of steps to collect per environment
        num_states: Total number of states in the environment
        seed: Random seed

    Returns:
        transition_counts: Array of shape [num_states, num_actions, num_states]
        episodes: Dictionary with OGBench-compatible format:
            - 'observations': Array of shape [num_envs, max_buffer_size] with state indices
            - 'actions': Array of shape [num_envs, max_buffer_size] with actions
            - 'terminals': Array of shape [num_envs, max_buffer_size] with terminal flags
            - 'valids': Array of shape [num_envs, max_buffer_size] with validity flags
        metrics: Dictionary with collection metrics
    """
    # Get environment parameters
    env_params = env.get_params()
    num_actions = env.action_space

    # Create vectorized functions
    vmap_init_reset, vmap_reset, vmap_step = make_env_fns(env, env_params)

    # Initialize random key
    rng = jax.random.PRNGKey(seed)

    # Initialize transition count matrix
    transition_counts = jnp.zeros((num_states, num_actions, num_states), dtype=jnp.float32)

    # Allocate larger buffer to account for extra slots when done signals occur
    # Worst case: every step is a done, requiring 2 slots each
    max_buffer_size = num_steps * 2 + 1

    # Initialize storage for episodes in OGBench format
    observations = jnp.empty((num_envs, max_buffer_size), dtype=jnp.int32)
    actions = jnp.empty((num_envs, max_buffer_size), dtype=jnp.int32)
    terminals = jnp.empty((num_envs, max_buffer_size), dtype=jnp.int32)
    valids = jnp.empty((num_envs, max_buffer_size), dtype=jnp.int32)

    # Define the single update step
    def _update_step(runner_state, step_idx):
        (transition_counts, env_states, observations, actions, terminals,
         valids, write_indices, last_written, rng) = runner_state

        # Split RNG for action selection and environment step
        rng, rng_act, rng_step = jax.random.split(rng, 3)

        # Select random actions
        action_array = jax.random.randint(
            rng_act,
            shape=(num_envs,),
            minval=0,
            maxval=num_actions,
        )

        # Step environments
        next_env_states, state_indices, next_state_indices, rewards, dones, _ = vmap_step(num_envs)(
            rng_step, env_states, action_array
        )

        # Accumulate transition counts for all parallel environments
        transition_counts = transition_counts.at[state_indices, action_array, next_state_indices].add(1.0)

        # Vectorized update for each environment
        def update_single_env(i):
            write_idx = write_indices[i]   # Current write index for env i corresponds to the next free slot
            state_idx = state_indices[i]
            next_state_idx = next_state_indices[i]
            action = action_array[i]
            done = dones[i]

            # Store action and terminal at write_idx-1, since they 
            # correspond to the current state at write_idx-1
            act_updated = actions[i].at[write_idx-1].set(action) 
            term_updated = terminals[i].at[write_idx-1].set(done.astype(jnp.int32)) 
            valid_updated = valids[i].at[write_idx].set(1-done.astype(jnp.int32))

            # Store next state
            obs_updated = observations[i].at[write_idx].set(next_state_idx)

            # Handle done vs non-done cases
            def handle_done(_):
                # Mark terminal
                term = term_updated.at[write_idx].set(1)

                # Increment write index by 2 (skip over terminal and invalid slot)
                # Reset state will be stored in the next iteration
                new_write_idx = write_idx + 2

                return obs_updated, act_updated, term, valid_updated, new_write_idx, write_idx

            def handle_normal(_):

                def after_reset(_):
                    # Store current observation and make it valid (reset state)
                    obs = obs_updated.at[write_idx-1].set(state_idx)
                    valid = valid_updated.at[write_idx-1].set(1)
                    return obs, valid

                def normal_step(_):
                    # No special handling needed
                    return obs_updated, valid_updated

                # Check if we're just after a reset (valid[write_idx-1] == 0)
                # This means the previous position was invalid (after terminal) 
                after_terminal = valid_updated[write_idx-1] == 0
                obs, valid = jax.lax.cond(
                    after_terminal,
                    after_reset,
                    normal_step,
                    operand=None
                )

                # Increment write index by 1
                new_write_idx = write_idx + 1

                return obs, act_updated, term_updated, valid, new_write_idx, write_idx

            return jax.lax.cond(done, handle_done, handle_normal, operand=None)

        # Update all environments
        updates = jax.vmap(update_single_env)(jnp.arange(num_envs))
        observations, actions, terminals, valids, write_indices, last_written = updates

        # Reset environments if done
        reset_env_states = vmap_reset(num_envs)(rng, next_env_states, dones)

        # Return updated state
        runner_state = (transition_counts, reset_env_states, observations, actions,
                       terminals, valids, write_indices, last_written, rng)

        return runner_state, None

    # Initialize environments
    rng, reset_rng = jax.random.split(rng)
    env_states = vmap_init_reset(num_envs)(reset_rng)

    # Get initial state indices and store at position 0
    initial_state_indices = jax.vmap(lambda state: env.get_state_representation(state))(env_states)
    observations = observations.at[:, 0].set(initial_state_indices)
    valids = valids.at[:, 0].set(1)   # Initial states are valid (vacuously)

    # Start write_indices at 1 since we already wrote initial states at position 0
    write_indices = jnp.ones((num_envs,), dtype=jnp.int32)
    last_written = jnp.zeros((num_envs,), dtype=jnp.int32)

    # Initialize runner state
    runner_state = (transition_counts, env_states, observations, actions,
                   terminals, valids, write_indices, last_written, rng)
    
    # Run collection loop using scan
    runner_state, _ = jax.lax.scan(
        _update_step, runner_state, jnp.arange(num_steps)
    )

    # Extract final results
    (final_transition_counts, _, final_observations, final_actions,
     final_terminals, final_valids, _, final_written_indices, _) = runner_state
    
    # Fill terminals at the last indices: returns 1 if it was 1, otherwise 0
    final_terminals = final_terminals.at[jnp.arange(num_envs), final_written_indices].apply(
        lambda x: jnp.where(x == 1, 1, 0)
    )

    # Package episodes in OGBench format
    episodes = {
        'observations': final_observations,
        'actions': final_actions,
        'terminals': final_terminals,
        'valids': final_valids,
        'lengths': final_written_indices,  # Written indices represent final lengths
    }

    # Metrics
    metrics = {
        "total_transitions": int(final_transition_counts.sum()),
        "num_envs": num_envs,
        "num_steps": num_steps,
    }

    return final_transition_counts, episodes, metrics


def collect_transition_counts_batched_portals(
    base_env,
    portal_configs,
    portal_masks,
    num_rollouts_per_env,
    num_steps,
    canonical_states,
    seed=42
):
    """
    Collect transition counts for multiple different portal environments in parallel.

    This processes num_portal_envs different environments simultaneously, where each
    environment has different portal configurations. Returns separate transition counts
    for each environment. Works only with canonical (free) states, excluding obstacles.

    Args:
        base_env: Base GridWorld environment (without portals)
        portal_configs: Array of shape [num_portal_envs, max_portals, 3]
                       where each portal is [source_state, action, dest_state]
                       States are in CANONICAL space (not full state space)
        portal_masks: Boolean array of shape [num_portal_envs, max_portals]
                     indicating which portals are valid
        num_rollouts_per_env: Number of parallel rollouts per portal environment
        num_steps: Number of steps per rollout
        canonical_states: Array of free (non-obstacle) state indices
        seed: Random seed

    Returns:
        transition_counts: Array of shape [num_portal_envs, num_canonical_states, num_actions, num_canonical_states]
        metrics: Dictionary with collection metrics
    """
    num_portal_envs = portal_configs.shape[0]
    max_portals = portal_configs.shape[1]
    num_actions = base_env.action_space
    width = base_env.width
    height = base_env.height
    num_canonical_states = len(canonical_states)

    # Create mapping: full state index -> canonical state index
    num_states_full = width * height
    full_to_canonical = jnp.full(num_states_full, -1, dtype=jnp.int32)
    for canonical_idx, full_idx in enumerate(canonical_states):
        full_to_canonical = full_to_canonical.at[int(full_idx)].set(canonical_idx)

    # Get base environment parameters (obstacles, etc.)
    obstacles = base_env.obstacles if base_env.has_obstacles else jnp.array([])
    has_obstacles = base_env.has_obstacles

    # Initialize random key
    rng = jax.random.PRNGKey(seed)

    # Initialize transition count matrix for all portal environments
    # Shape: [num_portal_envs, num_canonical_states, num_actions, num_canonical_states]
    transition_counts = jnp.zeros(
        (num_portal_envs, num_canonical_states, num_actions, num_canonical_states),
        dtype=jnp.float32
    )

    # Create canonical to full mapping for portal destinations
    # canonical_states is already in full state space, so we can use it directly
    canonical_to_full = jnp.array(canonical_states, dtype=jnp.int32)

    # Define portal-aware step function for a single agent
    def step_with_portals(position, action, portal_sources, portal_actions, portal_dests, portal_mask):
        """Step function that checks portals for one agent in one portal environment.

        Portal sources/dests are in CANONICAL state space.
        """
        # Get current state index in full space, then convert to canonical
        current_state_idx_full = position[1] * width + position[0]
        current_state_idx_canonical = full_to_canonical[current_state_idx_full]

        # Check if any portal matches (state, action) - using canonical indices
        matches = (portal_sources == current_state_idx_canonical) & (portal_actions == action) & portal_mask
        has_portal = jnp.any(matches)

        # Get destination (first matching portal or -1)
        portal_idx = jnp.argmax(matches)
        portal_dest_canonical = portal_dests[portal_idx]

        # Convert canonical dest to full state index for position calculation
        portal_dest_full = canonical_to_full[portal_dest_canonical]

        # Apply portal or normal physics
        def use_portal():
            dest_x = portal_dest_full % width
            dest_y = portal_dest_full // width
            return jnp.array([dest_x, dest_y], dtype=position.dtype)

        def use_normal_physics():
            # Action effects: up, right, down, left
            action_effects = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=position.dtype)
            action_effect = action_effects[action]

            new_position_raw = position + action_effect
            new_position = jnp.clip(
                new_position_raw,
                jnp.array([0, 0], dtype=position.dtype),
                jnp.array([width - 1, height - 1], dtype=position.dtype)
            )

            # Check obstacles
            if has_obstacles:
                is_obstacle = jnp.any(jnp.all(new_position == obstacles, axis=1))
                new_position = jnp.where(is_obstacle, position, new_position)

            return new_position

        next_position = jax.lax.cond(has_portal, use_portal, use_normal_physics)
        return next_position

    # Vmap over agents for one portal environment
    step_batch_agents = jax.vmap(
        step_with_portals,
        in_axes=(0, 0, None, None, None, None)
    )

    # Vmap over portal environments
    step_batch_envs = jax.vmap(
        step_batch_agents,
        in_axes=(0, 0, 0, 0, 0, 0)
    )

    # Define single update step
    def _update_step(runner_state, unused):
        transition_counts, positions, rng = runner_state
        # positions shape: [num_portal_envs, num_rollouts_per_env, 2]

        # Split RNG
        rng, rng_act, rng_reset = jax.random.split(rng, 3)

        # Sample random actions for all agents in all environments
        actions = jax.random.randint(
            rng_act,
            shape=(num_portal_envs, num_rollouts_per_env),
            minval=0,
            maxval=num_actions,
        )

        # Get current state indices in full space
        state_indices_full = positions[:, :, 1] * width + positions[:, :, 0]  # [num_portal_envs, num_rollouts]

        # Convert to canonical space
        state_indices_canonical = full_to_canonical[state_indices_full]  # [num_portal_envs, num_rollouts]

        # Extract portal information for each environment (already in canonical space)
        portal_sources = portal_configs[:, :, 0]  # [num_portal_envs, max_portals]
        portal_actions = portal_configs[:, :, 1]  # [num_portal_envs, max_portals]
        portal_dests = portal_configs[:, :, 2]    # [num_portal_envs, max_portals]

        # Step all agents in all environments
        next_positions = step_batch_envs(
            positions, actions,
            portal_sources, portal_actions, portal_dests, portal_masks
        )

        # Get next state indices in full space, then convert to canonical
        next_state_indices_full = next_positions[:, :, 1] * width + next_positions[:, :, 0]
        next_state_indices_canonical = full_to_canonical[next_state_indices_full]

        # Accumulate transition counts for each portal environment separately
        # We need to update transition_counts[env_idx, s_canonical, a, s_next_canonical] for each transition
        def update_counts_for_env(env_idx, counts):
            """Update counts for a single portal environment."""
            env_state_indices = state_indices_canonical[env_idx]  # [num_rollouts]
            env_actions = actions[env_idx]
            env_next_state_indices = next_state_indices_canonical[env_idx]

            # Update this environment's count matrix (using canonical indices)
            env_counts = counts[env_idx]
            env_counts = env_counts.at[env_state_indices, env_actions, env_next_state_indices].add(1.0)
            return env_counts

        # Vmap over all portal environments to update their counts
        updated_counts = jax.vmap(update_counts_for_env, in_axes=(0, None))(
            jnp.arange(num_portal_envs), transition_counts
        )

        return (updated_counts, next_positions, rng), None

    # Initialize positions for all agents in all environments
    # Use random free positions, avoiding obstacles and portal sources
    rng, reset_rng = jax.random.split(rng)

    # For each portal environment, find states without portals and sample randomly
    initial_positions = []

    for env_idx in range(num_portal_envs):
        # Get portal sources for this environment (in canonical space)
        env_portal_sources = set()
        for portal_idx in range(max_portals):
            if portal_masks[env_idx, portal_idx]:
                source_canonical = int(portal_configs[env_idx, portal_idx, 0])
                env_portal_sources.add(source_canonical)

        # Find canonical states that are NOT portal sources
        valid_start_states = []
        for canonical_idx in range(num_canonical_states):
            if canonical_idx not in env_portal_sources:
                valid_start_states.append(canonical_idx)

        if len(valid_start_states) == 0:
            # Fallback: use all canonical states if somehow all have portals
            valid_start_states = list(range(num_canonical_states))

        # Sample random positions for this environment's agents
        rng, sample_rng = jax.random.split(rng)
        sampled_canonical = np.random.RandomState(int(sample_rng[0])).choice(
            valid_start_states,
            size=num_rollouts_per_env,
            replace=True
        )

        # Convert canonical indices to (x, y) positions
        env_positions = []
        for canonical_idx in sampled_canonical:
            full_state_idx = int(canonical_to_full[canonical_idx])
            x = full_state_idx % width
            y = full_state_idx // width
            env_positions.append([x, y])

        initial_positions.append(env_positions)

    # Convert to JAX array: [num_portal_envs, num_rollouts_per_env, 2]
    initial_positions = jnp.array(initial_positions, dtype=jnp.int32)

    runner_state = (transition_counts, initial_positions, rng)

    # Run collection loop
    runner_state, _ = jax.lax.scan(_update_step, runner_state, None, num_steps)


    final_counts, _, _ = runner_state

    metrics = {
        "num_portal_envs": num_portal_envs,
        "num_rollouts_per_env": num_rollouts_per_env,
        "num_steps": num_steps,
        "total_transitions_per_env": num_rollouts_per_env * num_steps,
    }

    return final_counts, metrics


def generate_portal_environments_data(
    base_env_name: str = "GridRoom-4",
    num_envs: int = 10,
    num_portals: int = 10,
    num_rollouts_per_env: int = 100,
    num_steps: int = 100,
    seed: int = 42
) -> Tuple[jnp.ndarray, Dict]:
    """
    Generate transition data from multiple portal environments.

    Args:
        base_env_name: Name of base environment text file
        num_envs: Number of different portal environments
        num_portals: Number of portals per environment
        num_rollouts_per_env: Number of rollouts per environment
        num_steps: Number of steps per rollout
        seed: Random seed

    Returns:
        Tuple of (transition_counts, metadata)
    """
    from src.envs.env import create_environment_from_text
    from src.envs.portal_gridworld import create_random_portal_env
    from src.utils.envs import get_canonical_free_states

    print(f"Generating data from {num_envs} portal environments...")
    print(f"  Base environment: {base_env_name}")
    print(f"  Portals per env: {num_portals}")
    print(f"  Rollouts per env: {num_rollouts_per_env}")
    print(f"  Steps per rollout: {num_steps}")

    base_env = create_environment_from_text(file_name=base_env_name, max_steps=num_steps)
    num_states = base_env.observation_space_size()
    num_actions = base_env.action_space_size()

    print(f"  Total states: {num_states}")
    print(f"  Actions: {num_actions}")

    canonical_states = get_canonical_free_states(base_env)
    num_canonical_states = len(canonical_states)
    print(f"  Free states: {num_canonical_states}")

    rng = np.random.RandomState(seed)
    portal_configs_list = []
    portal_masks_list = []

    for env_idx in range(num_envs):
        env_seed = rng.randint(0, 2**31)
        portal_env = create_random_portal_env(
            base_env=base_env,
            num_portals=num_portals,
            seed=env_seed
        )

        portals = []
        for (source_state, action), dest_state in portal_env.portals.items():
            source_canonical = jnp.where(canonical_states == source_state, size=1, fill_value=-1)[0][0]
            dest_canonical = jnp.where(canonical_states == dest_state, size=1, fill_value=-1)[0][0]
            portals.append([int(source_canonical), int(action), int(dest_canonical)])

        while len(portals) < num_portals:
            portals.append([0, 0, 0])

        portal_configs_list.append(portals[:num_portals])
        portal_masks_list.append([True] * len(portal_env.portals) + [False] * (num_portals - len(portal_env.portals)))

    portal_configs = jnp.array(portal_configs_list, dtype=jnp.int32)
    portal_masks = jnp.array(portal_masks_list, dtype=bool)

    print(f"\nCollecting transitions...")
    transition_counts, metrics = collect_transition_counts_batched_portals(
        base_env=base_env,
        portal_configs=portal_configs,
        portal_masks=portal_masks,
        num_rollouts_per_env=num_rollouts_per_env,
        num_steps=num_steps,
        canonical_states=canonical_states,
        seed=seed
    )

    print(f"  Collected transition counts: {transition_counts.shape}")

    obstacles = []
    if base_env.has_obstacles:
        obstacles = [(int(obs[0]), int(obs[1])) for obs in base_env.obstacles]

    metadata = {
        "num_envs": num_envs,
        "num_states": num_states,
        "num_canonical_states": num_canonical_states,
        "num_actions": num_actions,
        "canonical_states": canonical_states,
        "base_env_name": base_env_name,
        "num_portals": num_portals,
        "grid_width": base_env.width,
        "grid_height": base_env.height,
        "obstacles": obstacles,
        "first_env_portals": portal_configs_list[0] if num_envs > 0 else [],
    }

    return transition_counts, metadata
