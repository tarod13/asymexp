import numpy as np
import jax.numpy as jnp

from src.config.ded_clf import Args
from src.data_collection import collect_transition_counts_and_episodes
from src.utils.episodic_replay_buffer import EpisodicReplayBuffer
from src.utils.envs import get_canonical_free_states

from src.envs.door_gridworld import (
    create_door_gridworld_from_base,
    create_random_doors,
)

from src.utils.laplacian import (
    get_transition_matrix,
    compute_sampling_distribution,
    compute_eigendecomposition,
    compute_laplacian,
)


def state_idx_to_xy(state_idx: int, width: int) -> tuple:
    """Convert state index to (x,y) coordinates."""
    y = state_idx // width
    x = state_idx % width
    return x, y


def create_replay_buffer_only(env, canonical_states, args: Args):
    """
    Create replay buffer by collecting episodes (without recomputing eigenvectors).
    Used when resuming training to avoid redundant computation.

    Args:
        env: Environment to collect data from
        canonical_states: Array of free state indices
        args: Training arguments

    Returns:
        replay_buffer: EpisodicReplayBuffer filled with collected episodes
    """
    print("Collecting episodes for replay buffer...")
    num_states = env.width * env.height

    # Collect transition counts and episodes
    transition_counts_full, raw_episodes, metrics = collect_transition_counts_and_episodes(
        env=env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_states=num_states,
        seed=args.seed,
    )
    print(f"Collected {metrics['total_transitions']} transitions.")

    # Initialize replay buffer
    max_valid_length = int(raw_episodes['lengths'].max()) + 1
    replay_buffer = EpisodicReplayBuffer(
        max_episodes=args.num_envs,
        max_episode_length=max_valid_length,
        observation_type='canonical_state',
        seed=args.seed
    )

    # Convert and add episodes to buffer using vectorized method
    print("\nPopulating replay buffer with episodes...")
    replay_buffer.populate_with_tabular_episodes(
        canonical_states=canonical_states,
        episodes=raw_episodes,
    )
    print(f"  Added {len(replay_buffer)} trajectory sequences to replay buffer")
    return replay_buffer


def collect_data_and_compute_eigenvectors(env, args: Args):
    """
    Collect transition data and compute ground truth complex eigenvectors.

    Computes the non-symmetric Laplacian: L = I - (1-γ)P·SR_γ
    This Laplacian admits complex eigenvalues and distinct left/right eigenvectors.
    The definition matches what the graph loss approximates via geometric sampling.

    Args:
        env: Base environment (possibly with doors already applied)

    Returns:
        laplacian_matrix: Non-symmetric Laplacian L = I - (1-γ)P·SR_γ
        eigendecomp: Dictionary with complex eigenvalues and left/right eigenvectors
        state_coords: Array of (x,y) coordinates for each state
        canonical_states: Array of free state indices
        sampling_probs: 1D array of empirical sampling probabilities for each canonical state
        door_config: Door configuration (if use_doors=True)
        data_env: The environment used for data collection (with doors if applicable)
        replay_buffer: Episodic replay buffer filled with collected episodes
    """
    print("Collecting transition data...")
    num_states = env.width * env.height

    # Get canonical (free) states from base environment
    canonical_states = get_canonical_free_states(env)
    num_canonical = len(canonical_states)
    print(f"Number of free states: {num_canonical} (out of {num_states} total)")

    # Create door environment if requested
    door_config = None
    data_env = env  # Default to base environment

    if args.use_doors:
        print(f"\nCreating {args.num_doors} irreversible doors...")
        door_config = create_random_doors(
            env,
            canonical_states,
            num_doors=args.num_doors,
            seed=args.door_seed
        )
        print(f"  Created {door_config['num_doors']} doors (out of {door_config['total_reversible']} reversible transitions)")

        # Create environment with doors in the dynamics
        data_env = create_door_gridworld_from_base(env, door_config['doors'], canonical_states)
        print("  Created DoorGridWorld environment with irreversible transitions")

    # Collect transition counts and episodes in a single efficient pass
    print("Collecting transition data and episodes...")
    transition_counts_full, raw_episodes, metrics = collect_transition_counts_and_episodes(
        env=data_env,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_states=num_states,
        seed=args.seed,
    )

    print(f"Collected {metrics['total_transitions']} transitions.")

    # Initialize replay buffer
    # Find max valid episode length for buffer sizing
    max_valid_length = int(raw_episodes['lengths'].max()) + 1
    replay_buffer = EpisodicReplayBuffer(
        max_episodes=args.num_envs,
        max_episode_length=max_valid_length,
        observation_type='canonical_state',
        seed=args.seed
    )

    # Convert and add episodes to buffer using vectorized method
    print("\nPopulating replay buffer with episodes...")
    replay_buffer.populate_with_tabular_episodes(
        canonical_states=canonical_states,
        episodes=raw_episodes,
    )
    print(f"  Added {len(replay_buffer)} trajectory sequences to replay buffer")

    # Extract canonical state subspace
    transition_counts = transition_counts_full[jnp.ix_(canonical_states, jnp.arange(env.action_space), canonical_states)]

    # Build transition matrix (non-symmetrized)
    print("\nBuilding transition matrix...")
    transition_matrix = get_transition_matrix(transition_counts)

    # Compute sampling distribution D
    print("Computing empirical sampling distribution...")
    sampling_distribution = compute_sampling_distribution(transition_counts)  # Shape: (num_canonical, num_canonical)
    sampling_probs = jnp.diag(sampling_distribution)  # Shape: (num_canonical,)
    print(f"  Sampling prob range: [{sampling_probs.min():.6f}, {sampling_probs.max():.6f}]")
    print(f"  Sampling prob std: {sampling_probs.std():.6f}")
    
    # Compute non-symmetric Laplacian for complex eigenvectors
    print(f"\nComputing non-symmetric Laplacian with gamma={args.gamma}, delta={args.delta}...")
    print(f"  L = (1+δ)I - (1-γ)P·SR_γ (matches geometric sampling with k >= 1)")

    laplacian_matrix = compute_laplacian(transition_matrix, args.gamma, delta=args.delta)
    eigendecomp = compute_eigendecomposition(
        laplacian_matrix,
        k=args.num_eigenvector_pairs,
        ascending=True  # For Laplacians, we want smallest eigenvalues
    )

    print(f"\nSmallest {args.num_eigenvector_pairs} eigenvalues (complex):")
    print("  Eigenvalue (real + imag)")
    for i in range(args.num_eigenvector_pairs):
        ev_real = eigendecomp['eigenvalues_real'][i]
        ev_imag = eigendecomp['eigenvalues_imag'][i]
        print(f"  λ_{i}: {ev_real:.6f} + {ev_imag:.6f}i")

    # Print ranges of eigenvector values to debug visualization issues
    print(f"\nEigenvector value ranges (first {args.num_eigenvector_pairs} eigenvectors):")
    for i in range(args.num_eigenvector_pairs):
        # Right eigenvectors
        right_real = eigendecomp['right_eigenvectors_real'][:, i]
        right_imag = eigendecomp['right_eigenvectors_imag'][:, i]
        print(f"  Right eigenvector {i} (real): min={np.min(right_real):.6f}, max={np.max(right_real):.6f}, mean={np.mean(right_real):.6f}")
        print(f"  Right eigenvector {i} (imag): min={np.min(right_imag):.6f}, max={np.max(right_imag):.6f}, mean={np.mean(right_imag):.6f}")

        # Left eigenvectors
        left_real = eigendecomp['left_eigenvectors_real'][:, i]
        left_imag = eigendecomp['left_eigenvectors_imag'][:, i]
        print(f"  Left eigenvector {i} (real):  min={np.min(left_real):.6f}, max={np.max(left_real):.6f}, mean={np.mean(left_real):.6f}")
        print(f"  Left eigenvector {i} (imag):  min={np.min(left_imag):.6f}, max={np.max(left_imag):.6f}, mean={np.mean(left_imag):.6f}")
        print()

    # Create state coordinate mapping (only for canonical/free states)
    state_coords = []
    for state_idx in canonical_states:
        state_idx = int(state_idx)
        x, y = state_idx_to_xy(state_idx, env.width)
        state_coords.append([x, y])
    state_coords = jnp.array(state_coords, dtype=jnp.float32)

    # Center and scale coordinates to approximately [-1, 1] range
    # Center around grid center
    center = jnp.array([env.width / 2.0, env.height / 2.0], dtype=jnp.float32)
    state_coords = state_coords - center

    # Scale by half the maximum dimension (to get range approximately [-1, 1])
    scale = max(env.width, env.height) / 2.0
    state_coords = state_coords / scale

    print(f"\nCoordinate normalization:")
    print(f"  Center: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"  Scale: {scale:.2f}")
    print(f"  Coordinate range: x=[{state_coords[:, 0].min():.3f}, {state_coords[:, 0].max():.3f}], "
          f"y=[{state_coords[:, 1].min():.3f}, {state_coords[:, 1].max():.3f}]")

    return laplacian_matrix, eigendecomp, state_coords, canonical_states, sampling_probs, door_config, data_env, replay_buffer, transition_matrix