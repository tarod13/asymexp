import jax.numpy as jnp


def get_transition_matrix(
    transition_counts: jnp.ndarray,
    make_stochastic: bool = True,
) -> jnp.ndarray:
    """
    Build transition matrix from counts (non-symmetrized).

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]
        make_stochastic: Whether to row-normalize to get proper transition probabilities
    Returns:
        Transition matrix of shape [num_states, num_states]
    """
    # Sum over actions if needed
    if len(transition_counts.shape) == 3:
        transition_matrix = jnp.sum(transition_counts, axis=1)  # [num_states, num_states]
    else:
        transition_matrix = transition_counts

    if make_stochastic:
        row_sums = jnp.sum(transition_matrix.clip(1e-8), axis=1, keepdims=True)
        transition_matrix = transition_matrix.clip(1e-8) / row_sums

    return transition_matrix


def compute_successor_representation(
    transition_matrix: jnp.ndarray,
    gamma: float,
    max_horizon: int = None,
) -> jnp.ndarray:
    """
    Compute the successor representation SR_γ = (I - γP)^(-1) or finite-horizon version.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor
        max_horizon: If provided, compute finite-horizon SR_γ^(T) = Σ_{k=0}^{T} γ^k P^k
                     Otherwise, compute infinite-horizon SR_γ = (I - γP)^(-1)

    Returns:
        Successor representation of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    if max_horizon is None:
        # Infinite-horizon: SR_γ = (I - γP)^(-1)
        sr_matrix = jnp.linalg.inv(identity - gamma * transition_matrix)
    else:
        # Finite-horizon: SR_γ^(T) = Σ_{k=0}^{T} γ^k P^k
        # Using closed form: SR_γ^(T) = (I - γP)^{-1}(I - γ^{T+1}P^{T+1})
        # Compute P^{T+1} using built-in matrix power (uses efficient algorithm)
        P_power_T_plus_1 = jnp.linalg.matrix_power(
            transition_matrix, max_horizon + 1
        )

        # Apply finite geometric series formula (left multiplication is mathematically correct)
        gamma_power = gamma ** (max_horizon + 1)
        sr_matrix = jnp.linalg.solve(
            identity - gamma * transition_matrix, 
            identity - gamma_power * P_power_T_plus_1,
        )

    return sr_matrix


def compute_sampling_distribution(
    transition_counts: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the empirical sampling distribution from transition counts.

    Args:
        transition_counts: Shape [num_states, num_actions, num_states] or [num_states, num_states]

    Returns:
        Diagonal matrix D where D_{ii} is the empirical sampling probability of state i
    """
    # Sum over actions and next states to get visit counts per state
    if len(transition_counts.shape) == 3:
        # Sum over actions and next states
        state_visit_counts = jnp.sum(transition_counts, axis=(1, 2))
    else:
        # Sum over next states
        state_visit_counts = jnp.sum(transition_counts, axis=1)

    # Normalize to get probabilities
    total_visits = jnp.sum(state_visit_counts)
    sampling_probs = state_visit_counts / total_visits

    # Create diagonal matrix
    D = jnp.diag(sampling_probs)

    return D


def compute_laplacian(
    transition_matrix: jnp.ndarray,
    gamma: float,
    delta: float = 0.0,
    max_horizon: int = None,
) -> jnp.ndarray:
    """
    Compute the non-symmetric Laplacian L = (1+δ)I - (1-γ)P·SR_γ.

    This definition matches what the episodic replay buffer approximates with
    geometric sampling (k >= 1). The transition matrix P applied to SR_γ gives
    the expected discounted future occupancy starting from the next state.

    The δ parameter shifts eigenvalues away from zero, improving numerical stability.
    With δ > 0, the smallest eigenvalue is δ instead of 0.

    Args:
        transition_matrix: Shape [num_states, num_states], stochastic transition matrix P
        gamma: Discount factor used in successor representation
        delta: Eigenvalue shift parameter (default 0.0). Recommended: 0.1 for stability.
        max_horizon: If provided, use finite-horizon SR_γ^(T) instead of infinite-horizon

    Returns:
        Non-symmetric Laplacian of shape [num_states, num_states]
    """
    num_states = transition_matrix.shape[0]
    identity = jnp.eye(num_states)

    # Compute successor representation (infinite or finite-horizon)
    sr_matrix = compute_successor_representation(transition_matrix, gamma, max_horizon)

    # Compute Laplacian: L = (1+δ)I - (1-γ)P·SR_γ
    laplacian = (1 + delta) * identity - (1 - gamma) * (transition_matrix @ sr_matrix)

    return laplacian