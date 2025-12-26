"""
Episodic replay buffer for sampling state transitions from collected episodes.

Samples transitions (s_t, s_{t+k}) where k is drawn from a truncated geometric distribution.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, List


class EpisodicReplayBuffer:
    """
    Replay buffer that stores episodes and samples transitions with geometric time offsets.

    Args:
        episodes: List of episodes, where each episode is a sequence of state indices
        gamma: Decay parameter for geometric distribution (higher = prefer shorter gaps)
        max_offset: Maximum time offset between s_t and s_{t+k}
    """

    def __init__(
        self,
        episodes: List[np.ndarray],
        gamma: float = 0.99,
        max_offset: int = None
    ):
        self.episodes = episodes
        self.gamma = gamma
        self.max_offset = max_offset

        # Compute episode lengths
        self.episode_lengths = np.array([len(ep) for ep in episodes])
        self.num_episodes = len(episodes)

        # Precompute truncated geometric probabilities
        self._precompute_geometric_probs()

        print(f"Replay buffer created with {self.num_episodes} episodes")
        print(f"  Mean episode length: {self.episode_lengths.mean():.1f}")
        print(f"  Total transitions: {self.episode_lengths.sum()}")

    def _precompute_geometric_probs(self):
        """Precompute probabilities for truncated geometric distribution."""
        if self.max_offset is None:
            self.max_offset = int(self.episode_lengths.max())

        # Geometric distribution: P(k) = (1 - gamma) * gamma^(k-1)
        k_values = np.arange(1, self.max_offset + 1)
        probs = (1 - self.gamma) * (self.gamma ** (k_values - 1))

        # Normalize to ensure it sums to 1 (truncated distribution)
        self.offset_probs = probs / probs.sum()

    def sample_batch(
        self,
        batch_size: int,
        rng_key: jax.random.PRNGKey
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions (s_t, s_{t+k}) from episodes.

        Args:
            batch_size: Number of transitions to sample
            rng_key: JAX random key

        Returns:
            state_indices: Array of starting state indices, shape (batch_size,)
            next_state_indices: Array of target state indices, shape (batch_size,)
        """
        rng_key, episode_key, position_key, offset_key = jax.random.split(rng_key, 4)

        state_indices = []
        next_state_indices = []

        for i in range(batch_size):
            # Sample an episode (weighted by length for uniform state sampling)
            episode_weights = self.episode_lengths / self.episode_lengths.sum()
            episode_idx = np.random.choice(self.num_episodes, p=episode_weights)
            episode = self.episodes[episode_idx]
            episode_len = len(episode)

            # Sample a starting position in the episode
            # We need to ensure there's room for at least offset=1
            if episode_len <= 1:
                # Episode too short, sample same state twice
                t = 0
                s_t = episode[t]
                s_next = episode[t]
            else:
                # Sample time offset k from truncated geometric
                max_valid_offset = min(self.max_offset, episode_len - 1)
                valid_probs = self.offset_probs[:max_valid_offset]
                valid_probs = valid_probs / valid_probs.sum()  # Re-normalize

                k = np.random.choice(max_valid_offset, p=valid_probs) + 1

                # Sample starting position ensuring we don't go past episode end
                max_t = episode_len - k
                t = np.random.randint(0, max_t + 1)

                s_t = episode[t]
                s_next = episode[t + k]

            state_indices.append(s_t)
            next_state_indices.append(s_next)

        return np.array(state_indices), np.array(next_state_indices)


def collect_episodes(
    env,
    num_episodes: int,
    max_steps_per_episode: int,
    seed: int = 42
) -> List[np.ndarray]:
    """
    Collect episodes by running random policy in the environment.

    Args:
        env: The environment to collect from
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        seed: Random seed

    Returns:
        List of episodes, where each episode is an array of state indices
    """
    import jax.random as jr

    episodes = []
    rng_key = jr.PRNGKey(seed)

    for ep_idx in range(num_episodes):
        rng_key, reset_key = jr.split(rng_key)

        # Reset environment
        state = env.reset(reset_key)
        episode_states = [int(state)]

        for step in range(max_steps_per_episode):
            rng_key, action_key, step_key = jr.split(rng_key, 3)

            # Random action
            action = jr.randint(action_key, (), 0, env.action_space)

            # Step
            next_state, reward, done, info = env.step(step_key, state, action)
            episode_states.append(int(next_state))

            state = next_state

            if done:
                break

        episodes.append(np.array(episode_states, dtype=np.int32))

    return episodes
