from typing import Dict, Type, TypeVar, Generic, List, Tuple
import collections
import itertools
import dataclasses
import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

T = TypeVar("T", np.ndarray, jnp.ndarray)

@dataclasses.dataclass
class EpisodeBatch(Generic[T]):
    """A container for batchable replayed episodes"""
    obs: T
    next_obs: T

    def __post_init__(self) -> None:
        # some security to be removed later
        assert isinstance(self.obs, (np.ndarray, jnp.ndarray))
        assert isinstance(self.next_obs, (np.ndarray, jnp.ndarray))

@dataclasses.dataclass
class LocalizedEpisodeBatch(Generic[T]):
    """A container for batchable replayed episodes"""
    obs: T
    next_obs: T
    action: T
    location: T
    next_location: T
    grid_location: T
    next_grid_location: T

    def __post_init__(self) -> None:
        # some security to be removed later
        assert isinstance(self.obs, (np.ndarray, jnp.ndarray))
        assert isinstance(self.next_obs, (np.ndarray, jnp.ndarray))
        assert isinstance(self.action, (np.ndarray, jnp.ndarray))
        assert isinstance(self.location, (np.ndarray, jnp.ndarray))
        assert isinstance(self.next_location, (np.ndarray, jnp.ndarray))
        assert isinstance(self.grid_location, (np.ndarray, jnp.ndarray))
        assert isinstance(self.next_grid_location, (np.ndarray, jnp.ndarray))


def discounted_sampling(ranges, discount):   # TODO: Consider if necessary.
    """Draw samples from the discounted distribution over 0, ...., n - 1, 
    where n is a range. The input ranges is a batch of such n`s.

    The discounted distribution is defined as
    p(y = i) = (1 - discount) * discount^i / (1 - discount^n).

    This function implement inverse sampling. We first draw
    seeds from uniform[0, 1) then pass them through the inverse cdf
    floor[ log(1 - (1 - discount^n) * seeds) / log(discount) ]
    to get the samples.
    """
    assert np.min(ranges) >= 1
    assert discount >= 0 and discount <= 1
    seeds = np.random.uniform(size=ranges.shape)
    if discount == 0:
        samples = np.zeros_like(seeds, dtype=np.int64)
    elif discount == 1:
        samples = np.floor(seeds * ranges).astype(np.int64)
    else:
        samples = (np.log(1 - (1 - np.power(discount, ranges)) * seeds) 
                / np.log(discount))
        samples = np.floor(samples).astype(np.int64)
    return samples


def uniform_sampling(ranges):   # TODO: Consider if necessary.
    return discounted_sampling(ranges, discount=1.0)


class EpisodicReplayBuffer:
    """Only store full episodes.
    
    Sampling returns EpisodicStep objects.
    """

    def __init__(self, max_episodes, max_episode_length, observation_type, seed):
        self._max_episodes = max_episodes
        self._max_episode_length = max_episode_length
        self._idx = 0
        self._episodes: Dict[str, np.ndarray] = collections.defaultdict()
        self._full = False
        self._episodes_length = np.array([0] * max_episodes, dtype=np.int64)
        self._seed = seed   # Should this be here?
        self._observation_type = observation_type   # Should this be here?

    def __len__(self) -> int:
        return self._max_episodes if self._full else self._idx
    
    def _get_episode_lengths(self, idx):
        return self._episodes_length[idx]

    def reset(self):
        self._idx = 0
        self._full = False
        self._episodes = collections.defaultdict()
        self._episodes_length = np.array([0] * self._max_episodes, dtype=np.int64)
    
    def _transform_observations(self, observations, env_info):
        # If observation_type is 'canonical_state', return raw indices without transformation
        if self._observation_type == 'canonical_state':
            return observations.flatten()

        if observations.dtype == "uint8":
            observations = observations.astype(np.float32) / 255.0
        elif observations.dtype == "int32":
            n_idx = observations.shape[-1]

            if n_idx == 1:
                obs_idx = observations.flatten()
                observations = np.zeros((observations.shape[0], env_info.get('n'),), dtype=np.float32)
                observations[np.arange(observations.shape[0]), obs_idx] = 1.0
            elif n_idx == 3:
                n = env_info.get('nx') + env_info.get('ny') + env_info.get('nt')
                obs_x_idx = observations[:,0]
                obs_y_idx = env_info.get('nx') + observations[:,1]
                obs_t_idx = env_info.get('nx') + env_info.get('ny') + observations[:,2]
                observations = np.zeros((observations.shape[0], n), dtype=np.float32)
                observations[np.arange(observations.shape[0]), obs_x_idx] = 1.0
                observations[np.arange(observations.shape[0]), obs_y_idx] = 1.0
                observations[np.arange(observations.shape[0]), obs_t_idx] = 1.0
            else:
                raise ValueError("Unknown observation type.")

        return observations

    def add_episode(self, episode: Dict):
        for step_component, component_list in episode.items():
            # Convert to numpy array
            component_array = np.array(component_list, dtype=component_list[0].dtype)

            n = component_array.shape[0]
            if step_component not in self._episodes:
                # The buffer is created with appropriate size
                _shape = component_array.shape
                if self._max_episode_length is not None:
                    _shape = (self._max_episode_length,) + _shape[1:]   # TODO: Check when max_episode_length is None
                self._episodes[step_component] = np.empty((self._max_episodes,) + _shape, dtype=component_array.dtype)

            # Add component to episode buffer in current index
            self._episodes[step_component][self._idx][:n] = component_array

            # PRE-COMPUTE TERMINAL INDICES (Optimization Approach 1)
            if step_component == 'terminals':
                # Find all terminal positions in this episode
                terminal_positions = np.where(component_array == 1)[0]

                # Create buffers for storing terminal indices if they don't exist
                if 'terminal_indices' not in self._episodes:
                    # Maximum possible terminals per episode (conservative estimate)
                    max_terminals = 100
                    self._episodes['terminal_indices'] = np.full(
                        (self._max_episodes, max_terminals), -1, dtype=np.int32
                    )
                    self._episodes['num_terminals'] = np.zeros(
                        self._max_episodes, dtype=np.int32
                    )

                # Store terminal positions for this episode
                num_terms = len(terminal_positions)
                if num_terms > 0:
                    # Ensure we don't exceed buffer size
                    max_terminals = self._episodes['terminal_indices'].shape[1]
                    num_terms_to_store = min(num_terms, max_terminals)
                    self._episodes['terminal_indices'][self._idx, :num_terms_to_store] = terminal_positions[:num_terms_to_store]
                    self._episodes['num_terminals'][self._idx] = num_terms_to_store
                else:
                    # No terminals in this episode
                    self._episodes['num_terminals'][self._idx] = 0

            # Save length of episode
            if step_component == 'obs':
                self._episodes_length[self._idx] = n

        self._idx = (self._idx + 1) % self._max_episodes
        self._full = self._full or self._idx == 0

    def sample(self, batch_size, discount, env_info={}):   # TODO: Consider if necessary.
        # Sample episodes
        episode_idx = np.random.randint(len(self), size=batch_size)

        # Sample transitions
        transition_ranges = self._get_episode_lengths(episode_idx)
        obs_idx = uniform_sampling(transition_ranges - 1)   # -1 to sample future observations. This assumes length of episode is at least 2.

        # Calculate remaining trajectory length using pre-computed terminal indices (VECTORIZED)
        if 'terminal_indices' in self._episodes:
            # APPROACH 1+2: Use pre-computed terminal indices with vectorized operations
            # Gather pre-computed terminal indices for all sampled episodes
            # Shape: (batch_size, max_terminals_per_episode)
            batch_term_indices = self._episodes['terminal_indices'][episode_idx]
            batch_num_terms = self._episodes['num_terminals'][episode_idx]

            # Create mask for valid terminals that are >= obs_idx
            # Broadcasting: (batch_size, max_terminals) >= (batch_size, 1)
            valid_mask = batch_term_indices >= obs_idx[:, None]

            # Also mask out invalid terminal slots (marked with -1)
            valid_mask &= (batch_term_indices >= 0)

            # Compute relative indices (distance from obs_idx)
            relative_term_indices = batch_term_indices - obs_idx[:, None]

            # Set invalid positions to a large number so they sort to the end
            relative_term_indices = np.where(valid_mask, relative_term_indices, 999999)

            # Sort to get first and second terminals at indices 0 and 1
            sorted_terms = np.sort(relative_term_indices, axis=1)

            # Count how many valid terminals each sample has
            num_valid_terms = np.sum(valid_mask, axis=1)

            # Vectorized selection using np.where
            max_durations = np.where(
                num_valid_terms >= 2,
                sorted_terms[:, 1],  # Use second terminal if available
                np.where(
                    num_valid_terms >= 1,
                    sorted_terms[:, 0],  # Use first terminal if available
                    transition_ranges - obs_idx - 1  # Default: use remaining episode length
                )
            )
        elif 'terminals' in self._episodes:
            # FALLBACK: Old sequential implementation (if terminals exist but not pre-computed)
            # This should not be reached if add_episode() is working correctly
            max_durations = np.zeros(batch_size, dtype=np.int32)
            for i in range(batch_size):
                ep_idx = episode_idx[i]
                start_idx = obs_idx[i]
                ep_length = transition_ranges[i]

                terminals = self._episodes['terminals'][ep_idx, start_idx:ep_length]
                terminal_indices = np.where(terminals == 1)[0]

                if len(terminal_indices) > 0:
                    if len(terminal_indices) > 1:
                        max_durations[i] = terminal_indices[1]
                    else:
                        max_durations[i] = terminal_indices[0]
                else:
                    max_durations[i] = ep_length - start_idx - 1
        else:
            # No terminals available, use full episode length
            max_durations = transition_ranges - obs_idx - 1

        transition_durations = discounted_sampling(max_durations, discount=discount) + 1   # +1 because the minimal transition length is 1
        next_obs_idx = obs_idx + transition_durations

        # Get the samples
        obs = self._episodes['obs'][episode_idx, obs_idx]
        next_obs = self._episodes['obs'][episode_idx, next_obs_idx]

        # Transform the samples
        obs = self._transform_observations(obs, env_info)
        next_obs = self._transform_observations(next_obs, env_info)
            
        _have_location = 'loc' in self._episodes
        _have_grid_location = 'grid_loc' in self._episodes
        _have_action = 'action' in self._episodes
        if _have_location and _have_action:
            action = self._episodes['action'][episode_idx, obs_idx]
            location = self._episodes['loc'][episode_idx, obs_idx]
            next_location = self._episodes['loc'][episode_idx, next_obs_idx]
            if _have_grid_location:
                grid_location = self._episodes['grid_loc'][episode_idx, obs_idx]
                next_grid_location = self._episodes['grid_loc'][episode_idx, next_obs_idx]
            else:
                grid_location = None
                next_grid_location = None
            return LocalizedEpisodeBatch(
                obs=obs, next_obs=next_obs, action=action, 
                location=location, next_location=next_location,
                grid_location=grid_location, next_grid_location=next_grid_location
            )

        return EpisodeBatch(obs=obs, next_obs=next_obs)
    
    def get_component(self, component_name):
        if component_name == 'next_obs':
            return [
                steps[1:l] 
                for l, steps in zip(
                    self._episodes_length, self._episodes['obs'])
            ]
        else:
            return [
                steps[:l-1] 
                for l, steps in zip(
                    self._episodes_length, self._episodes[component_name])
            ]

    def get_mean_obs(self, env_info={}):
        obs_list = self.get_component('obs')
        obs_list = list(itertools.chain.from_iterable(obs_list))
        mean_obs = None
        mean_obs_squared = None
        for i, obs in enumerate(obs_list):
            obs = self._transform_observations(
                obs.reshape((1, *obs.shape)), env_info
            ).astype(float)[0]
            if mean_obs is None:
                mean_obs = obs
                mean_obs_squared = obs**2
            else:
                mean_obs += (obs - mean_obs) / (i + 1)
                mean_obs_squared += (obs**2 - mean_obs_squared) / (i + 1)
        std_obs = (mean_obs_squared - mean_obs**2).clip(1e-6, None)**0.5
        return mean_obs, std_obs

    def get_filtered_mean_obs(self, env_info={}):
        mean_obs, std_obs = self.get_mean_obs(env_info)

        obs_list = self.get_component('obs')
        obs_list = list(itertools.chain.from_iterable(obs_list))

        filtered_mean_obs = mean_obs.copy()
        pixel_counts = jnp.zeros_like(mean_obs)

        for obs in obs_list:
            obs = self._transform_observations(
                obs.reshape((1, *obs.shape)), env_info
            ).astype(float)[0]
            diff = jnp.absolute(obs - mean_obs).sum(-1, keepdims=True)
            close_pixels = diff < std_obs 
            non_initialized_pixels = pixel_counts == 0

            # Initialize pixels
            filtered_mean_obs = jnp.where(
                non_initialized_pixels & close_pixels,
                obs,
                filtered_mean_obs
            )
            pixel_counts = jnp.where(
                non_initialized_pixels & close_pixels,
                jnp.ones_like(pixel_counts),
                pixel_counts
            )

            filtered_mean_obs += close_pixels * (obs - filtered_mean_obs) / (pixel_counts + 1)
            pixel_counts += (close_pixels & jnp.invert(non_initialized_pixels)).astype(float)
        
        return filtered_mean_obs
    
    def get_background(self, env_info={}):
        background = self.get_filtered_mean_obs(env_info)
        # obs_list = self.get_component('obs')
        # obs_list = list(itertools.chain.from_iterable(obs_list))

        # found_background_mask = jnp.zeros_like(background).mean(-1, keepdims=True).astype(jnp.bool_)
        # all_pixels = False
        # i = 0
        # while not all_pixels and (i < len(obs_list)):
        #     obs = obs_list[i].astype(np.float32) / 255.0
        #     diff = obs - background
        #     all_pixel_diff = jnp.abs(diff).sum(-1, keepdims=True)
        #     similar_pixels_mask = all_pixel_diff < correct_threshold
        #     similar_pixels_mask = (
        #         similar_pixels_mask
        #         & np.invert(found_background_mask)
        #     )
        #     background = jnp.where(
        #         similar_pixels_mask.repeat(3, axis=-1), 
        #         obs,
        #         background,
        #     )
        #     found_background_mask = (
        #         found_background_mask | similar_pixels_mask
        #     )

        #     all_pixels = (found_background_mask == 1).all()
        #     i += 1
        return background
        
    def save_chunks(self, save_path, chunk_episodes=100, chunk_steps=1000):
        path_ = Path(save_path+"episode_batch_0.npz")
        path_.parent.mkdir(parents=True, exist_ok=True)
        num_chunks_per_episode = int(np.ceil(self._max_episode_length / chunk_steps))
        num_chunk_groups = int(np.ceil(len(self) / chunk_episodes))

        for i in range(num_chunk_groups):
            for j in range(num_chunks_per_episode):
                start_episode_idx = i * chunk_episodes
                end_episode_idx = (i + 1) * chunk_episodes
                start_step_idx = j * chunk_steps
                end_step_idx = (j + 1) * chunk_steps
                np.savez(
                    save_path + f"episode_batch_{self._seed}_{self._observation_type}_{self._max_episodes}_{self._max_episode_length}_{i}_{j}.npz",
                    **{
                        key: value[start_episode_idx:end_episode_idx, start_step_idx:end_step_idx] 
                        for key, value in self._episodes.items()
                    }
                )
            np.savez(save_path + "episode_lengths.npz", lengths=self._episodes_length)

    def load_chunks(self, load_path):
        # Get number of chunks from load_path
        num_chunk_groups = 0
        while True:
            if not os.path.exists(load_path + f"episode_batch_{self._seed}_{self._observation_type}_{self._max_episodes}_{self._max_episode_length}_{num_chunk_groups}_{0}.npz"):
                break
            num_chunk_groups += 1
        
        num_chunks_per_episode = 0
        while True:
            if not os.path.exists(load_path + f"episode_batch_{self._seed}_{self._observation_type}_{self._max_episodes}_{self._max_episode_length}_{0}_{num_chunks_per_episode}.npz"):
                break
            num_chunks_per_episode += 1

        episode_lengths = np.load(load_path + "episode_lengths.npz")["lengths"]

        for i in range(num_chunk_groups):
            data = []
            for j in range(num_chunks_per_episode):
                chunk_data = np.load(load_path + f"episode_batch_{self._seed}_{self._observation_type}_{i}_{j}.npz")
                data.append(chunk_data)
            data = {key: np.concatenate([chunk_data[key] for chunk_data in data], axis=1) for key in data[0].keys()}
            for key in data.keys():
                if key not in self._episodes:
                    self._episodes[key] = np.empty((self._max_episodes,) + data[key].shape[1:], dtype=data[key].dtype)
                start_idx = self._idx
                end_idx = (self._idx + len(data[key])) % self._max_episodes
                if start_idx < end_idx:
                    self._episodes[key][start_idx:end_idx] = data[key]
                    self._episodes_length[start_idx:end_idx] = episode_lengths[i * len(data[key]):(i + 1) * len(data[key])]
                else:
                    self._episodes[key][start_idx:] = data[key][:self._max_episodes - start_idx]
                    self._episodes[key][:end_idx] = data[key][self._max_episodes - start_idx:]
                    self._episodes_length[start_idx:] = episode_lengths[i * len(data[key]):i * len(data[key]) + self._max_episodes - start_idx]
                    self._episodes_length[:end_idx] = episode_lengths[i * len(data[key]) + self._max_episodes - start_idx:(i + 1) * len(data[key])]
                
                self._idx = (self._idx + len(data[key])) % self._max_episodes
                self._full = self._full or self._idx == 0


class GridBuffer:
    """Only store unique observations for evaluation.
    
    """
    def __init__(
            self, 
            observation_type: str, 
            seed: int, 
            max_obs_per_tile: int = 1000
        ):
        # Set the parameters
        self._observation_type = observation_type
        self._seed = seed
        self._max_obs_per_tile = max_obs_per_tile
        
        # Initialize the buffer
        self._observations: Dict[Dict[np.ndarray]] = collections.OrderedDict()
        
    def add_observation(self, localized_observation: Dict):
        # Extract components
        observation = localized_observation['observation']
        location = localized_observation['location']
        grid_location = localized_observation['grid_location']
        if isinstance(grid_location, np.ndarray):
            grid_location = tuple(coordinate for coordinate in grid_location)

        # The tile buffer is created with appropriate size if it does not exist
        if grid_location not in self._observations:
            observation_buffer = np.empty(
                (self._max_obs_per_tile,) + observation.shape, 
                dtype=observation.dtype,
            )
            location_buffer = np.empty(
                (self._max_obs_per_tile,) + location.shape, 
                dtype=location.dtype,
            )
            self._observations[grid_location] = {
                'observation': observation_buffer,
                'location': location_buffer,
                'idx': -1,
            }
        
        # Add observation to tile buffer
        idx_last = self._observations[grid_location]['idx']
        idx = (idx_last + 1) % self._max_obs_per_tile
        self._observations[grid_location]['observation'][idx] = observation
        self._observations[grid_location]['location'][idx] = location
        self._observations[grid_location]['idx'] = idx

    def reset(self):
        self._observations = collections.OrderedDict()

    def _transform_observations(self, observations, env_info, type_=None):
        if type_ is None:
            type_ = observations[0].dtype

        if type_ == "uint8":
            observations = observations.astype(np.float32) / 255.0
        elif type_ == "int32":
            batch_size = observations.shape[0]
            n_idx = observations.shape[-1]

            if n_idx == 1:
                obs_idx = observations.flatten()
                observations = np.zeros((batch_size, env_info.get('n'),), dtype=np.float32)
                observations[np.arange(batch_size), obs_idx] = 1.0
            elif n_idx == 3:
                n = env_info.get('nx') + env_info.get('ny') + env_info.get('nt')
                obs_x_idx = observations[:,0]
                obs_y_idx = env_info.get('nx') + observations[:,1]
                obs_t_idx = env_info.get('nx') + env_info.get('ny') + observations[:,2]
                observations = np.zeros((batch_size, n), dtype=np.float32)
                observations[np.arange(batch_size), obs_x_idx] = 1.0
                observations[np.arange(batch_size), obs_y_idx] = 1.0
                observations[np.arange(batch_size), obs_t_idx] = 1.0

        return observations

    def get_observations(self, env_info={}, transform=True):
        # Create new dictionary based on the orientation
        observation_dict = {}

        # Fill the dictionary
        for grid_location, observation_buffer_dict in self._observations.items():           
            theta = grid_location[-1]

            # Create the orientation dictionary if it does not exist
            if theta not in observation_dict:
                observation_dict[theta] = {
                    'observation': [],
                    'location': [],
                }

            # Extract the buffers
            idx = observation_buffer_dict['idx']
            observation_buffer = observation_buffer_dict['observation'][:idx+1]
            location_buffer = observation_buffer_dict['location'][:idx+1]

            # Add the buffers to the orientation dictionary
            observation_dict[theta]['observation'].append(observation_buffer)
            observation_dict[theta]['location'].append(location_buffer)
        
        # Concatenate the buffers
        for theta in observation_dict.keys():
            observation_dict[theta]['observation'] = np.concatenate(
                observation_dict[theta]['observation'], axis=0)
            observation_dict[theta]['location'] = np.concatenate(
                observation_dict[theta]['location'], axis=0)
            
        # Remove repeated observations
        for theta in observation_dict.keys():
            unique_idx = np.unique(
                observation_dict[theta]['location'][:,:2], axis=0, return_index=True
            )[1]
            observation_dict[theta]['observation'] = observation_dict[theta]['observation'][unique_idx]
            observation_dict[theta]['location'] = observation_dict[theta]['location'][unique_idx]

        # Transform the observations
        if transform:
            for theta in observation_dict.keys():
                observation_dict[theta]['observation'] = self._transform_observations(
                    observation_dict[theta]['observation'], env_info)        

        return observation_dict
    
    def get_last_observations(self, env_info={}):
        # Fill a list with the last observation of each tile
        observation_list = []
        for observation_buffer_dict in self._observations.values():           
            idx = observation_buffer_dict['idx']
            last_idx = idx % self._max_obs_per_tile
            last_observation = observation_buffer_dict['observation'][last_idx]
            observation_list.append(last_observation)
            
        # Transform to numpy array
        observations = np.stack(observation_list, axis=0)
            
        # Transform the observations
        observations = self._transform_observations(
            observations, env_info)

        return observations
    
    def get_all_observations(self, env_info={}):
        # Fill a list with the last observation of each tile
        observation_list = []
        for observation_buffer_dict in self._observations.values(): 
            idx = observation_buffer_dict['idx']
            last_idx = idx % self._max_obs_per_tile          
            observations = observation_buffer_dict['observation'][:last_idx+1]
            if len(observations) == 0:
                continue
            observation_list.append(observations)

        # Transform to numpy array
        observations = np.concatenate(observation_list, axis=0)
            
        # Transform the observations
        observations = self._transform_observations(
            observations, env_info)

        return observations
    
    def get_vertices(self):
        return list(self._observations.keys())
    
    # def save(self, save_path):
    #     save_path = save_path + f"grid_buffer_{self._seed}_{self._observation_type}.npz"
    #     path_ = Path(save_path)
    #     path_.parent.mkdir(parents=True, exist_ok=True)
    #     np.savez(
    #         save_path,
    #         **{str(key): value for key, value in self._observations.items()}
    #     )

    # def load(self, load_path):
    #     data = np.load(load_path + f"grid_buffer_{self._seed}_{self._observation_type}.npz")
    #     for key in data.keys():
    #         tuple_key = tuple(map(int, key.split('(')[1].split(')')[0].split(',')))
    #         self._observations[tuple_key] = data[key]
    
    #     self._grid_vertices = [k for k, v in self._observations.items() if v is not None]


class GridListBuffer:
    """Only store unique observations for evaluation.
    
    """
    def __init__(
            self, 
            observation_type: str, 
            seed: int, 
            max_obs_per_tile: int = 1000
        ):
        # Set the parameters
        self._observation_type = observation_type
        self._seed = seed
        self._max_obs_per_tile = max_obs_per_tile  # TODO: Enforce its use
        
        # Initialize the buffer
        self._observations: Dict[Dict[np.ndarray]] = collections.OrderedDict()
        
    def add_observation(self, localized_observation: Dict):
        # Extract components
        observation = localized_observation['observation']
        location = localized_observation['location']
        grid_location = localized_observation['grid_location']
        if isinstance(grid_location, np.ndarray):
            grid_location = tuple(coordinate for coordinate in grid_location)

        # The tile buffer is created with appropriate size if it does not exist
        if grid_location not in self._observations:
            observation_buffer = []
            location_buffer = []
            self._observations[grid_location] = {
                'observation': observation_buffer,
                'location': location_buffer,
            }
        
        # Add observation to tile buffer
        self._observations[grid_location]['observation'].append(observation)
        self._observations[grid_location]['location'].append(location)

    def reset(self):
        self._observations = collections.OrderedDict()

    def _transform_observations(self, observations, env_info, type_=None):
        if type_ is None:
            type_ = observations[0].dtype

        if type_ == "uint8":
            observations = observations.astype(np.float32) / 255.0
        elif type_ == "int32":
            batch_size = observations.shape[0]
            n_idx = observations.shape[-1]

            if n_idx == 1:
                obs_idx = observations.flatten()
                observations = np.zeros((batch_size, env_info.get('n'),), dtype=np.float32)
                observations[np.arange(batch_size), obs_idx] = 1.0
            elif n_idx == 3:
                n = env_info.get('nx') + env_info.get('ny') + env_info.get('nt')
                obs_x_idx = observations[:,0]
                obs_y_idx = env_info.get('nx') + observations[:,1]
                obs_t_idx = env_info.get('nx') + env_info.get('ny') + observations[:,2]
                observations = np.zeros((batch_size, n), dtype=np.float32)
                observations[np.arange(batch_size), obs_x_idx] = 1.0
                observations[np.arange(batch_size), obs_y_idx] = 1.0
                observations[np.arange(batch_size), obs_t_idx] = 1.0

        return observations

    def get_observations(self, env_info={}, transform=True):
        # Create new dictionary based on the orientation
        observation_dict = {}

        # Fill the dictionary
        for grid_location, observation_buffer_dict in self._observations.items():           
            theta = grid_location[-1]

            # Create the orientation dictionary if it does not exist
            if theta not in observation_dict:
                observation_dict[theta] = {
                    'observation': [],
                    'location': [],
                }

            # Extract the buffers
            observation_buffer = observation_buffer_dict['observation']
            location_buffer = observation_buffer_dict['location']

            # Add the buffers to the orientation dictionary
            observation_dict[theta]['observation'].extend(observation_buffer)
            observation_dict[theta]['location'].extend(location_buffer)

        # Concatenate the buffers
        for theta in observation_dict.keys():
            observation_dict[theta]['observation'] = np.stack(
                observation_dict[theta]['observation'], axis=0)
            observation_dict[theta]['location'] = np.stack(
                observation_dict[theta]['location'], axis=0)
            
        # Remove repeated observations
        for theta in observation_dict.keys():
            unique_idx = np.unique(
                observation_dict[theta]['location'][:,:2], axis=0, return_index=True
            )[1]
            observation_dict[theta]['observation'] = observation_dict[theta]['observation'][unique_idx]
            observation_dict[theta]['location'] = observation_dict[theta]['location'][unique_idx]

        # Transform the observations
        if transform:
            for theta in observation_dict.keys():
                observation_dict[theta]['observation'] = self._transform_observations(
                    observation_dict[theta]['observation'], env_info)        

        return observation_dict
    
    def get_last_observations(self, env_info={}):
        # Fill a list with the last observation of each tile
        observation_list = []
        for observation_buffer_dict in self._observations.values():           
            last_observation = observation_buffer_dict['observation'][-1]
            observation_list.append(last_observation)
            
        # Transform to numpy array
        observations = np.stack(observation_list, axis=0)
            
        # Transform the observations
        observations = self._transform_observations(
            observations, env_info)

        return observations
    
    def get_all_observations(self, env_info={}):
        # Fill a list with the last observation of each tile
        observation_list = []
        location_list = []
        for grid_location, observation_buffer_dict in self._observations.items(): 
            observations = observation_buffer_dict['observation']
            if len(observations) == 0:
                continue
            observation_list.extend(observations)
            location_list.extend(len(observations)*[np.array(grid_location)])

        # Transform to numpy array
        observations = np.stack(observation_list, axis=0)
        grid_locations = np.stack(location_list, axis=0)
            
        # Transform the observations
        observations = self._transform_observations(
            observations, env_info)

        return observations, grid_locations
    
    def get_vertices(self):
        return list(self._observations.keys())