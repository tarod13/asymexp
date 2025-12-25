import os
import random
import time

from typing import Dict
from tqdm import tqdm
import tyro
import pickle

import numpy as np
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import gymnasium as gym
import miniworld   # Necessary to import MiniWorld environments
import src.miniworld.envs   # Necessary to import custom MiniWorld environments

from src.common import EpisodicReplayBuffer, GridBuffer, order_eigenvectors, compute_cosine_similarity
from src.allo.policies import Policy, RandomPolicy, RandomPolicyWithWallAvoidance
from src.allo.networks import symlog, Encoder, ModularEncoder
from src.allo.rl_loop import make_env, populate_buffers
from src.allo.utils import binary_to_boolean, load_eigensystem, save_model_
from src.common.saving import load_model
from src.allo.visualization import plot_representations_from_dict_v3, plot_eigenvectors, plot_observations
from src.allo.params import Args

import wandb

import certifi
# Get the path to the CA certificate bundle
ca_cert_path = certifi.where()
# Set the WANDB_CA_CERTS environment variable
os.environ['WANDB_CA_CERTS'] = ca_cert_path

os.environ['WANDB_API_KEY']='83c25550226f8a86fdd4874026d2c0804cd3dc05'
os.environ['WANDB_ENTITY']='tarod13'

BOOL_ATTRS = [
    'sg_end_rep',
    'prop_grad_single_batch',
    'use_max_graph',
    'use_splatter',
    'use_colored_walls',
    'use_triangular_cues',
    'use_backwards',
    'use_combined',
    'use_levy',
    'light1_enabled',
    'light2_enabled',
    'light3_enabled',
    'light4_enabled',
    'light5_enabled',
    'resize_obs',
    'grayscale_obs',
    'init_dual_diag',
    'add_observation_noise',
    'make_first_one',
    'use_signed_output',
    'turn_off_above_threshold',
]


def learn_complex_laplacian(args, replay_buffer=None, grid_buffer=None):  
    # Process args
    for attr in BOOL_ATTRS:
        if hasattr(args, attr):
            setattr(args, attr, binary_to_boolean(getattr(args, attr)))

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]

    # Set up run    
    run_name = f"{args.env_name}__{args.exp_name}__{args.observation_type}__{args.exp_number}__{args.seed}__{int(time.time())}"
    if args.track:
        if args.wandb_mode == 'offline':
            os.environ['WANDB_MODE'] = 'offline'
        run = wandb.init(
            project=args.wandb_project_name,
            config=vars(args),
            save_code=True,
            dir=args.wandb_dir,
            mode=args.wandb_mode,
        )
        
    else:
        run = None

    # Create env folder to store results
    if args.track or args.save_model or args.do_evaluation:
        path = f'./results/data/{args.env_name}/some_file.pkl'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, encoder_key = jax.random.split(rng_key, 2)

    # Env setup
    room_kwargs = {
        "wall_tex": args.wall_tex,
        "floor_tex": args.floor_tex,
        "ceil_tex": args.ceil_tex,
        "use_splatter": args.use_splatter,
        "use_colored_walls": args.use_colored_walls,
        "use_triangular_cues": args.use_triangular_cues,
        "light_pos": [args.light_pos_x, args.light_pos_y, args.light_pos_z],
        "second_light_pos": [args.second_light_pos_x, args.second_light_pos_y, args.second_light_pos_z],
        "light3_pos": [args.light3_pos_x, args.light3_pos_y, args.light3_pos_z],    
        "light4_pos": [args.light4_pos_x, args.light4_pos_y, args.light4_pos_z],
        "light5_pos": [args.light5_pos_x, args.light5_pos_y, args.light5_pos_z],
        "light1_enabled": args.light1_enabled,
        "light2_enabled": args.light2_enabled,
        "light3_enabled": args.light3_enabled,
        "light4_enabled": args.light4_enabled,
        "light5_enabled": args.light5_enabled,
        "light1_constant_attenuation": args.light1_constant_attenuation,
        "light1_linear_attenuation": args.light1_linear_attenuation,
        "light1_quadratic_attenuation": args.light1_quadratic_attenuation,
        "wall_height": args.wall_height,
        "cam_pitch": args.cam_pitch,
    }

    env = make_env(
        args.env_name, 
        max_episode_steps=args.max_episode_length, 
        view_mode=args.observation_type,
        size=args.env_size,
        coarseness=args.coarseness,
        use_backwards=args.use_backwards,
        use_combined=args.use_combined,
        use_levy=args.use_levy,
        resize_obs=args.resize_obs,
        grayscale_obs=args.grayscale_obs,
        **room_kwargs,
    )
    env_info = {}
    if args.observation_type in ['hot', 'hot3']:
        env_info = {
            'n': env.n,
            'nx': env.n_tiles_x,
            'ny': env.n_tiles_y,
            'nt': env.n_tiles_theta,
        }
    miniworld_version = miniworld.__version__
    print(f"Created environment. Miniworld v{miniworld_version}, Env: {args.env_name}")   # TODO: use logger instead

    step = env.reset(seed=args.seed, options={})   # WARNING: Seed is only required the first time reset is called
    observation, info = step
    
    # Preprocess observation
    if args.observation_type in ['top', '3d']:
        observation = observation['observation'].astype(np.float32) / 255.0
    elif args.observation_type in ['hot']:
        observation = np.zeros((env_info.get('n'),), dtype=np.float32)
    elif args.observation_type in ['hot3']:
        n = env_info.get('nx') + env_info.get('ny') + env_info.get('nt')
        observation = np.zeros((n,), dtype=np.float32)
    elif args.observation_type in ['xyt']:
        observation = observation['observation'].astype(np.float32)
    observation = observation.reshape((1, *observation.shape))

    # Policy setup
    policy = globals()[args.policy_class](env)

    # Create buffers if necessary
    # 1. Replay buffer
    if replay_buffer is None:
        replay_buffer = EpisodicReplayBuffer(
            args.num_episodes,
            args.max_episode_length,
            args.observation_type,
            args.seed,
        )  
    # 2. Grid buffer
    if grid_buffer is None:
        grid_buffer = GridBuffer(args.observation_type, args.seed, args.max_obs_per_tile)

    # Populate buffers
    replay_buffer, grid_buffer = populate_buffers(
        args.save_experience,
        args.load_experience,
        args.env_name,
        env,
        replay_buffer,
        grid_buffer,
        policy,
        max(1, int(args.num_episodes * args.initial_fraction)),
        args.max_episode_length,
        args.num_save_chunk_episodes,
        args.num_save_chunk_steps,
        env_info=env_info,
        add_noisy_copies=args.add_noisy_copies,
        num_noisy_copies=args.num_noisy_copies,
    )
    
    # Initialize the Laplacian encoder
    if not args.observation_type in ['top', '3d']:
        args.num_hidden_layers = args.num_hidden_layers + 3
    encoder_kwargs = {
        'hidden_dim': args.hidden_dim,
        'mlp_input_nl': args.mlp_input_nl,
        'observation_type': args.observation_type,
        'output_nl': args.output_nl,
        'kernel_size': args.kernel_size,
        'kernel_size_2': args.kernel_size_2,
        'kernel_size_3': args.kernel_size_3,
        'num_hidden_layers': args.num_hidden_layers,
        'make_first_one': args.make_first_one,
        'use_signed_output': args.use_signed_output,
    }
    
    encoder = Encoder(
        num_features=args.num_eigenvectors,
        **encoder_kwargs,
    )
    
    assert args.num_eigenvectors > 1, "Number of eigenvectors must be greater than 1."
    if args.init_dual_diag:
        initial_dual_mask = jnp.eye(args.num_eigenvectors)
    else:
        initial_dual_mask = jnp.tril(jnp.ones((args.num_eigenvectors,args.num_eigenvectors)))  
    
    initial_params = {
        'encoder': encoder.init(encoder_key, observation),
        'duals': args.duals_initial_val * initial_dual_mask,
        'barrier_coefs': args.barrier_initial_val * jnp.ones((1, 1)),
        'error_integral': jnp.zeros((args.num_eigenvectors, args.num_eigenvectors)),
    }

    encoder_tx = optax.adam(learning_rate=args.learning_rate)
        
    # Create masks for different parameter groups
    encoder_mask = {
        'encoder': True, 
        'duals': False, 
        'barrier_coefs': False, 
        'error_integral': False,
    }
    other_mask = {
        'encoder': False, 
        'duals': True, 
        'barrier_coefs': True, 
        'error_integral': False,
    }
    
    # Create the combined optimizer using the masks
    sgd_tx = optax.sgd(learning_rate=args.learning_rate)
    
    tx = optax.chain(
        optax.masked(encoder_tx, encoder_mask),
        optax.masked(sgd_tx, other_mask)
    )

    encoder_state = TrainState.create(
        apply_fn=encoder.apply,
        params=initial_params,
        tx=tx,
    )
    encoder.apply = jax.jit(encoder.apply)

    # Load model if loading is turned on
    if args.load_model:
        if args.load_model_path is None:
            load_path = f'./results/models/{args.load_env_name}/{args.load_model_id}.pkl'
        else:
            load_path = args.load_model_path
        params = load_model(load_path)[0]
        if 'error_integral' not in params:
            params['error_integral'] = jnp.zeros((args.num_eigenvectors, args.num_eigenvectors))
        encoder_state = encoder_state.replace(params=params)
        print(f"Successfully loaded model from {load_path}")

    # Define the update functions
    @jax.jit
    def update_encoder(
        encoder_state: TrainState,
        observations: np.ndarray,
        next_observations: np.ndarray,
        observations_2: np.ndarray,
    ):
        def check_previous_entries_below_threshold(matrix, threshold):
            # Create a matrix that contains 1 where the absolute value is below threshold
            below_threshold = (jnp.abs(matrix) < threshold).astype(jnp.float32)
            
            # Compute a scan that checks if all values up to each position are below threshold
            row_all_below = jnp.prod(below_threshold, axis=1, keepdims=True)
            
            # Finally, for each row i, check if all previous rows satisfy the condition
            cumulative_results = jnp.cumprod(row_all_below, axis=0)

            result_with_zero = jnp.ones((matrix.shape[0] + 1, 1), dtype=cumulative_results.dtype)
    
            # Then update all but the first position with the cumulative results
            result_with_zero = result_with_zero.at[1:, :].set(cumulative_results)
            
            # Finally, remove the last element
            final_results = result_with_zero[:-1, :]
            
            return final_results

        
        def encoder_loss(params):
            # Compute representations
            encoder_params = params['encoder']
            phi = encoder.apply(encoder_params, observations)[0]
            phi_2 = encoder.apply(encoder_params, observations_2)[0]
            next_phi = encoder.apply(encoder_params, next_observations)[0]

            # Get sizes
            d = args.num_eigenvectors
            n = phi.shape[0]

            # Get duals
            dual_variables = params['duals']
            barrier_coefficients = params['barrier_coefs']
            diagonal_duals = jnp.diag(dual_variables)
            eigenvalue_sum = - 0.5 * diagonal_duals.sum()

            inner_product_matrix_1 = jnp.einsum(
                'ij,ik->jk', phi, jax.lax.stop_gradient(phi)) / n
            inner_product_matrix_2 = jnp.einsum(
                'ij,ik->jk', phi_2, jax.lax.stop_gradient(phi_2)) / n
            
            error_matrix_1 = jnp.tril(inner_product_matrix_1 - jnp.eye(d))
            error_matrix_2 = jnp.tril(inner_product_matrix_2 - jnp.eye(d))   

            # Compute error matrices below threshold
            cum_error_matrix_1_below_threshold = jax.lax.cond(
                args.turn_off_above_threshold,
                lambda x: check_previous_entries_below_threshold(x, args.cum_error_threshold),
                lambda x: jnp.ones([x.shape[0], 1]),
                error_matrix_1,
            )
            cum_error_matrix_1_below_threshold = jax.lax.stop_gradient(cum_error_matrix_1_below_threshold)         

            # Compute dual loss
            error_integral = params['error_integral']
            dual_loss_pos = (
                jax.lax.stop_gradient(dual_variables) 
                * cum_error_matrix_1_below_threshold * error_matrix_1
            ).sum()

            dual_loss_P = jax.lax.stop_gradient(args.step_size_duals * error_matrix_1)
            dual_loss_I = args.step_size_duals_I * jax.lax.stop_gradient(error_integral)
            dual_loss_neg = - (dual_variables * cum_error_matrix_1_below_threshold * (dual_loss_P + dual_loss_I)).sum()            

            # Compute barrier loss
            quadratic_error_matrix = 2 * cum_error_matrix_1_below_threshold * error_matrix_1 * jax.lax.stop_gradient(error_matrix_2)
            quadratic_error = quadratic_error_matrix.sum()
            barrier_loss_pos = jax.lax.stop_gradient(barrier_coefficients[0,0]) * quadratic_error
            barrier_loss_neg = -barrier_coefficients[0,0] * jax.lax.stop_gradient(jnp.absolute(quadratic_error))

            # Compute reprensetation variances
            phi_centered = (phi - jnp.mean(phi, axis=0, keepdims=True))  # remove first column
            phi_variances = (phi_centered**2).mean(0, keepdims=True)

            delta = jnp.exp(-1 / args.graph_variance_scale)
            if args.perturbation_type == 'squared-null-grad':
                graph_perturbation = args.graph_epsilon * (
                    ((phi_centered-1)**2).mean(0, keepdims=True)
                ).clip(0,1)
            elif args.perturbation_type == 'squared':
                graph_perturbation = args.graph_epsilon * (
                    ((jnp.absolute(phi_centered-1)+delta)**2-delta**2).mean(0, keepdims=True)
                ).clip(0,1)
            elif args.perturbation_type == 'exponential':
                graph_perturbation = args.graph_epsilon * (
                    jnp.exp(-phi_variances / args.graph_variance_scale) 
                    - delta
                ).clip(0, 1)
            else:
                graph_perturbation = jnp.zeros_like(phi_variances)
            
            graph_perturbation = graph_perturbation.at[0,0].set(0.0)


            # Compute graph drawing losses
            diff = (phi - next_phi) * cum_error_matrix_1_below_threshold.reshape(1,-1)            
            graph_losses = 0.5*((diff)**2).mean(0, keepdims=True)
            graph_loss = (graph_losses + graph_perturbation).sum()

            # Compute auxiliary metrics
            norm_phi = (phi**2).mean(0, keepdims=True)
            norm_errors_1 = jnp.diag(error_matrix_1)
            distance_to_constraint_manifold = jnp.tril(error_matrix_1**2).sum()
            total_norm_error = jnp.absolute(norm_errors_1).sum()
            total_error = jnp.absolute(error_matrix_1).sum()
            total_two_component_error = jnp.absolute(error_matrix_1[:,:min(2,d)]).sum()
            
            # Total loss
            positive_loss = graph_loss + dual_loss_pos + barrier_loss_pos
            negative_loss = dual_loss_neg + barrier_loss_neg

            allo = positive_loss + negative_loss    

            # Auxiliary metrics
            aux = {
                'graph_loss': graph_loss,
                'dual_loss': dual_loss_pos,
                'dual_loss_neg': dual_loss_neg,
                'barrier_loss': barrier_loss_pos,
                'approx_eigenvalue_sum': eigenvalue_sum,
                'barrier_coef': barrier_coefficients[0,0],
                'total_norm_error': total_norm_error,
                'total_error': total_error,
                'total_two_component_error': total_two_component_error,
                'distance_to_constraint_manifold': distance_to_constraint_manifold,
                'distance_to_origin': norm_phi.sum(),
            }
            # Add dual variables to aux
            for i in range(min(11,args.num_eigenvectors)):
                aux[f'dual_{i}'] = dual_variables[i,i]
                aux[f'graph_perturbation_{i}'] = graph_perturbation[0,i]

                for j in range(0, min(2,i)):
                    aux[f'dual_{i}_{j}'] = dual_variables[i,j]

            return allo, (error_matrix_1, aux)

        # Compute loss and gradients
        (allo, (error_matrix, aux)), grads = jax.value_and_grad(
            encoder_loss, has_aux=True)(encoder_state.params)
        
        # encoder_state = encoder_state.apply_gradients(grads=grads)
        
        # Apply optimizer updates
        updates, new_opt_state = encoder_state.tx.update(
            grads, encoder_state.opt_state, encoder_state.params)
        new_params = optax.apply_updates(encoder_state.params, updates)

        # Perform custom integral update with the error matrix
        new_params['error_integral'] = args.integral_decay * new_params['error_integral'] + error_matrix

        # Clip the barrier coefficients
        new_params['barrier_coefs'] = jnp.clip(new_params['barrier_coefs'], 0, args.max_barrier_coefs)
        
        # Create new state
        new_encoder_state = encoder_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=encoder_state.step + 1
        )

        # Get grad norm
        # Flatten the gradients into a single vector
        grads_flat, _ = jax.tree_util.tree_flatten(grads)
        grads_vector = jnp.concatenate([jnp.ravel(g) for g in grads_flat])
        grad_norm = jnp.linalg.norm(grads_vector)
        aux['grad_norm'] = grad_norm
        
        return new_encoder_state, allo, aux

    
    # Load eigensystem if evaluation is turned on
    if args.do_evaluation:
        load_info = {
            'env_name': args.load_env_name,
            'policy_class': args.baseline_policy_class.lower(),
            'env_size': int(args.env_size),
            'num_episodes': args.baseline_num_episodes,
            'num_steps': args.baseline_num_steps,
            'load_label': args.load_label,
        }

        # Get grid observations
        grid_obs = grid_buffer.get_last_observations(env_info)
        grid_vertices = grid_buffer.get_vertices()
        
        # try:
        separate_eigensystem = load_eigensystem(load_info)
        complete_eigensystem = {}
        complete_eigenvalues = np.round(separate_eigensystem.pop('obs_eigenvalues'), 6)
        eigenvectors = separate_eigensystem.pop('obs_eigenvectors')
        index_mapping = separate_eigensystem.get('obs_index_mapping')
        del separate_eigensystem
        for i in range(len(complete_eigenvalues)):
            if complete_eigensystem.get(complete_eigenvalues[i]) is None:
                complete_eigensystem[complete_eigenvalues[i]] = []
            complete_eigensystem[complete_eigenvalues[i]].append(eigenvectors[:,i])
        
        # Get the first num_eigenvectors eigenvectors
        eigenvalues = complete_eigenvalues[:args.num_eigenvectors]
        eigenvalue_sum = sum(eigenvalues)
        eigensystem = {}
        accounted_eigenvectors = 0
        for key in sorted(complete_eigensystem.keys(), reverse=True):
            current_eigenvectors = complete_eigensystem[key]
            accounted_eigenvectors += len(current_eigenvectors)
            assert accounted_eigenvectors <= args.num_eigenvectors, \
                "Number of eigenvectors exceeds the limit. Please increase the number of eigenvectors."
            eigensystem[key] = complete_eigensystem[key]
            if accounted_eigenvectors == args.num_eigenvectors:
                break

        # Order eigenvectors
        eigensystem = order_eigenvectors(
            eigensystem, None, grid_vertices, index_mapping)[0]

        # Stack eigenvectors in single matrix
        gt_eigenvectors = []
        for eigval in eigensystem.keys():
            gt_eigenvectors.extend(eigensystem[eigval])
        gt_eigenvectors = np.stack(gt_eigenvectors, axis=1)
        if eigenvectors.shape[0] > gt_eigenvectors.shape[0]:
            print('Warning: Ground truth eigenvectors have more entries than grids in the Grid Buffer.')

        plot_eigenvectors(
            env, gt_eigenvectors, grid_vertices, None, 
            args.env_name, args.num_episodes, 4, 
            args.policy_class, args.env_size,
            logger=run, label_='_gt', save_fig=args.save_plots,
            use_global_lims=False, use_colorbar=True,
        )

        if args.track:
            eigenvalue_dict = {f'eigenvalue_{i}': eigenvalues[i] for i in range(len(eigenvalues))}
            eigenvalue_dict['eigenvalue_sum'] = eigenvalue_sum
            run.log(eigenvalue_dict)
        # except:
        #     print('Evaluation turned off. Eigenloading step failed...')
        #     args.do_evaluation = False
    
    def evaluate(
            encoder: nn.Module, 
            encoder_state: TrainState, 
        ):
        # Compute cosine similarity
        cs_result, cs_result_rotated = compute_cosine_similarity(
            encoder, encoder_state, eigensystem, grid_obs, batch_size=args.batch_size,
        )
        avg_cosine_similarity, similarities, approx_eigvec = cs_result
        avg_cosine_similarity_rotated, similarities_rotated, rotated_approx_eigvec = cs_result_rotated

        # Return cosine similarity and similarities as a dictionary
        similarity_metrics = {
            'avg_cosine_similarity_rotated': avg_cosine_similarity_rotated,
            'avg_cosine_similarity': avg_cosine_similarity,
        }
        for i, (similarity, similarity_rotated) in enumerate(zip(similarities, similarities_rotated)):
            similarity_metrics[f'cosine_similarity_rotated_{i}'] = similarity_rotated.item()
            similarity_metrics[f'cosine_similarity_{i}'] = similarity.item()

        # Compute eigenvalue error
        dual_variables = encoder_state.params['duals']
        diagonal_duals = jnp.diag(dual_variables)
        if not args.sg_end_rep:
            diagonal_duals = 0.5 * diagonal_duals
        approx_eigenvalues = (args.gamma + diagonal_duals) / (args.gamma * (1+diagonal_duals))
        eigenvalue_error = approx_eigenvalues - eigenvalues
        for i, error in enumerate(eigenvalue_error):
            similarity_metrics[f'eigenvalue_error_{i}'] = error.item()

        # Plot eigenvectors
        if args.log_plots:
            plot_eigenvectors(
                env, approx_eigvec, grid_vertices, None, 
                args.env_name, args.num_episodes, 4, 
                args.policy_class, args.env_size,
                logger=run, save_fig=args.save_plots,
                use_global_lims=False, use_colorbar=True,
            )

        approx_dict = {
            'approx_eigenvectors': approx_eigvec,
            'approx_eigenvalues': approx_eigenvalues,
            'obs_index_mapping': grid_vertices,
            'eigenvalues': eigenvalues,
            'eigenvectors': gt_eigenvectors,
        }

        return similarity_metrics, approx_dict

    def visualize(
            encoder: nn.Module, 
            encoder_state: TrainState, 
            env: gym.Env,
            env_info: Dict = {},
        ):
        # Obtain observations
        eigenvector_dict = grid_buffer.get_observations(env_info, transform=False)

        # Choose random observations
        random_angles = np.random.choice(list(eigenvector_dict.keys()), 4, replace=True)
        obs_lengths = [len(eigenvector_dict[theta]['observation']) for theta in random_angles]
        random_indices = [np.random.randint(obs_lengths[i]) for i in range(4)]
        random_observations = [eigenvector_dict[random_angles[i]]['observation'][random_indices[i]] for i in range(4)]

        # Plot observations
        plot_observations(
            random_observations, args.observation_type, args.env_name, 
            args.num_episodes, args.policy_class, args.env_size, 
            logger=run, save_fig=args.save_plots,
        )

        # Compute representations
        for theta in eigenvector_dict.keys():
            observations = eigenvector_dict[theta].pop('observation')
            eigenvector_list = []
            n_batches = int(np.ceil(observations.shape[0] / args.batch_size))
            for i in range(n_batches):
                # Get batch
                batch = observations[i*args.batch_size:(i+1)*args.batch_size]

                # Transform batch
                batch = grid_buffer._transform_observations(batch, env_info)

                # Compute batch representations
                eigenvectors = encoder.apply(encoder_state.params['encoder'], batch)[0]
                eigenvector_list.append(eigenvectors)
            eigenvectors = jnp.concatenate(eigenvector_list, axis=0)            
            eigenvector_dict[theta]['eigenvector'] = eigenvectors

        # Plot eigenvectors        
        plot_representations_from_dict_v3(
            eigenvector_dict, env, min(args.num_eigenvectors, 40), args.num_plot_tiles,
            args.env_name, args.num_episodes, 4, args.policy_class,
            args.env_size, logger=run, save_fig=args.save_plots,
            use_global_lims=False, use_colorbar=True,
        )

    
    # Start the training process
    start_time = time.time()
    max_cosine_similarity = -1

    remaining_episodes = args.num_episodes - int(args.num_episodes * args.initial_fraction)
    data_generation_freq = max(1, args.num_gradient_steps // remaining_episodes)
     
    for gradient_step in tqdm(range(args.num_gradient_steps)):

        ########################################################
        # Training block
        data = replay_buffer.sample(args.batch_size, args.gamma, env_info)
        data_2 = replay_buffer.sample(args.batch_size, args.gamma, env_info)        

        encoder_state, allo, metrics = update_encoder(
            encoder_state,
            data.obs,
            data.next_obs,
            data_2.obs,
        )

        is_log_step = (
            ((gradient_step % args.log_freq)  == 0)
            or (gradient_step == args.num_gradient_steps - 1)
        )
        if is_log_step:
            if args.track:
                log_dict = {
                    "SPS": int(gradient_step / (time.time() - start_time)),
                    "gradient_step": gradient_step,
                    "allo": allo.item(),
                }
                for k, v in metrics.items():
                    log_dict[f"{k}"] = v.item()
                run.log(log_dict)

        ########################################################
        # Evaluation block
        is_last_step = gradient_step == (args.num_gradient_steps - 1)
        is_eval_step = (gradient_step % args.eval_freq)  == 0
        do_evaluation = (is_eval_step or is_last_step) and args.do_evaluation

        if (is_eval_step or is_last_step) and not args.do_evaluation:
            print_message = f"gradient_step={gradient_step}"
            print_message += f", allo={allo.item()}"
            print(print_message)
        
        if do_evaluation:
            similarity_metrics, approx_dict = evaluate(encoder, encoder_state)
            cosine_similarity = similarity_metrics['avg_cosine_similarity']
            print(f"gradient_step={gradient_step}, cosine_similarity={cosine_similarity}")
            if args.track:
                run.log(similarity_metrics)

                # Save the model if the cosine similarity is better than the previous best
                if cosine_similarity > max_cosine_similarity:
                    max_cosine_similarity = cosine_similarity
                    saving_label = '_best'
                else:
                    saving_label = '_last'
                if args.save_model:
                    save_model_(
                        args.env_name, 
                        run_name+saving_label, 
                        encoder_state.params, 
                        encoder_state.opt_state, 
                        logger=run, 
                        args=args,
                    )

                # Save the eigensystem
                path = f'./results/data/{args.env_name}/eigensystem_{run_name}{saving_label}.pkl'
                with open(path, 'wb') as file:
                    pickle.dump(approx_dict, file)

                # Log the eigensystem as an artifact
                if args.save_artifact:
                    run.log_artifact(path, name=f'eigensystem{saving_label}.pkl')

        ########################################################
        # Visualization block
        is_visual_step = (gradient_step % args.visual_freq)  == 0
        do_visualization = (is_visual_step or is_last_step) and args.do_visualization
        if do_visualization:
            visualize(encoder, encoder_state, env, env_info)

        ########################################################
        # Saving block
        is_save_step = (gradient_step % args.save_freq)  == 0
        do_saving = (is_save_step or is_last_step) and args.save_model
        if do_saving:
            model_version = f'{run_name}_step_{gradient_step}'
            save_model_(
                args.env_name, 
                model_version, 
                encoder_state.params, 
                encoder_state.opt_state, 
                logger=run,
                args=args,
            )

        ########################################################
        # Data generation block
        is_data_gen_step = (gradient_step % data_generation_freq)  == 0
        do_data_gen = is_data_gen_step and remaining_episodes > 0
        if do_data_gen:
            replay_buffer, grid_buffer = populate_buffers(
                args.save_experience,
                args.load_experience,
                args.env_name,
                env,
                replay_buffer,
                grid_buffer,
                policy,
                1,
                args.max_episode_length,
                args.num_save_chunk_episodes,
                args.num_save_chunk_steps,
                env_info=env_info,
                add_noisy_copies=args.add_noisy_copies,
                num_noisy_copies=args.num_noisy_copies,
            )
            remaining_episodes -= 1

    env.close()

    return encoder_state, run, replay_buffer, grid_buffer

if __name__ == "__main__":
    args = tyro.cli(Args)
    learn_complex_laplacian(args)