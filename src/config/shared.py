from dataclasses import dataclass


@dataclass
class SharedArgs:
    # Environment
    env_file_name: str = "GridRoom-4-Doors"  # Name of environment file in src/envs/txt/ (e.g., 'GridRoom-4-Doors')
    max_episode_length: int = 1000
    windy: bool = False  # If True, use WindyGridWorldEnv instead of GridWorldEnv
    wind: float = 0.0   # Wind strength in (-1, 1); negative = leftward, positive = rightward
    random_wind: bool = False  # If True and windy=True, resample wind ~ Uniform(-0.9, 0.9) each episode
    num_eval_winds: int = 11  # (random_wind only) number of fixed wind values for per-wind GT evaluation, equally spaced in [-0.99, 0.99]

    # Data collection
    num_envs: int = 1000
    num_steps: int = 1000

    # Model
    num_eigenvector_pairs: int = 10  # Number of complex eigenvector pairs to learn
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    use_residual: bool = True  # Whether to use residual connections in the network
    use_layernorm: bool = True  # Whether to use LayerNorm in the network
    num_head_hidden_layers: int = 1  # Number of hidden layers in each head after the shared backbone

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 256
    num_gradient_steps: int = 100000
    gamma: float = 0.9  # Discount factor for successor representation
    delta: float = 0.1  # Eigenvalue shift parameter: L = (1+δ)I - M (improves numerical stability)
    ema_learning_rate: float = 0.0003  # EMA update rate for eigenvalue estimates

    # Gradient clipping
    use_global_grad_clip: bool = True  # If True, use global norm clipping (original). If False, clip encoder and lambda separately

    # Loss normalization
    norm_graph_loss: bool = False  # If True, divide each per-eigenvector graph_loss component by its 2-norm (<x_i,x_i> or <y_i,y_i>)

    # Sampling / importance-sampling mode
    # "rejection" : rejection sampling to flatten state distribution (no IS weights needed)
    # "weighted"  : sample from buffer as-is, apply IS ratio weights in the loss
    # "none"      : sample from buffer as-is, no IS correction (correct when distribution is uniform)
    sampling_mode: str = "none"
    rejection_oversample_factor: int = 3  # How many extra samples to draw per rejection round (only used for "rejection")

    # Constraint error approximation method (also controls sampling strategy in shared_training)
    constraint_mode: str = "single_batch"  # Options: "ema", "two_batch", "single_batch", "same_episodes"
    # - "ema": EMA approximation (standard sampling)
    # - "two_batch": Unbiased with two independent batches (standard sampling)
    # - "single_batch": Biased with single batch (standard sampling)
    # - "same_episodes": Intermediate bias - two batches from same episodes (special sampling)

    # Constraint enforcement mechanism (selects how constraint violations drive the loss)
    constraint_enforcement_method: str = "clf"  # Options: "clf", "barrier"
    # - "clf": CLF controller — computes a QP-style correction u = barrier·∇V and adds ip(feat, sg(u))
    # - "barrier": Increasing barrier — directly penalises V with a growing coefficient: sg(barrier_coef)·V
    barrier_initial_val: float = 0.5    # Initial value of barrier_coef (barrier method only)
    min_barrier_coefs: float = 0.0      # Minimum allowed barrier_coef (barrier method only)
    max_barrier_coefs: float = 10.0     # Maximum allowed barrier_coef (barrier method only)
    lr_barrier_coefs: float = 0.01      # Step size for barrier_coef external update (barrier method only)

    # Graph loss variants
    two_sided_graph_loss: bool = False

    # Episodic replay buffer
    max_time_offset: int | None = None  # Maximum time offset for sampling (None = episode length)

    # Episode-centric sampling (primarily for random_wind, ensures per-MDP consistency in batches)
    sample_episodes: bool = False  # If True, sample num_sampled_episodes distinct episodes per batch
    num_sampled_episodes: int = 16  # Number of distinct episodes per batch; batch_size must be divisible by this
    num_wind_buckets: int = 20  # Buckets for wind-conditioned EMA; only active when random_wind=True and sample_episodes=True

    # Logging and saving
    log_freq: int = 500
    plot_freq: int = 1000
    save_freq: int = 1000
    checkpoint_freq: int = 5000  # How often to save checkpoints (in gradient steps)
    save_model: bool = True
    plot_during_training: bool = False  # If True, creates plots during training (slow). If False, only exports data.
    results_dir: str = "./results"

    # Resuming training
    resume_from: str | None = None  # Path to results directory to resume from (e.g., './results/room4/room4__allo__0__42__1234567890')

    # Diagnostics
    eval_rank: bool = False  # If True, compute effective rank of each Dense kernel at log intervals

    # Visualization
    hitting_times_ncols: int = 6       # Columns per row-pair in the hitting-times figure
    hitting_times_nrow_pairs: int = 3  # Row-pairs shown (each pair = target row + source row); default 3 → 6 physical rows
    hitting_times_log_scale: bool = False  # Plot log(steps + 1) instead of raw steps

    # Misc
    seed: int = 42
    exp_name: str | None = None
    exp_number: int = 0
