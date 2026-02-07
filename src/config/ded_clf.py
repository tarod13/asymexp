from dataclasses import dataclass


@dataclass
class Args:
    # Environment
    env_type: str = "file"  # 'room4', 'maze', 'spiral', 'obstacles', 'empty', or 'file'
    env_file: str | None = None  # Path to environment text file (if env_type='file')
    env_file_name: str | None = "GridRoom-4-Doors"  # Name of environment file in src/envs/txt/ (e.g., 'GridRoom-4')
    max_episode_length: int = 1000

    # Data collection
    num_envs: int = 1000
    num_steps: int = 1000

    # Irreversible doors
    use_doors: bool = False
    num_doors: int = 5  # Number of irreversible doors to create
    door_seed: int = 42  # Seed for door placement

    # Model
    num_eigenvector_pairs: int = 10  # Number of complex eigenvector pairs to learn
    hidden_dim: int = 256
    num_hidden_layers: int = 3

    # Training
    learning_rate: float = 1e-5
    batch_size: int = 256
    num_gradient_steps: int = 100000
    gamma: float = 0.95  # Discount factor for successor representation
    delta: float = 0.1  # Eigenvalue shift parameter: L = (1+δ)I - M (improves numerical stability)
    lambda_x: float = 10.0  # Exponential decay parameter for CLF
    lambda_xy: float = 10.0  # Exponential decay parameter for CLF for xy phase
    chirality_factor: float = 0.1  # Weight for chirality term
    ema_learning_rate: float = 0.0001  # EMA update rate for eigenvalue estimates
    dual_learning_rate: float = 3e-4  # Learning rate for dual variables (duals_learner)
    barrier: float = 1.0  # Barrier strength for dual norm constraints (duals_learner)

    # Episodic replay buffer
    max_time_offset: int | None = None  # Maximum time offset for sampling (None = episode length)
    
    # Logging and saving
    log_freq: int = 100
    plot_freq: int = 1000
    save_freq: int = 1000
    checkpoint_freq: int = 5000  # How often to save checkpoints (in gradient steps)
    save_model: bool = True
    plot_during_training: bool = False  # If True, creates plots during training (slow). If False, only exports data.
    results_dir: str = "./results"

    # Resuming training
    resume_from: str | None = None  # Path to results directory to resume from (e.g., './results/room4/room4__allo__0__42__1234567890')

    # Misc
    seed: int = 42
    exp_name: str | None = None
    exp_number: int = 0