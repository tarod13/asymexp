from typing import Dict
from pathlib import Path
import pickle
import numpy as np
from flax.training.train_state import TrainState
from src.config.ded_clf import Args


def save_checkpoint(
    encoder_state: TrainState,
    metrics_history: list,
    gradient_step: int,
    save_path: Path,
    args: Args,
    rng_state: np.ndarray = None
):
    """
    Save a training checkpoint.

    Args:
        encoder_state: Current training state
        metrics_history: List of metrics dictionaries
        gradient_step: Current gradient step
        save_path: Path to save the checkpoint
        args: Training arguments
        rng_state: Current random state (optional)
    """
    checkpoint = {
        'gradient_step': gradient_step,
        'params': encoder_state.params,
        'opt_state': encoder_state.opt_state,
        'metrics_history': metrics_history,
        'args': vars(args),
    }

    if rng_state is not None:
        checkpoint['rng_state'] = rng_state

    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """
    Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Resuming from gradient step: {checkpoint['gradient_step']}")

    return checkpoint