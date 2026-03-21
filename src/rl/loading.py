"""
Utilities for loading pre-trained Laplacian representation checkpoints and
computing ground-truth Laplacian eigenvectors from an environment.
"""

import json
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from src.utils.laplacian import compute_laplacian, compute_eigendecomposition


def load_model(
    model_dir: Path,
    use_gt: bool = False,
    checkpoint_prefix: str = "final_",
) -> dict:
    """
    Load eigenvectors and eigenvalues from a results directory produced by
    train_lap_rep.py.

    Parameters
    ----------
    model_dir : Path
        Results directory written by train_lap_rep.py.
    use_gt : bool
        When True, load the ground-truth eigenvectors saved alongside the
        learned ones (gt_left_real.npy etc.) instead of the learned files.
    checkpoint_prefix : str
        Prefix used when naming the learned-eigenvector .npy files and the
        model checkpoint.  The default ``"final_"`` loads the files written at
        the end of training (``final_learned_*.npy``, ``models/final_model.pkl``).
        Pass e.g. ``"latest_"`` to load in-progress checkpoints instead.
        Has no effect when ``use_gt=True``.

    Returns a dict with keys:
        training_args    – original training hyper-parameters (dict)
        canonical_states – np.ndarray of free-state full indices
        left_real / left_imag / right_real / right_imag  – [N, K] float arrays
        eigenvalues_real / eigenvalues_imag              – [K] float arrays
        eigenvalue_type  – 'kernel' (learned) or 'laplacian' (GT)
    """
    model_dir = Path(model_dir)

    with open(model_dir / "args.json") as f:
        training_args = json.load(f)

    with open(model_dir / "viz_metadata.pkl", "rb") as f:
        viz_metadata = pickle.load(f)
    canonical_states = np.array(viz_metadata["canonical_states"])

    if use_gt:
        left_real  = np.load(model_dir / "gt_left_real.npy")
        left_imag  = np.load(model_dir / "gt_left_imag.npy")
        right_real = np.load(model_dir / "gt_right_real.npy")
        right_imag = np.load(model_dir / "gt_right_imag.npy")
        eig_real   = np.load(model_dir / "gt_eigenvalues_real.npy")
        eig_imag   = np.load(model_dir / "gt_eigenvalues_imag.npy")
        eigenvalue_type = "laplacian"
    else:
        # Raw learned eigenvectors (adjoint left eigenvectors, as used during training)
        left_real  = np.load(model_dir / f"{checkpoint_prefix}learned_left_real.npy")
        left_imag  = np.load(model_dir / f"{checkpoint_prefix}learned_left_imag.npy")
        right_real = np.load(model_dir / f"{checkpoint_prefix}learned_right_real.npy")
        right_imag = np.load(model_dir / f"{checkpoint_prefix}learned_right_imag.npy")
        # Eigenvalue estimates stored inside the model checkpoint
        ckpt_path = model_dir / "models" / f"{checkpoint_prefix}model.pkl"
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        p = ckpt["params"]
        if "lambda_real" in p:
            eig_real = np.array(p["lambda_real"])
            eig_imag = np.array(p["lambda_imag"])
        else:
            # 'separate' eigenvalue estimation: average x and y estimates
            eig_real = np.array(0.5 * (p["lambda_x_real"] + p["lambda_y_real"]))
            eig_imag = np.array(0.5 * (p["lambda_x_imag"] + p["lambda_y_imag"]))
        eigenvalue_type = "kernel"

    return dict(
        training_args=training_args,
        canonical_states=canonical_states,
        left_real=left_real,
        left_imag=left_imag,
        right_real=right_real,
        right_imag=right_imag,
        eigenvalues_real=eig_real,
        eigenvalues_imag=eig_imag,
        eigenvalue_type=eigenvalue_type,
    )


def compute_gt_model_data(
    env,
    canonical_states: np.ndarray,
    gamma: float,
    delta: float,
    num_eigenvectors: int,
) -> dict:
    """
    Compute the exact ground-truth Laplacian eigenvectors from the environment.

    Builds the analytic transition matrix P by iterating over every canonical
    state and action, respecting portals (stochastic destinations), soft doors
    (partially blocked transitions), and normal physics.  Then computes
    L = (1+δ)I - (1-γ)P·SR_γ and its eigendecomposition — exactly as training
    does when the GT files are not yet available on disk.
    """
    N = len(canonical_states)
    full_to_canonical = {int(s): i for i, s in enumerate(canonical_states)}
    # action effects: up=(0,-1), right=(+1,0), down=(0,+1), left=(-1,0)
    action_effects = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    asym    = env.asymmetric_transitions if env.has_doors    else {}
    portals = env.portals                if env.has_portals  else {}

    P = np.zeros((N, N), dtype=np.float64)
    for a, (dx, dy) in enumerate(action_effects):
        for s_idx in range(N):
            full_s = int(canonical_states[s_idx])
            y, x   = divmod(full_s, env.width)

            # 1. Portal (takes priority over doors and physics)
            if (full_s, a) in portals:
                dests, probs = portals[(full_s, a)]
                for dest, prob in zip(dests, probs):
                    d_idx = full_to_canonical.get(int(dest), s_idx)
                    P[s_idx, d_idx] += 0.25 * float(prob)
                continue

            # 2. Door — reduces forward probability; remainder stays in place
            door_prob = asym.get((full_s, a), 1.0)

            # 3. Normal physics
            nx, ny = x + dx, y + dy
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                dest_idx = s_idx  # boundary → stay
            else:
                next_full = ny * env.width + nx
                dest_idx  = full_to_canonical.get(next_full, s_idx)  # obstacle → stay

            P[s_idx, dest_idx] += 0.25 * door_prob
            P[s_idx, s_idx]    += 0.25 * (1.0 - door_prob)

    laplacian = compute_laplacian(jnp.array(P), gamma=gamma, delta=delta)
    eig       = compute_eigendecomposition(laplacian, k=num_eigenvectors, ascending=True)
    return dict(
        training_args    = {"gamma": gamma, "delta": delta},
        canonical_states = canonical_states,
        left_real        = np.array(eig["left_eigenvectors_real"]),
        left_imag        = np.array(eig["left_eigenvectors_imag"]),
        right_real       = np.array(eig["right_eigenvectors_real"]),
        right_imag       = np.array(eig["right_eigenvectors_imag"]),
        eigenvalues_real = np.array(eig["eigenvalues_real"]),
        eigenvalues_imag = np.array(eig["eigenvalues_imag"]),
        eigenvalue_type  = "laplacian",
    )
