"""
Step-by-step visual debugging for the reward-shaping pipeline.

Entry point: run_step_by_step_debug(...)

Outputs written to <debug_dir>/:
  vector_field.png               — quiver plot of shaping gravity before training
  step_NNNN.png                  — per-step figure (heatmaps + time-series)
  episode_td_error_heatmap.png   — spatial average of |ΔQ| over the episode
"""

from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.envs.gridworld import GridWorldState

# Action indices (matches gridworld.py action_effects ordering)
_UP    = 0
_RIGHT = 1
_DOWN  = 2
_LEFT  = 3

# Arrow length bounds (in grid-cell data units)
_ARROW_MIN_LEN = 0.08   # shortest arrow drawn for the weakest non-zero field
_ARROW_MAX_LEN = 0.42   # longest arrow; 0.42 < 0.5 so it stays inside its tile


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

def _build_transition_table(
    env,
    canonical_states_jax: jnp.ndarray,
    full_to_can_jax: jnp.ndarray,
    N: int,
) -> np.ndarray:
    """Return next_can[N, 4]: canonical index reached from state s via action a.

    Wall collisions keep the agent in state s (self-loop), matching the
    training loop behaviour.
    """
    fixed_key = jax.random.PRNGKey(0)

    def step_one(s_can: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        full_idx  = canonical_states_jax[s_can]
        env_state = GridWorldState(
            position=jnp.stack(
                [full_idx % env.width, full_idx // env.width]
            ).astype(env.dtype),
            terminal=jnp.array(False),
            steps=jnp.array(0, dtype=env.dtype),
        )
        next_state, _, _, _ = env.step(fixed_key, env_state, a)
        next_full = env.get_state_representation(next_state)
        can_idx   = full_to_can_jax[next_full]
        return jnp.where(can_idx >= 0, can_idx, s_can)

    # vmap over actions (inner), then over states (outer)
    actions_all = jnp.arange(4, dtype=jnp.int32)
    next_can_jax = jax.jit(
        jax.vmap(lambda s: jax.vmap(lambda a: step_one(s, a))(actions_all))
    )(jnp.arange(N, dtype=jnp.int32))

    return np.array(next_can_jax)  # [N, 4]


# ---------------------------------------------------------------------------
# Shared: build a [H, W] grid from per-canonical-state values
# (walls/unvisited cells remain NaN)
# ---------------------------------------------------------------------------

def _values_to_grid(
    values_N: np.ndarray,
    canonical_states: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    grid = np.full((H, W), np.nan, dtype=np.float32)
    for i, full_idx in enumerate(canonical_states):
        y, x      = divmod(int(full_idx), W)
        grid[y, x] = float(values_N[i])
    return grid


# ---------------------------------------------------------------------------
# Shared: draw grid lines (thin gray, same as training plots)
# ---------------------------------------------------------------------------

def _draw_grid_lines(ax, H: int, W: int) -> None:
    for i in range(H + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.3, alpha=0.3)


# ---------------------------------------------------------------------------
# Heatmap panel helper — matches plot_potential_vs_value style exactly
# ---------------------------------------------------------------------------

def _draw_heatmap(
    ax,
    fig,
    title: str,
    values_N: np.ndarray,
    canonical_states: np.ndarray,
    env,
    s_curr: int | None = None,
    s_next: int | None = None,
    cmap_name: str = "viridis",
) -> object:
    """Draw a per-state heatmap; walls = NaN → black (set_bad).

    Returns the colorbar object so the caller can remove it on the next frame.
    """
    H, W  = env.height, env.width
    grid  = _values_to_grid(values_N, canonical_states, H, W)

    cmap = cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="black")

    vmin = float(np.nanmin(grid)) if not np.all(np.isnan(grid)) else 0.0
    vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0

    im = ax.imshow(
        grid, cmap=cmap, origin="upper", interpolation="nearest",
        extent=[-0.5, W - 0.5, H - 0.5, -0.5],
        vmin=vmin, vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _draw_grid_lines(ax, H, W)

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")

    for s_idx, color, label in [
        (s_curr, "white",  "s\u209c"),        # sₜ
        (s_next, "yellow", "s\u209c\u208a\u2081"),  # sₜ₊₁
    ]:
        if s_idx is None:
            continue
        full_idx = int(canonical_states[s_idx])
        gy, gx   = divmod(full_idx, W)
        rect     = mpatches.Rectangle(
            (gx - 0.5, gy - 0.5), 1.0, 1.0,
            linewidth=2, edgecolor=color, facecolor="none", zorder=10,
        )
        ax.add_patch(rect)
        ax.text(gx, gy - 0.38, label, color=color,
                ha="center", va="top", fontsize=7, zorder=11,
                fontweight="bold")

    return cb


# ---------------------------------------------------------------------------
# Vector field
# ---------------------------------------------------------------------------

def _plot_vector_field(
    env,
    canonical_states: np.ndarray,
    next_can: np.ndarray,    # [N, 4]
    potential: np.ndarray,   # [N]
    gamma: float,
    goal_idx: int,
    start_idx: int,
    save_path: Path,
) -> None:
    """Save a quiver plot of shaping gravity F(s,a) = γΦ(s') − Φ(s).

    Arrows are normalised to unit direction then linearly re-scaled to
    [_ARROW_MIN_LEN, _ARROW_MAX_LEN] in data-unit space so they never
    overflow their tile, and the weakest non-zero arrow is still visible.
    """
    N  = len(canonical_states)
    W  = env.width
    H  = env.height

    # Vectorised shaping field [N, 4]
    Phi       = potential
    Phi_prime = Phi[next_can]
    F_all     = gamma * Phi_prime - Phi[:, None]

    # Net x/y shaping force per cell (signed)
    Vx = F_all[:, _RIGHT] - F_all[:, _LEFT]
    Vy = F_all[:, _DOWN]  - F_all[:, _UP]

    # 2-D coordinates
    xs = np.array([int(s) % W  for s in canonical_states], dtype=float)
    ys = np.array([int(s) // W for s in canonical_states], dtype=float)

    mag     = np.sqrt(Vx**2 + Vy**2)
    nonzero = mag > 1e-10

    # --- Re-scale arrows to [_ARROW_MIN_LEN, _ARROW_MAX_LEN] ---
    mag_min = mag[nonzero].min() if nonzero.any() else 1.0
    mag_max = mag[nonzero].max() if nonzero.any() else 1.0
    denom   = (mag_max - mag_min) if mag_max > mag_min else 1.0

    scaled_len = np.zeros_like(mag)
    scaled_len[nonzero] = (
        _ARROW_MIN_LEN
        + (mag[nonzero] - mag_min) / denom * (_ARROW_MAX_LEN - _ARROW_MIN_LEN)
    )

    # Unit-vector components scaled to desired length
    Ux = np.where(nonzero, Vx / np.where(nonzero, mag, 1.0) * scaled_len, 0.0)
    Uy = np.where(nonzero, Vy / np.where(nonzero, mag, 1.0) * scaled_len, 0.0)

    # ── Figure ───────────────────────────────────────────────────────
    grid_bg = _values_to_grid(
        np.zeros(N, dtype=np.float32), canonical_states, H, W
    )
    cmap_bg = cm.get_cmap("Greys").copy()
    cmap_bg.set_bad(color="black")

    fig, ax = plt.subplots(figsize=(max(6, W * 0.7), max(5, H * 0.7)))

    # Maze background (free cells = light gray, walls = black)
    free_grid = np.where(np.isnan(grid_bg), np.nan, 0.25)  # free → 0.25 → light gray
    ax.imshow(
        free_grid, cmap=cmap_bg, origin="upper", interpolation="nearest",
        extent=[-0.5, W - 0.5, H - 0.5, -0.5],
        vmin=0.0, vmax=1.0,
    )
    _draw_grid_lines(ax, H, W)

    # Quiver — scale=1 with scale_units="xy" means arrow length == vector magnitude
    # (in data units), so our pre-scaled vectors give the exact desired lengths.
    if nonzero.any():
        q = ax.quiver(
            xs[nonzero], ys[nonzero],
            Ux[nonzero], Uy[nonzero],
            mag[nonzero],
            angles="xy", scale_units="xy", scale=1.0,
            cmap="plasma", clim=(mag_min, mag_max),
            width=0.005, headwidth=4, headlength=5, headaxislength=4,
            zorder=5,
        )
        plt.colorbar(q, ax=ax, label="|F(→)−F(←)|² + |F(↓)−F(↑)|²  (raw magnitude)")

    # Goal / start markers
    g_full = int(canonical_states[goal_idx])
    s_full = int(canonical_states[start_idx])
    gx, gy = g_full % W, g_full // W
    sx, sy = s_full % W, s_full // W
    ax.plot(gx, gy, marker="*", ms=14, color="gold",  zorder=15, label="goal")
    ax.plot(sx, sy, marker="o", ms=10, color="cyan",  zorder=15,
            markeredgecolor="black", label="start")
    ax.legend(fontsize=8, loc="upper right")

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title(
        "Shaping gravity: net F(s, a) vector field\n"
        r"$V_x=F(\rightarrow)-F(\leftarrow)$, "
        r"$V_y=F(\downarrow)-F(\uparrow)$  "
        "(arrows normalised to [min,max] size)",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [debug] vector_field → {save_path}")


# ---------------------------------------------------------------------------
# Episode TD-error heatmap
# ---------------------------------------------------------------------------

def _plot_td_heatmap(
    env,
    td_sum: np.ndarray,
    td_count: np.ndarray,
    canonical_states: np.ndarray,
    save_path: Path,
) -> None:
    H, W   = env.height, env.width
    N      = len(canonical_states)

    # Compute per-canonical-state mean TD error
    avg_vals = np.zeros(N, dtype=np.float32)
    for i, full_idx in enumerate(canonical_states):
        y, x = divmod(int(full_idx), W)
        if td_count[y, x] > 0:
            avg_vals[i] = float(td_sum[y, x] / td_count[y, x])

    fig, ax = plt.subplots(figsize=(max(6, W * 0.7), max(5, H * 0.7)))

    grid = _values_to_grid(avg_vals, canonical_states, H, W)
    cmap = cm.get_cmap("hot").copy()
    cmap.set_bad(color="black")

    im = ax.imshow(
        grid, cmap=cmap, origin="upper", interpolation="nearest",
        extent=[-0.5, W - 0.5, H - 0.5, -0.5],
    )
    plt.colorbar(im, ax=ax, label="mean |ΔQ|")
    _draw_grid_lines(ax, H, W)

    ax.set_title("Episode TD-error heatmap (mean |ΔQ| per grid cell)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [debug] TD heatmap  → {save_path}")


# ---------------------------------------------------------------------------
# Per-step figure (reuses axes; removes colorbars before each redraw)
# ---------------------------------------------------------------------------

def _draw_step_figure(
    fig,
    ax_pot, ax_val,
    ax_R, ax_F, ax_Q, ax_TD,
    cb_store: list,           # mutable list[cb | None, cb | None] for pot and val
    t: int,
    s: int,
    s_prime: int,
    a: int,
    R: float,
    F_shape: float,
    Q_sa: float,
    td_abs: float,
    hist_R: list, hist_F: list, hist_Q: list, hist_TD: list,
    potential_N: np.ndarray,
    Q_table: np.ndarray,
    canonical_states: np.ndarray,
    env,
) -> None:
    action_names = ["Up", "Right", "Down", "Left"]
    steps = list(range(t + 1))

    # Remove existing colorbars (they are separate axes; ax.cla() does not touch them)
    for cb in cb_store:
        if cb is not None:
            try:
                cb.remove()
            except Exception:
                pass
    cb_store.clear()

    for ax in (ax_pot, ax_val, ax_R, ax_F, ax_Q, ax_TD):
        ax.cla()

    # --- Col 0: heatmaps ---
    cb_pot = _draw_heatmap(
        ax_pot, fig, f"Φ(s)  [step {t}]",
        potential_N, canonical_states, env, s, s_prime, cmap_name="viridis",
    )
    cb_val = _draw_heatmap(
        ax_val, fig, f"V(s)=max Q  [step {t}]",
        Q_table.max(axis=1), canonical_states, env, s, s_prime, cmap_name="plasma",
    )
    cb_store.extend([cb_pot, cb_val])

    # --- Col 1: time series ---
    def _plot_ts(ax, data, ylabel, color):
        ax.plot(steps, data, color=color, lw=1.2)
        ax.axvline(t, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    _plot_ts(ax_R,  hist_R,  "R",           "#2ca02c")
    _plot_ts(ax_F,  hist_F,  "F (shaping)", "#1f77b4")
    _plot_ts(ax_Q,  hist_Q,  "Q(s\u209c,a\u209c)", "#ff7f0e")
    _plot_ts(ax_TD, hist_TD, "|ΔQ|",        "#d62728")
    ax_TD.set_xlabel("step", fontsize=8)

    fig.suptitle(
        f"step {t}  |  s={s}→{s_prime}  a={action_names[a]}  "
        f"R={R:.2f}  F={F_shape:.4f}  Q(s,a)={Q_sa:.4f}  |ΔQ|={td_abs:.4f}",
        fontsize=9,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_step_by_step_debug(
    env,
    canonical_states: np.ndarray,
    full_to_can: np.ndarray,
    potential: np.ndarray,
    goal_canonical: int,
    start_canonical: int,
    gamma: float,
    lr: float,
    epsilon: float,
    shaping_coef: float,
    max_steps: int,
    debug_dir: Path,
) -> None:
    """Run one training episode with step-by-step visualisation."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    N = len(canonical_states)
    H, W = env.height, env.width

    canonical_states_jax = jnp.array(canonical_states, dtype=jnp.int32)
    full_to_can_jax      = jnp.array(full_to_can,      dtype=jnp.int32)
    potential_np         = np.array(potential,          dtype=np.float32)

    print("  [debug] Building transition table …")
    next_can = _build_transition_table(env, canonical_states_jax, full_to_can_jax, N)

    _plot_vector_field(
        env, canonical_states, next_can, potential_np, gamma,
        goal_canonical, start_canonical,
        debug_dir / "vector_field.png",
    )

    Q        = np.zeros((N, 4), dtype=np.float32)
    td_sum   = np.zeros((H, W), dtype=np.float64)
    td_count = np.zeros((H, W), dtype=np.int64)

    hist_R:  list[float] = []
    hist_F:  list[float] = []
    hist_Q:  list[float] = []
    hist_TD: list[float] = []

    # ── Create the step figure once ───────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(
        4, 2, figure=fig,
        width_ratios=[1.4, 1.0],
        hspace=0.55, wspace=0.38,
    )
    ax_pot = fig.add_subplot(gs[0:2, 0])
    ax_val = fig.add_subplot(gs[2:4, 0])
    ax_R   = fig.add_subplot(gs[0, 1])
    ax_F   = fig.add_subplot(gs[1, 1])
    ax_Q   = fig.add_subplot(gs[2, 1])
    ax_TD  = fig.add_subplot(gs[3, 1])

    # Mutable list to track active colorbar objects (pot, val)
    cb_store: list = []

    rng = np.random.default_rng(0)
    s   = start_canonical
    print(f"  [debug] Starting episode: start={start_canonical} goal={goal_canonical}")

    for t in range(max_steps):
        # ε-greedy
        if rng.random() < epsilon:
            a = int(rng.integers(0, 4))
        else:
            q_s    = Q[s]
            max_q  = q_s.max()
            greedy = np.where(q_s == max_q)[0]
            a      = int(rng.choice(greedy))

        # Environment step
        step_key  = jax.random.PRNGKey(t)
        full_idx  = int(canonical_states[s])
        env_state = GridWorldState(
            position=jnp.array([full_idx % env.width, full_idx // env.width],
                                dtype=env.dtype),
            terminal=jnp.array(False),
            steps=jnp.array(0, dtype=env.dtype),
        )
        next_env_state, _, _, _ = env.step(step_key, env_state, jnp.array(a))
        next_full = int(env.get_state_representation(next_env_state))
        can_idx   = int(full_to_can[next_full])
        s_prime   = can_idx if can_idx >= 0 else s

        # Reward + shaping
        reached     = (s_prime == goal_canonical)
        R           = 1.0 if reached else 0.0
        F_shape     = float(shaping_coef * (gamma * potential_np[s_prime] - potential_np[s]))
        r_total     = R + F_shape

        # TD update
        Q_sa_before = float(Q[s, a])
        target      = r_total if reached else r_total + gamma * float(Q[s_prime].max())
        td_error    = target - Q_sa_before
        Q[s, a]    += lr * td_error
        td_abs      = abs(td_error)

        hist_R.append(R)
        hist_F.append(F_shape)
        hist_Q.append(Q_sa_before)
        hist_TD.append(td_abs)

        cy, cx           = divmod(full_idx, W)
        td_sum[cy, cx]   += td_abs
        td_count[cy, cx] += 1

        _draw_step_figure(
            fig,
            ax_pot, ax_val,
            ax_R, ax_F, ax_Q, ax_TD,
            cb_store,
            t, s, s_prime, a, R, F_shape, Q_sa_before, td_abs,
            hist_R, hist_F, hist_Q, hist_TD,
            potential_np, Q, canonical_states, env,
        )
        frame_path = debug_dir / f"step_{t:04d}.png"
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")

        print(f"    step {t:4d}  s={s:4d}→{s_prime:4d}  "
              f"a={a}  R={R:.1f}  F={F_shape:+.4f}  "
              f"Q={Q_sa_before:.4f}  |ΔQ|={td_abs:.4f}  → {frame_path.name}")

        s = s_prime
        if reached:
            print(f"  [debug] Goal reached at step {t}.")
            break
    else:
        print(f"  [debug] Episode ended without reaching goal after {max_steps} steps.")

    plt.close(fig)

    _plot_td_heatmap(env, td_sum, td_count, canonical_states, debug_dir / "episode_td_error_heatmap.png")
    print(f"  [debug] All frames written to {debug_dir}")
