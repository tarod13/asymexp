"""
Step-by-step visual debugging for the reward-shaping pipeline.

Entry point: run_step_by_step_debug(...)

Outputs written to <debug_dir>/:
  vector_field.png               — quiver plot of shaping gravity before training
  step_NNNN.png                  — per-step figure (heatmaps + time-series)
  episode_td_error_heatmap.png   — spatial average of |ΔQ| over the episode
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
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
# Maze background helper
# ---------------------------------------------------------------------------

def _make_maze_bg(env) -> np.ndarray:
    """[H, W] float array: 0=free, 1=wall (for imshow background)."""
    H, W   = env.height, env.width
    bg     = np.ones((H, W), dtype=np.float32)  # start all walls
    obs_np = np.array(env.obstacles)             # full-index list of obstacle cells
    free   = np.ones(H * W, dtype=bool)
    free[obs_np] = False
    for idx in range(H * W):
        if free[idx]:
            y, x     = divmod(idx, W)
            bg[y, x] = 0.0
    return bg


# ---------------------------------------------------------------------------
# Heatmap panel helper
# ---------------------------------------------------------------------------

def _draw_heatmap(
    ax,
    title: str,
    values_N: np.ndarray,       # [N] floats
    canonical_states: np.ndarray,
    env,
    s_curr: int | None = None,
    s_next: int | None = None,
    cmap: str = "viridis",
) -> None:
    """Draw a per-state heatmap overlaid on the maze, with optional state boxes."""
    H, W  = env.height, env.width
    grid  = np.full((H, W), np.nan, dtype=np.float32)
    for i, full_idx in enumerate(canonical_states):
        y, x     = divmod(int(full_idx), W)
        grid[y, x] = float(values_N[i])

    bg = _make_maze_bg(env)
    # Draw wall background
    ax.imshow(
        np.where(np.isnan(grid), bg, np.nan),
        origin="upper", cmap="gray_r", vmin=0, vmax=1,
        extent=[-0.5, W - 0.5, H - 0.5, -0.5],
    )
    # Draw heatmap values
    im = ax.imshow(
        grid, origin="upper", cmap=cmap,
        extent=[-0.5, W - 0.5, H - 0.5, -0.5],
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    for s_idx, color, label in [
        (s_curr, "white",  "sₜ"),
        (s_next, "yellow", "sₜ₊₁"),
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
                ha="center", va="top", fontsize=6, zorder=11,
                fontweight="bold")


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
    """Save a quiver plot of shaping gravity F(s,a) = γΦ(s') − Φ(s)."""
    N  = len(canonical_states)
    W  = env.width

    # Vectorised shaping field [N, 4]
    Phi       = potential                            # [N]
    Phi_prime = Phi[next_can]                        # [N, 4]
    F_all     = gamma * Phi_prime - Phi[:, None]     # [N, 4]

    # Quiver components — grid y increases downward (origin='upper')
    # so positive Vy correctly points downward (same direction as Down action)
    Vx = F_all[:, _RIGHT] - F_all[:, _LEFT]
    Vy = F_all[:, _DOWN]  - F_all[:, _UP]

    # 2-D coordinates for each canonical state
    xs = np.array([int(s) % W             for s in canonical_states], dtype=float)
    ys = np.array([int(s) // W            for s in canonical_states], dtype=float)

    # Magnitude for colour coding
    mag = np.sqrt(Vx**2 + Vy**2)

    fig, ax = plt.subplots(figsize=(8, 6))
    H       = env.height

    bg = _make_maze_bg(env)
    ax.imshow(bg, origin="upper", cmap="gray_r", vmin=0, vmax=1,
              extent=[-0.5, W - 0.5, H - 0.5, -0.5], alpha=0.4)

    # Quiver — scale_units='xy' keeps arrows in grid-cell units
    nonzero = mag > 1e-10
    if nonzero.any():
        q = ax.quiver(
            xs[nonzero], ys[nonzero],
            Vx[nonzero], Vy[nonzero],
            mag[nonzero],
            angles="xy", scale_units="xy", scale=4.0,
            cmap="plasma", clim=(0, mag.max() + 1e-10),
            width=0.004, headwidth=4, headlength=5,
            zorder=5,
        )
        plt.colorbar(q, ax=ax, label="|F(Right)−F(Left)|, |F(Down)−F(Up)|")

    # Goal and start markers
    g_full = int(canonical_states[goal_idx])
    s_full = int(canonical_states[start_idx])
    gx, gy = g_full % W, g_full // W
    sx, sy = s_full % W, s_full // W

    ax.plot(gx, gy, marker="*",  ms=14, color="gold",  zorder=15, label="goal")
    ax.plot(sx, sy, marker="o",  ms=10, color="cyan",   zorder=15,
            markeredgecolor="black", label="start")
    ax.legend(fontsize=8)

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)   # y-axis: 0 at top
    ax.set_title("Shaping gravity: net F(s, a) vector field\n"
                 r"$V_x=F(\rightarrow)-F(\leftarrow)$, "
                 r"$V_y=F(\downarrow)-F(\uparrow)$")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"  [debug] vector_field → {save_path}")


# ---------------------------------------------------------------------------
# Episode TD-error heatmap
# ---------------------------------------------------------------------------

def _plot_td_heatmap(
    env,
    td_sum: np.ndarray,   # [H, W]
    td_count: np.ndarray, # [H, W]
    save_path: Path,
) -> None:
    H, W   = env.height, env.width
    avg_td = np.where(td_count > 0, td_sum / np.maximum(td_count, 1), np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    bg      = _make_maze_bg(env)
    ax.imshow(bg, origin="upper", cmap="gray_r", vmin=0, vmax=1,
              extent=[-0.5, W - 0.5, H - 0.5, -0.5], alpha=0.4)
    im = ax.imshow(avg_td, origin="upper", cmap="hot",
                   extent=[-0.5, W - 0.5, H - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label="mean |ΔQ|")
    ax.set_title("Episode TD-error heatmap (mean |ΔQ| per grid cell)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"  [debug] TD heatmap  → {save_path}")


# ---------------------------------------------------------------------------
# Per-step figure drawing (updates axes in-place)
# ---------------------------------------------------------------------------

def _draw_step_figure(
    fig,
    ax_pot, ax_val,
    ax_R, ax_F, ax_Q, ax_TD,
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

    for ax in (ax_pot, ax_val, ax_R, ax_F, ax_Q, ax_TD):
        ax.cla()

    # --- Col 0: heatmaps ---
    _draw_heatmap(ax_pot, f"Φ(s)  [step {t}]",
                  potential_N, canonical_states, env, s, s_prime, cmap="viridis")
    _draw_heatmap(ax_val, f"V(s)=max Q  [step {t}]",
                  Q_table.max(axis=1), canonical_states, env, s, s_prime, cmap="plasma")

    # --- Col 1: time series ---
    def _plot_ts(ax, data, ylabel, color):
        ax.plot(steps, data, color=color, lw=1.2)
        ax.axvline(t, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    _plot_ts(ax_R,  hist_R,  "R",          "#2ca02c")
    _plot_ts(ax_F,  hist_F,  "F (shaping)","#1f77b4")
    _plot_ts(ax_Q,  hist_Q,  "Q(sₜ,aₜ)",  "#ff7f0e")
    _plot_ts(ax_TD, hist_TD, "|ΔQ|",       "#d62728")
    ax_TD.set_xlabel("step", fontsize=7)

    fig.suptitle(
        f"step {t}  |  s={s}→{s_prime}  a={action_names[a]}  "
        f"R={R:.2f}  F={F_shape:.4f}  Q(s,a)={Q_sa:.4f}  |ΔQ|={td_abs:.4f}",
        fontsize=8,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_step_by_step_debug(
    env,
    canonical_states: np.ndarray,
    full_to_can: np.ndarray,
    potential: np.ndarray,       # [N] float32 — shaping potential for this seed/goal
    goal_canonical: int,
    start_canonical: int,
    gamma: float,
    lr: float,
    epsilon: float,
    shaping_coef: float,
    max_steps: int,
    debug_dir: Path,
) -> None:
    """Run one training episode with step-by-step visualisation.

    Writes debug_dir/vector_field.png, debug_dir/step_NNNN.png for every
    step, and debug_dir/episode_td_error_heatmap.png at the end.
    """
    debug_dir.mkdir(parents=True, exist_ok=True)
    N = len(canonical_states)
    H, W = env.height, env.width

    canonical_states_jax = jnp.array(canonical_states, dtype=jnp.int32)
    full_to_can_jax      = jnp.array(full_to_can,      dtype=jnp.int32)
    potential_np         = np.array(potential,         dtype=np.float32)

    # Build transition table (vectorised, one-time cost)
    print("  [debug] Building transition table …")
    next_can = _build_transition_table(env, canonical_states_jax, full_to_can_jax, N)

    # Vector field plot
    _plot_vector_field(
        env, canonical_states, next_can, potential_np, gamma,
        goal_canonical, start_canonical,
        debug_dir / "vector_field.png",
    )

    # Q-table (plain numpy — updated in Python loop)
    Q = np.zeros((N, 4), dtype=np.float32)

    # Accumulators for spatial TD heatmap
    td_sum   = np.zeros((H, W), dtype=np.float64)
    td_count = np.zeros((H, W), dtype=np.int64)

    # History lists for time-series panels
    hist_R:  list[float] = []
    hist_F:  list[float] = []
    hist_Q:  list[float] = []
    hist_TD: list[float] = []

    # ── Create the step figure once ───────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(
        4, 2,
        figure=fig,
        width_ratios=[1.4, 1.0],
        hspace=0.55, wspace=0.38,
    )
    ax_pot = fig.add_subplot(gs[0:2, 0])
    ax_val = fig.add_subplot(gs[2:4, 0])
    ax_R   = fig.add_subplot(gs[0, 1])
    ax_F   = fig.add_subplot(gs[1, 1])
    ax_Q   = fig.add_subplot(gs[2, 1])
    ax_TD  = fig.add_subplot(gs[3, 1])

    # RNG for action selection
    rng = np.random.default_rng(0)

    s = start_canonical
    print(f"  [debug] Starting episode: start={start_canonical} goal={goal_canonical}")

    for t in range(max_steps):
        # ε-greedy action selection
        if rng.random() < epsilon:
            a = int(rng.integers(0, 4))
        else:
            q_s     = Q[s]
            max_q   = q_s.max()
            greedy  = np.where(q_s == max_q)[0]
            a       = int(rng.choice(greedy))

        # Environment step (JAX eager)
        step_key = jax.random.PRNGKey(t)
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

        # Reward and shaping
        reached  = (s_prime == goal_canonical)
        R        = 1.0 if reached else 0.0
        F_shape  = float(shaping_coef * (gamma * potential_np[s_prime] - potential_np[s]))
        r_total  = R + F_shape

        # TD update
        Q_sa_before = float(Q[s, a])
        target      = r_total if reached else r_total + gamma * float(Q[s_prime].max())
        td_error    = target - Q_sa_before
        Q[s, a]    += lr * td_error
        td_abs      = abs(td_error)

        # Record history
        hist_R.append(R)
        hist_F.append(F_shape)
        hist_Q.append(Q_sa_before)
        hist_TD.append(td_abs)

        # Accumulate spatial TD
        cy, cx       = divmod(full_idx, W)
        td_sum[cy, cx]   += td_abs
        td_count[cy, cx] += 1

        # Draw and save per-step figure
        _draw_step_figure(
            fig,
            ax_pot, ax_val,
            ax_R, ax_F, ax_Q, ax_TD,
            t, s, s_prime, a, R, F_shape, Q_sa_before, td_abs,
            hist_R, hist_F, hist_Q, hist_TD,
            potential_np, Q, canonical_states, env,
        )
        frame_path = debug_dir / f"step_{t:04d}.png"
        fig.savefig(frame_path, dpi=80, bbox_inches="tight")

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

    # Episode TD heatmap
    _plot_td_heatmap(env, td_sum, td_count, debug_dir / "episode_td_error_heatmap.png")
    print(f"  [debug] All frames written to {debug_dir}")
