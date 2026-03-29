"""
Step-by-step visual debugging for the reward-shaping pipeline.

Entry point: run_step_by_step_debug(...)

Outputs written to <debug_dir>/:
  vector_field.png               — quiver plot of shaping gravity before training
  step_NNNN.png                  — per-step figure (heatmaps + time-series)
  episode_td_error_heatmap.png   — spatial average of |ΔQ| over the episode
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.envs.gridworld import GridWorldState
from src.utils.plotting import _draw_door_markers, _draw_portal_tile_overlays

# Action indices (matches gridworld.py action_effects ordering)
_UP    = 0
_RIGHT = 1
_DOWN  = 2
_LEFT  = 3

# Arrow length bounds (in grid-cell data units)
_ARROW_MIN_LEN = 0.08   # shortest arrow for the weakest non-zero field
_ARROW_MAX_LEN = 0.42   # longest arrow; stays inside its tile (< 0.5)

# Maze colours (match training-plot convention: wall_color='gray')
_WALL_RGBA  = np.array([0.5, 0.5, 0.5, 1.0], dtype=float)   # mid-gray
_FREE_RGBA  = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)   # white


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

def _build_transition_table(
    env,
    canonical_states_jax: jnp.ndarray,
    full_to_can_jax: jnp.ndarray,
    N: int,
) -> np.ndarray:
    """Return next_can[N, 4]: canonical index reached from state s via action a."""
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

    actions_all = jnp.arange(4, dtype=jnp.int32)
    next_can_jax = jax.jit(
        jax.vmap(lambda s: jax.vmap(lambda a: step_one(s, a))(actions_all))
    )(jnp.arange(N, dtype=jnp.int32))

    return np.array(next_can_jax)  # [N, 4]


# ---------------------------------------------------------------------------
# Maze RGBA background helper
# ---------------------------------------------------------------------------

def _make_maze_bg_rgba(env, canonical_states: np.ndarray) -> np.ndarray:
    """Return [H, W, 4] RGBA: free cells = white, walls = gray.

    Matches the training-plot convention (wall_color='gray',
    visualize_eigenvector_on_grid).
    """
    H, W  = env.height, env.width
    bg    = np.tile(_WALL_RGBA, (H, W, 1))  # start all walls
    for full_idx in canonical_states:
        y, x    = divmod(int(full_idx), W)
        bg[y, x] = _FREE_RGBA
    return bg


# ---------------------------------------------------------------------------
# Shared: build a [H, W] grid from per-canonical-state values
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
# Shared: draw grid lines
# ---------------------------------------------------------------------------

def _draw_grid_lines(ax, H: int, W: int) -> None:
    for i in range(H + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.3, alpha=0.3)
    for j in range(W + 1):
        ax.axvline(j - 0.5, color="gray", linewidth=0.3, alpha=0.3)


# ---------------------------------------------------------------------------
# Heatmap panel helper
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
    portals=None,
    portal_sources=None,
    portal_ends=None,
) -> object:
    """Draw a per-state heatmap on a white-free / gray-wall maze background.

    Layer 1: RGBA background (white free, gray walls) — always fully opaque.
    Layer 2: data colormap — NaN renders as transparent so layer 1 shows through.

    Returns the colorbar object so the caller can call cb.remove() next frame.
    """
    H, W  = env.height, env.width
    grid  = _values_to_grid(values_N, canonical_states, H, W)

    cmap_data = cm.get_cmap(cmap_name).copy()
    cmap_data.set_bad(color="none")  # transparent → background shows through

    # Layer 1: maze background
    bg_rgba = _make_maze_bg_rgba(env, canonical_states)
    extent  = [-0.5, W - 0.5, H - 0.5, -0.5]
    ax.imshow(bg_rgba, origin="upper", interpolation="nearest", extent=extent)

    # Layer 2: data
    vmin = float(np.nanmin(grid)) if not np.all(np.isnan(grid)) else 0.0
    vmax = float(np.nanmax(grid)) if not np.all(np.isnan(grid)) else 1.0
    im   = ax.imshow(
        grid, cmap=cmap_data, origin="upper", interpolation="nearest",
        extent=extent, vmin=vmin, vmax=vmax,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _draw_grid_lines(ax, H, W)
    _draw_door_markers(ax, portals, W)
    _draw_portal_tile_overlays(ax, portal_sources, portal_ends, W)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")

    for s_idx, color, label in [
        (s_curr, "white",  "s\u209c"),
        (s_next, "yellow", "s\u209c\u208a\u2081"),
    ]:
        if s_idx is None:
            continue
        full_idx = int(canonical_states[s_idx])
        gy, gx   = divmod(full_idx, W)
        rect = mpatches.Rectangle(
            (gx - 0.5, gy - 0.5), 1.0, 1.0,
            linewidth=2, edgecolor=color, facecolor="none", zorder=10,
        )
        ax.add_patch(rect)
        ax.text(gx, gy - 0.38, label, color=color,
                ha="center", va="top", fontsize=7, zorder=11, fontweight="bold")

    return cb


# ---------------------------------------------------------------------------
# Vector field
# ---------------------------------------------------------------------------

def _plot_vector_field(
    env,
    canonical_states: np.ndarray,
    next_can: np.ndarray,
    potential: np.ndarray,
    gamma: float,
    goal_idx: int,
    start_idx: int,
    save_path: Path,
    portals=None,
    portal_sources=None,
    portal_ends=None,
    potential_mode: str = "negative",
) -> None:
    """Quiver plot of shaping gravity F(s,a) = γΦ(s') − Φ(s).

    Arrows are normalised to unit direction then linearly re-scaled to
    [_ARROW_MIN_LEN, _ARROW_MAX_LEN] in data-unit space.
    """
    N = len(canonical_states)
    W = env.width
    H = env.height

    Phi       = potential
    Phi_prime = Phi[next_can]
    F_all     = gamma * Phi_prime - Phi[:, None]

    Vx = F_all[:, _RIGHT] - F_all[:, _LEFT]
    Vy = F_all[:, _DOWN]  - F_all[:, _UP]

    xs  = np.array([int(s) % W  for s in canonical_states], dtype=float)
    ys  = np.array([int(s) // W for s in canonical_states], dtype=float)
    mag = np.sqrt(Vx**2 + Vy**2)

    nonzero = mag > 1e-10
    mag_min = mag[nonzero].min() if nonzero.any() else 1.0
    mag_max = mag[nonzero].max() if nonzero.any() else 1.0
    denom   = (mag_max - mag_min) if mag_max > mag_min else 1.0

    scaled_len            = np.zeros_like(mag)
    scaled_len[nonzero]   = (
        _ARROW_MIN_LEN
        + (mag[nonzero] - mag_min) / denom * (_ARROW_MAX_LEN - _ARROW_MIN_LEN)
    )
    safe_mag = np.where(nonzero, mag, 1.0)
    Ux = np.where(nonzero, Vx / safe_mag * scaled_len, 0.0)
    Uy = np.where(nonzero, Vy / safe_mag * scaled_len, 0.0)

    fig, ax = plt.subplots(figsize=(max(6, W * 0.7), max(5, H * 0.7)))

    bg_rgba = _make_maze_bg_rgba(env, canonical_states)
    extent  = [-0.5, W - 0.5, H - 0.5, -0.5]
    ax.imshow(bg_rgba, origin="upper", interpolation="nearest", extent=extent)
    _draw_grid_lines(ax, H, W)
    _draw_door_markers(ax, portals, W)
    _draw_portal_tile_overlays(ax, portal_sources, portal_ends, W)

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
        plt.colorbar(q, ax=ax, label="raw magnitude |F(→)−F(←)|² + |F(↓)−F(↑)|²")

    g_full = int(canonical_states[goal_idx])
    s_full = int(canonical_states[start_idx])
    gx, gy = g_full % W, g_full // W
    sx, sy = s_full % W, s_full // W
    ax.plot(gx, gy, marker="*", ms=14, color="gold", zorder=15, label="goal")
    ax.plot(sx, sy, marker="o", ms=10, color="cyan", zorder=15,
            markeredgecolor="black", label="start")
    ax.legend(fontsize=8, loc="upper right")

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_title(
        f"Shaping gravity: net F(s, a) vector field  [mode: {potential_mode}]\n"
        r"$V_x=F(\rightarrow)-F(\leftarrow)$, "
        r"$V_y=F(\downarrow)-F(\uparrow)$  "
        "(arrows normalised to [min, max] size)",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [debug] vector_field → {save_path}")


# ---------------------------------------------------------------------------
# Episode TD-error heatmap (same visual format as vector field)
# ---------------------------------------------------------------------------

def _plot_td_heatmap(
    env,
    td_sum: np.ndarray,
    td_count: np.ndarray,
    canonical_states: np.ndarray,
    save_path: Path,
    portals=None,
    portal_sources=None,
    portal_ends=None,
) -> None:
    H, W = env.height, env.width
    N    = len(canonical_states)

    # NaN for unvisited cells → they show as white (background) not black
    avg_vals = np.full(N, np.nan, dtype=np.float32)
    for i, full_idx in enumerate(canonical_states):
        y, x = divmod(int(full_idx), W)
        if td_count[y, x] > 0:
            avg_vals[i] = float(td_sum[y, x] / td_count[y, x])

    grid      = _values_to_grid(avg_vals, canonical_states, H, W)
    cmap_data = cm.get_cmap("hot").copy()
    cmap_data.set_bad(color="none")  # transparent → background shows through

    fig, ax = plt.subplots(figsize=(max(6, W * 0.7), max(5, H * 0.7)))
    extent  = [-0.5, W - 0.5, H - 0.5, -0.5]

    # Layer 1: maze background (white free, gray walls)
    bg_rgba = _make_maze_bg_rgba(env, canonical_states)
    ax.imshow(bg_rgba, origin="upper", interpolation="nearest", extent=extent)

    # Layer 2: TD error data (only visited cells)
    im = ax.imshow(grid, cmap=cmap_data, origin="upper", interpolation="nearest",
                   extent=extent)
    plt.colorbar(im, ax=ax, label="mean |ΔQ|")

    _draw_grid_lines(ax, H, W)
    _draw_door_markers(ax, portals, W)
    _draw_portal_tile_overlays(ax, portal_sources, portal_ends, W)
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
    cb_store: list,
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
    portals=None,
    portal_sources=None,
    portal_ends=None,
) -> None:
    action_names = ["Up", "Right", "Down", "Left"]
    steps = list(range(t + 1))

    for cb in cb_store:
        if cb is not None:
            try:
                cb.remove()
            except Exception:
                pass
    cb_store.clear()

    for ax in (ax_pot, ax_val, ax_R, ax_F, ax_Q, ax_TD):
        ax.cla()

    cb_pot = _draw_heatmap(
        ax_pot, fig, f"Φ(s)  [step {t}]",
        potential_N, canonical_states, env, s, s_prime, cmap_name="viridis",
        portals=portals, portal_sources=portal_sources, portal_ends=portal_ends,
    )
    cb_val = _draw_heatmap(
        ax_val, fig, f"V(s)=max Q  [step {t}]",
        Q_table.max(axis=1), canonical_states, env, s, s_prime, cmap_name="plasma",
        portals=portals, portal_sources=portal_sources, portal_ends=portal_ends,
    )
    cb_store.extend([cb_pot, cb_val])

    def _plot_ts(ax, data, ylabel, color):
        ax.plot(steps, data, color=color, lw=1.2)
        ax.axvline(t, color="red", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    _plot_ts(ax_R,  hist_R,  "R",                   "#2ca02c")
    _plot_ts(ax_F,  hist_F,  "F (shaping)",          "#1f77b4")
    _plot_ts(ax_Q,  hist_Q,  "Q(s\u209c,a\u209c)",  "#ff7f0e")
    _plot_ts(ax_TD, hist_TD, "|ΔQ|",                "#d62728")
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
    n_step_td: int = 1,
    portals=None,
    portal_sources=None,
    portal_ends=None,
    potential_mode: str = "negative",
    potential_temp: float = 1.0,
    potential_power: float = 0.5,
    potential_base: float = 0.99,
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
        portals=portals, portal_sources=portal_sources, portal_ends=portal_ends,
        potential_mode=potential_mode,
    )

    Q        = np.zeros((N, 4), dtype=np.float32)
    td_sum   = np.zeros((H, W), dtype=np.float64)
    td_count = np.zeros((H, W), dtype=np.int64)

    hist_R:  list[float] = []
    hist_F:  list[float] = []
    hist_Q:  list[float] = []
    hist_TD: list[float] = []

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
    cb_store: list = []

    n   = n_step_td
    buf = deque()   # stores (s, a, r_total) tuples; managed manually (no maxlen)

    # Pre-compute gamma powers for the return sum: gamma^0, gamma^1, …, gamma^{n-1}
    gamma_pows = [gamma ** k for k in range(n)]

    rng = np.random.default_rng(0)
    s   = start_canonical
    print(f"  [debug] Starting episode: start={start_canonical} goal={goal_canonical}"
          f"  potential_mode={potential_mode}  potential_temp={potential_temp}")

    for t in range(max_steps):
        if rng.random() < epsilon:
            a = int(rng.integers(0, 4))
        else:
            q_s    = Q[s]
            max_q  = q_s.max()
            greedy = np.where(q_s == max_q)[0]
            a      = int(rng.choice(greedy))

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

        reached = (s_prime == goal_canonical)
        R       = 1.0 if reached else 0.0
        F_shape = float(shaping_coef * (gamma * potential_np[s_prime] - potential_np[s]))
        r_total = R + F_shape

        # ── Write transition to n-step buffer ─────────────────────────
        buf.append((s, a, r_total))
        episode_end   = reached or (t == max_steps - 1)
        bootstrap_val = 0.0 if reached else float(Q[s_prime].max())

        # Collect (state, |ΔQ|) pairs for all updates that fire this step
        updates_this_step: list[tuple[int, float]] = []

        # ── Normal n-step update (buffer full, mid-episode) ────────────
        if len(buf) == n and not episode_end:
            s0, a0, _ = buf[0]
            G = sum(gamma_pows[k] * buf[k][2] for k in range(n)) + (gamma ** n) * bootstrap_val
            td_error = G - float(Q[s0, a0])
            Q[s0, a0] += lr * td_error
            updates_this_step.append((s0, abs(td_error)))
            buf.popleft()

        # ── Episode-end flush: update all remaining buffer entries ──────
        elif episode_end:
            buf_list = list(buf)   # snapshot; index 0 = oldest
            m = len(buf_list)
            for i in range(m):
                s_i, a_i, _ = buf_list[i]
                remaining = m - i
                G_i = (
                    sum(gamma_pows[k] * buf_list[i + k][2] for k in range(remaining))
                    + (gamma ** remaining) * bootstrap_val
                )
                td_error_i = G_i - float(Q[s_i, a_i])
                Q[s_i, a_i] += lr * td_error_i
                updates_this_step.append((s_i, abs(td_error_i)))
            buf.clear()

        # ── Per-step recording ─────────────────────────────────────────
        Q_sa_now   = float(Q[s, a])   # current state's Q value (post any update)
        td_abs     = sum(delta for _, delta in updates_this_step)

        hist_R.append(R)
        hist_F.append(F_shape)
        hist_Q.append(Q_sa_now)
        hist_TD.append(td_abs)

        # TD heatmap: accumulate only at states that were actually updated
        for s_upd, delta in updates_this_step:
            full_upd        = int(canonical_states[s_upd])
            cy, cx          = divmod(full_upd, W)
            td_sum[cy, cx]  += delta
            td_count[cy, cx] += 1

        _draw_step_figure(
            fig,
            ax_pot, ax_val,
            ax_R, ax_F, ax_Q, ax_TD,
            cb_store,
            t, s, s_prime, a, R, F_shape, Q_sa_now, td_abs,
            hist_R, hist_F, hist_Q, hist_TD,
            potential_np, Q, canonical_states, env,
            portals=portals, portal_sources=portal_sources, portal_ends=portal_ends,
        )
        frame_path = debug_dir / f"step_{t:04d}.png"
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")

        print(f"    step {t:4d}  s={s:4d}→{s_prime:4d}  "
              f"a={a}  R={R:.1f}  F={F_shape:+.4f}  "
              f"Q={Q_sa_now:.4f}  |ΔQ|={td_abs:.4f}  → {frame_path.name}")

        s = s_prime
        if reached:
            print(f"  [debug] Goal reached at step {t}.")
            break
    else:
        print(f"  [debug] Episode ended without reaching goal after {max_steps} steps.")

    plt.close(fig)

    _plot_td_heatmap(env, td_sum, td_count, canonical_states,
                     debug_dir / "episode_td_error_heatmap.png",
                     portals=portals, portal_sources=portal_sources, portal_ends=portal_ends)
    print(f"  [debug] All frames written to {debug_dir}")
