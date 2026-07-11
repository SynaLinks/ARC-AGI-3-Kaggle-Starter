"""Live "arcade cabinet" monitor for local runs. Never ships to Kaggle:
`scripts/build_notebook.py` only splices my_agent.py, and the hook there is
gated behind ARC_MONITOR=1 plus a guarded import.

Run it with:  make monitor GAME=ls20

Four panels, redrawn once per action:
  - the current frame, as the agent's world model sees it (RGB, ARC palette),
    with a ring + crosshair wherever the action about to run clicks (ACTION6);
  - a 2-D PCA (PC1 × PC2) of the world-model latents z: the top two principal
    components, refit each frame, each axis autoscaled independently so the
    cloud fills the panel. colour = latent distance from each state to the goal
    (ARC_PCA_COLOR=time falls back to time), ring = current state, grey fan =
    the CEM's imagined candidate states, ✕ = the elite plan, ★ = the goal;
  - the CEM plan decoded to pixels — the imagined frame after each planned
    action, i.e. what the agent *expects* to see, one tile per horizon step;
  - an arcade button row that lights up the action just pressed.

Each redraw is also recorded — the whole arcade, all four panels — as a PNG
under recordings/monitor/<game>-<timestamp>/ (override the root with
ARC_MONITOR_CACHE), so a run can be reviewed after the window closes.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

# uv-managed CPythons ship a _tkinter that matplotlib's tkagg can't load
# (missing _tkinter.__file__), so probe each backend by importing its module
# and fall back: TkAgg → QtAgg (PySide6).
import importlib

for _backend, _module in (("TkAgg", "matplotlib.backends.backend_tkagg"),
                          ("QtAgg", "matplotlib.backends.backend_qtagg")):
    try:
        importlib.import_module(_module)
        matplotlib.use(_backend)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BG = "#101018"
FG = "#d8d8e0"
DIM = "#3a3a55"

# One arcade colour per button (cycled if there are more actions than colours).
BUTTON_COLORS = ["#f93c31", "#1e93ff", "#ffdc00", "#4fcc30", "#ff851b", "#a356d6", "#cccccc"]


# Official action semantics (https://docs.arcprize.org/actions): cap glyph +
# caption text. The AGENT never sees these — they're for the human watching.
ACTION_SEMANTICS = {
    "RESET": ("R", "reset"),
    "ACTION1": ("↑", "up"),
    "ACTION2": ("↓", "down"),
    "ACTION3": ("←", "left"),
    "ACTION4": ("→", "right"),
    "ACTION5": ("F", "interact"),
    "ACTION6": ("◎", "click x,y"),
    "ACTION7": ("↺", "undo"),
}


def _short(name: str) -> str:
    """What's printed on the button cap: ↑ for ACTION1, ◎ for ACTION6, …"""
    if name in ACTION_SEMANTICS:
        return ACTION_SEMANTICS[name][0]
    if name.startswith("ACTION"):
        return name[len("ACTION"):]
    return name[:1]


def _caption(name: str) -> str:
    """Text under the row for the pressed button: 'ACTION1 ↑ up'."""
    if name in ACTION_SEMANTICS:
        glyph, meaning = ACTION_SEMANTICS[name]
        return f"{name} {glyph} {meaning}"
    return name


class ArcadeMonitor:
    """One window; call `update(rgb, z, action_name, ...)` once per chosen action."""

    # Only the last RECENT points get full colour and the trajectory line —
    # older ones dim into context.
    RECENT = 150

    # Every redraw is also recorded (the whole arcade figure) as a PNG under
    # <ARC_MONITOR_CACHE or recordings/monitor>/<game>-<timestamp>/ so a run
    # can be reviewed after the window closes. `make clean` removes it.
    CACHE_ROOT = Path(os.environ.get("ARC_MONITOR_CACHE", "recordings/monitor"))

    def __init__(self, action_names: list[str], game_id: str = "") -> None:
        self._actions = list(action_names)
        tag = f"{game_id}-" if game_id else ""
        self._cache_dir = self.CACHE_ROOT / f"{tag}{time.strftime('%Y%m%d-%H%M%S')}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._zs: list[np.ndarray] = []
        self._scores: list[float] = []  # objective score per z (NaN = unknown)
        self._plan: Optional[list] = None
        self._candidates: Optional[np.ndarray] = None
        self._goal: Optional[np.ndarray] = None
        self._step = 0
        self._color_mode = os.environ.get("ARC_PCA_COLOR", "objective")

        plt.ion()
        self.fig = plt.figure("ARC-AGI-3 — arcade monitor", figsize=(11, 8), facecolor=BG)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[5, 2, 1], hspace=0.3, wspace=0.15)
        self.ax_frame = self.fig.add_subplot(gs[0, 0], facecolor=BG)
        self.ax_pca = self.fig.add_subplot(gs[0, 1], facecolor=BG)
        self.ax_plan = self.fig.add_subplot(gs[1, :], facecolor=BG)
        self.ax_btn = self.fig.add_subplot(gs[2, :], facecolor=BG)
        self.fig.canvas.draw()
        plt.show(block=False)

    def update(
        self,
        rgb: np.ndarray,
        z: Optional[np.ndarray],
        action_name: str,
        plan: Optional[list[tuple[np.ndarray, str, np.ndarray]]] = None,
        score: Optional[float] = None,
        candidates: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
        click: Optional[tuple[int, int]] = None,
    ) -> None:
        """`plan` is the decoded CEM rollout: [(imagined rgb, action_name,
        imagined latent ẑ), …], one entry per horizon step; None while the
        world model is warming up. `score` is the latent distance from the
        current state to the goal (colour channel of the PCA panel).
        `candidates` is a (K, latent_dim) sample of the imagined latents the CEM
        considered this solve — the exploration fan. `goal` is the goal latent
        (latent_dim,) the CEM steers toward, drawn as a ★. `click` is the
        (x, y) the action about to execute targets (ACTION6), on the frame."""
        self._step += 1
        self._plan = plan
        self._candidates = candidates
        self._goal = goal
        if z is not None:
            self._zs.append(np.asarray(z, dtype=np.float64).ravel())
            self._scores.append(float(score) if score is not None else np.nan)
        self._draw_frame(rgb, click)
        self._draw_pca()
        self._draw_plan(plan)
        self._draw_buttons(action_name)
        self._record_arcade(action_name)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # let the GUI event loop breathe so the window repaints

    def _record_arcade(self, action_name: str) -> None:
        """Snapshot the whole arcade (all four panels, exactly as displayed)
        for this step."""
        try:
            self.fig.savefig(self._cache_dir / f"step_{self._step:05d}_{action_name}.png",
                             facecolor=BG, dpi=100)
        except Exception as exc:  # a full disk shouldn't kill the run
            print(f"[monitor] arcade recording failed: {exc!r}")

    # ── panels ──────────────────────────────────────────────────────────────

    def _draw_frame(self, rgb: np.ndarray, click: Optional[tuple[int, int]]) -> None:
        ax = self.ax_frame
        ax.clear()
        ax.imshow(rgb, interpolation="nearest")
        title = f"FRAME — step {self._step}"
        if click is not None:
            # Where the action about to execute will click (ACTION6): ring +
            # crosshair on this frame, i.e. the state the agent aimed at.
            x, y = click
            ax.scatter([x], [y], s=200, facecolors="none", edgecolors="#ff2d78",
                       linewidths=2, zorder=3)
            ax.axhline(y, color="#ff2d78", lw=0.6, alpha=0.5)
            ax.axvline(x, color="#ff2d78", lw=0.6, alpha=0.5)
            title += f" — click ({x},{y})"
        ax.set_title(title, color=FG, fontsize=10, family="monospace")
        ax.axis("off")

    def _fit_pca(self) -> tuple[np.ndarray, np.ndarray, float]:
        """The textbook 2-component PCA: centre the latents, SVD, keep the top
        two right-singular vectors. Refit from scratch each frame (a few hundred
        points — cheap). Returns (mean, basis (2, latent_dim), variance
        explained). Signs follow a deterministic convention (largest-magnitude
        loading positive, like sklearn's svd_flip) so the cloud never
        mirror-flips between frames even though SVD signs are arbitrary."""
        X = np.stack(self._zs)
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        basis = vt[:2]
        if basis.shape[0] < 2:  # fewer than 2 components observed so far
            basis = np.vstack([basis, np.zeros((2 - basis.shape[0], basis.shape[1]))])
        signs = np.sign(basis[np.arange(2), np.abs(basis).argmax(axis=1)])
        signs[signs == 0] = 1
        basis = basis * signs[:, None]
        total_var = float((s ** 2).sum())
        evr = float((s[:2] ** 2).sum()) / total_var if total_var > 0 else 0.0
        return mean, basis, evr

    def _point_colors(self, count: int) -> tuple[dict, str]:
        """Scatter kwargs + title tag for the last `count` points: latent
        distance to the goal when available (dark = near the goal), else time
        (ARC_PCA_COLOR=time forces time-based colouring)."""
        scores = np.asarray(self._scores[len(self._scores) - count:])
        finite = scores[np.isfinite(scores)]
        if self._color_mode != "time" and finite.size:
            # Robust range: one far-outlier distance shouldn't wash out the rest.
            vmin, vmax = np.percentile(finite, [5, 95])
            if vmax <= vmin:
                vmax = vmin + 1e-9
            return ({"c": np.where(np.isfinite(scores), scores, vmin),
                     "cmap": "plasma", "vmin": vmin, "vmax": vmax}, "colour=goal-dist")
        return ({"c": np.arange(count), "cmap": "viridis"}, "colour=time")

    def _draw_pca(self) -> None:
        ax = self.ax_pca
        ax.clear()
        ax.set_facecolor(BG)
        if len(self._zs) >= 3:
            mean, basis, evr = self._fit_pca()
            proj = (np.stack(self._zs) - mean) @ basis.T
            recent, old = proj[-self.RECENT:], proj[:-self.RECENT]

            # Deep history: dim context dots, no trajectory line.
            if len(old):
                ax.scatter(old[:, 0], old[:, 1], color=DIM, s=6, alpha=0.4)
            # Recent history: trajectory + coloured points.
            ax.plot(recent[:, 0], recent[:, 1], color=DIM, lw=0.8)
            color_kw, color_tag = self._point_colors(len(recent))
            ax.scatter(recent[:, 0], recent[:, 1], s=16, **color_kw)
            ax.scatter(proj[-1, 0], proj[-1, 1], s=160, facecolors="none",
                       edgecolors="#ff2d78", linewidths=2)

            # The CEM's exploration fan: imagined candidate states this solve.
            if self._candidates is not None and len(self._candidates):
                cand = (np.asarray(self._candidates, dtype=np.float64) - mean) @ basis.T
                ax.scatter(cand[:, 0], cand[:, 1], color="#8888aa", s=4, alpha=0.25)
            # The elite plan ẑ_{t+1..t+H}: dashed pink path from the current state.
            plan_zs = [e[2] for e in (self._plan or []) if len(e) > 2 and e[2] is not None]
            if plan_zs:
                imagined = np.stack([np.asarray(p, dtype=np.float64).ravel() for p in plan_zs])
                plan_proj = (imagined - mean) @ basis.T
                path = np.vstack([proj[-1], plan_proj])
                ax.plot(path[:, 0], path[:, 1], color="#ff2d78", lw=1.2, linestyle="--")
                ax.scatter(plan_proj[:, 0], plan_proj[:, 1], marker="x",
                           c="#ff2d78", s=40, linewidths=1.5)

            # The goal the CEM steers toward: a gold star.
            if self._goal is not None:
                g = (np.asarray(self._goal, dtype=np.float64).ravel()[None, :] - mean) @ basis.T
                ax.scatter(g[0, 0], g[0, 1], marker="*", s=280, c="#ffd400",
                           edgecolors=BG, linewidths=0.8, zorder=5)

            # Each axis autoscales independently (matplotlib default) so the
            # cloud fills the panel instead of collapsing onto the PC1 axis.
            ax.set_title(f"LATENT z — PCA {evr:.0%} var, {len(proj)} pts, "
                         f"{color_tag}, ✕=plan, ★=goal",
                         color=FG, fontsize=10, family="monospace")
            ax.set_xlabel("PC1", color=DIM, fontsize=8, family="monospace")
            ax.set_ylabel("PC2", color=DIM, fontsize=8, family="monospace")
        else:
            ax.set_title(f"LATENT z — PCA ({len(self._zs)} pts)",
                         color=FG, fontsize=10, family="monospace")
            msg = "collecting latents…" if self._zs else "no latent yet\n(world model off → random fallback)"
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                    color=DIM, fontsize=10, family="monospace")
        ax.tick_params(colors=DIM, labelsize=6)
        for spine in ax.spines.values():
            spine.set_color(DIM)

    def _draw_plan(self, plan: Optional[list[tuple[np.ndarray, str]]]) -> None:
        """The agent's imagination: one decoded tile per planned action, left
        to right = t+1 … t+H. Blurry early on — the decoder probe trains live."""
        ax = self.ax_plan
        ax.clear()
        ax.set_title("CEM PLAN — imagined frames (decoder probe)",
                     color=FG, fontsize=10, family="monospace")
        ax.axis("off")
        if not plan:
            ax.text(0.5, 0.5, "no plan yet\n(world model off → random fallback)",
                    transform=ax.transAxes, ha="center", va="center",
                    color=DIM, fontsize=10, family="monospace")
            return

        gap = 6  # px of background between tiles
        tiles = [np.clip(np.asarray(e[0], dtype=np.float64), 0.0, 1.0) for e in plan]
        h, w = tiles[0].shape[:2]
        n = len(tiles)
        canvas = np.full((h, n * w + (n - 1) * gap, 3), mcolors.to_rgb(BG))
        for i, tile in enumerate(tiles):
            x = i * (w + gap)
            canvas[:, x:x + w] = tile
        ax.imshow(canvas, interpolation="nearest")
        for i, entry in enumerate(plan):
            ax.text(i * (w + gap) + w / 2, h + 3, f"t+{i + 1}  {_short(entry[1])}",
                    ha="center", va="top", color=FG, fontsize=9,
                    family="monospace")

    def _draw_buttons(self, pressed: str) -> None:
        ax = self.ax_btn
        ax.clear()
        n = len(self._actions)
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")
        for i, name in enumerate(self._actions):
            color = BUTTON_COLORS[i % len(BUTTON_COLORS)]
            lit = name == pressed
            ax.add_patch(plt.Circle(
                (i + 0.5, 0.55), 0.32,
                facecolor=color if lit else BG,
                edgecolor="white" if lit else color,
                linewidth=2.5 if lit else 1.2,
            ))
            ax.text(i + 0.5, 0.55, _short(name), ha="center", va="center",
                    color="black" if lit else color, fontsize=11,
                    fontweight="bold", family="monospace")
        ax.text(n / 2, -0.35, f"▶ {_caption(pressed)}", ha="center", va="center",
                color=FG, fontsize=9, family="monospace")
