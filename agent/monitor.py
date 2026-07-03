"""Live "arcade cabinet" monitor for local runs. Never ships to Kaggle:
`scripts/build_notebook.py` only splices my_agent.py, and the hook there is
gated behind ARC_MONITOR=1 plus a guarded import.

Run it with:  make monitor GAME=ls20

Four panels, redrawn once per action:
  - the current frame, as the agent's world model sees it (RGB, ARC palette);
  - a 3-D PCA of every world-model latent z observed so far, wrapped in a
    wireframe sphere — drag it with the mouse to rotate (the trajectory
    through latent space, colour = time, ring = current state);
  - the CEM plan decoded to pixels — the imagined frame after each planned
    action, i.e. what the agent *expects* to see, one tile per horizon step;
  - an arcade button row that lights up the action just pressed.
"""
from __future__ import annotations

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


def _great_circle(points: np.ndarray, steps: int = 12) -> np.ndarray:
    """Polyline through unit vectors following the sphere surface: each pair of
    consecutive points is joined by a slerp'd great-circle arc, not a chord."""
    segs = []
    for u, v in zip(points[:-1], points[1:]):
        omega = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
        t = np.linspace(0.0, 1.0, steps)[:, None]
        if omega < 1e-6:  # coincident (or numerically so): chord == arc
            segs.append((1 - t) * u + t * v)
        else:
            segs.append((np.sin((1 - t) * omega) * u + np.sin(t * omega) * v) / np.sin(omega))
    return np.vstack(segs) if segs else points


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
    """One window; call `update(rgb, z, action_name, plan)` once per chosen action."""

    def __init__(self, action_names: list[str]) -> None:
        self._actions = list(action_names)
        self._zs: list[np.ndarray] = []
        self._plan: Optional[list] = None
        self._step = 0

        plt.ion()
        self.fig = plt.figure("ARC-AGI-3 — arcade monitor", figsize=(11, 8), facecolor=BG)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[5, 2, 1], hspace=0.3, wspace=0.15)
        self.ax_frame = self.fig.add_subplot(gs[0, 0], facecolor=BG)
        # 3-D so the latent cloud is a draggable sphere: matplotlib's Axes3D
        # rotates on left-drag out of the box.
        self.ax_pca = self.fig.add_subplot(gs[0, 1], projection="3d", facecolor=BG)
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
    ) -> None:
        """`plan` is the decoded CEM rollout: [(imagined rgb, action_name,
        imagined latent ẑ), …], one entry per horizon step; None while the
        world model is warming up."""
        self._step += 1
        self._plan = plan
        if z is not None:
            self._zs.append(np.asarray(z, dtype=np.float64).ravel())
        self._draw_frame(rgb)
        self._draw_pca()
        self._draw_plan(plan)
        self._draw_buttons(action_name)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # let the GUI event loop breathe so the window repaints

    # ── panels ──────────────────────────────────────────────────────────────

    def _draw_frame(self, rgb: np.ndarray) -> None:
        ax = self.ax_frame
        ax.clear()
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"FRAME — step {self._step}", color=FG, fontsize=10, family="monospace")
        ax.axis("off")

    def _draw_pca(self) -> None:
        ax = self.ax_pca
        # Keep the view the user dragged to: clear() resets elev/azim, so a
        # redraw every action would otherwise snap the sphere back.
        elev, azim = ax.elev, ax.azim
        ax.clear()
        ax.set_facecolor(BG)
        ax.set_title(f"LATENT z — PCA sphere ({len(self._zs)} pts, ✕ = plan) — drag to rotate",
                     color=FG, fontsize=10, family="monospace")
        if len(self._zs) >= 3:
            mean = np.stack(self._zs).mean(axis=0)
            centered = np.stack(self._zs) - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:3]
            if basis.shape[0] < 3:  # fewer than 3 components observed so far
                basis = np.vstack([basis, np.zeros((3 - basis.shape[0], basis.shape[1]))])
            proj = centered @ basis.T
            # Radial projection onto the unit sphere: every point sits ON the
            # surface, so the sphere shows the *direction* of z in PCA space.
            r = 1.0
            proj = proj / np.maximum(np.linalg.norm(proj, axis=1, keepdims=True), 1e-9)

            # Wireframe sphere enclosing the cloud — the thing you grab.
            u = np.linspace(0, 2 * np.pi, 24)
            v = np.linspace(0, np.pi, 13)
            ax.plot_wireframe(r * np.outer(np.cos(u), np.sin(v)),
                              r * np.outer(np.sin(u), np.sin(v)),
                              r * np.outer(np.ones_like(u), np.cos(v)),
                              color=DIM, linewidth=0.4, alpha=0.35)

            arc = _great_circle(proj)
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=DIM, lw=0.8)
            ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                       c=np.arange(len(proj)), cmap="viridis", s=16, depthshade=False)
            ax.scatter(proj[-1, 0], proj[-1, 1], proj[-1, 2], s=160, facecolors="none",
                       edgecolors="#ff2d78", linewidths=2, depthshade=False)
            # Current CEM plan: the imagined latents ẑ_{t+1..t+H}, projected
            # into the same basis — a dashed pink path from the current state.
            plan_zs = [e[2] for e in (self._plan or []) if len(e) > 2 and e[2] is not None]
            if plan_zs:
                imagined = np.stack([np.asarray(p, dtype=np.float64).ravel() for p in plan_zs])
                plan_proj = (imagined - mean) @ basis.T
                plan_proj = plan_proj / np.maximum(
                    np.linalg.norm(plan_proj, axis=1, keepdims=True), 1e-9)
                path = _great_circle(np.vstack([proj[-1], plan_proj]))
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color="#ff2d78", lw=1.2,
                        linestyle="--")
                ax.scatter(plan_proj[:, 0], plan_proj[:, 1], plan_proj[:, 2], marker="x",
                           c="#ff2d78", s=40, linewidths=1.5, depthshade=False)

            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)
            ax.set_zlim(-r, r)
            ax.set_box_aspect((1, 1, 1))
        else:
            msg = "collecting latents…" if self._zs else "no latent yet\n(world model off → random fallback)"
            ax.text2D(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
                      color=DIM, fontsize=10, family="monospace")
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        ax.tick_params(colors=DIM, labelsize=6)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
            axis.line.set_color(DIM)

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
