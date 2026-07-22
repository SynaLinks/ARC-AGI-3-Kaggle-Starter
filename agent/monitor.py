"""Live "arcade cabinet" monitor for local runs. Never ships to Kaggle:
`scripts/build_notebook.py` only splices my_agent.py, and the hook there is
gated behind ARC_MONITOR=1 plus a guarded import.

Run it with:  make monitor GAME=ls20

Three panels, redrawn once per action:
  - FRAME: the current frame the world model sees (RGB, ARC palette), with a
    ring + crosshair wherever the action about to run clicks (ACTION6);
  - IMAGINED PLAN: the decoded CEM rollout — the world model's imagined frame
    after each planned step (t+1 … t+H). The first tile is next step's reality
    (roughly), so you can eyeball prediction quality live;
  - an arcade button row that lights up the action just pressed.

Each redraw is also recorded as a PNG under
recordings/monitor/<game>-<timestamp>/ (override with ARC_MONITOR_CACHE), so a
run can be reviewed after the window closes.
"""
from __future__ import annotations

import importlib
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

# uv-managed CPythons ship a _tkinter matplotlib's TkAgg can't load, so probe
# each backend by importing its module and fall back: TkAgg → QtAgg (PySide6).
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
    """One window; call `update(rgb, action_name, prediction=…)` once per action."""

    # Every redraw is recorded as a PNG under <ARC_MONITOR_CACHE or
    # recordings/monitor>/<game>-<timestamp>/ so a run can be reviewed later.
    CACHE_ROOT = Path(os.environ.get("ARC_MONITOR_CACHE", "recordings/monitor"))

    def __init__(self, action_names: list[str], game_id: str = "") -> None:
        self._actions = list(action_names)
        tag = f"{game_id}-" if game_id else ""
        self._cache_dir = self.CACHE_ROOT / f"{tag}{time.strftime('%Y%m%d-%H%M%S')}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0

        plt.ion()
        self.fig = plt.figure("ARC-AGI-3 — arcade monitor", figsize=(11, 8), facecolor=BG)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[5, 3, 1], hspace=0.3)
        self.ax_frame = self.fig.add_subplot(gs[0], facecolor=BG)
        self.ax_plan = self.fig.add_subplot(gs[1], facecolor=BG)
        self.ax_btn = self.fig.add_subplot(gs[2], facecolor=BG)
        self.fig.canvas.draw()
        plt.show(block=False)

    def update(
        self,
        rgb: np.ndarray,
        action_name: str,
        plan: Optional[list] = None,
        click: Optional[tuple[int, int]] = None,
    ) -> None:
        """`rgb` is the current frame (H, W, 3) in [0, 1]. `plan` is the decoded
        CEM rollout ``[(imagined rgb, action_name), …]`` (one entry per horizon
        step), or None while the world model is warming up. `click` is the (x, y)
        an ACTION6 targets, drawn on the current frame."""
        self._step += 1
        self._draw_frame(rgb, click)
        self._draw_plan(plan)
        self._draw_buttons(action_name)
        self._record_arcade(action_name)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # let the GUI event loop breathe so the window repaints

    def _record_arcade(self, action_name: str) -> None:
        """Snapshot the whole arcade (all panels) for this step."""
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
            x, y = click  # where the action about to execute clicks (ACTION6)
            ax.scatter([x], [y], s=200, facecolors="none", edgecolors="#ff2d78",
                       linewidths=2, zorder=3)
            ax.axhline(y, color="#ff2d78", lw=0.6, alpha=0.5)
            ax.axvline(x, color="#ff2d78", lw=0.6, alpha=0.5)
            title += f" — click ({x},{y})"
        ax.set_title(title, color=FG, fontsize=10, family="monospace")
        ax.axis("off")

    def _draw_plan(self, plan: Optional[list]) -> None:
        """The decoded CEM plan: one imagined frame per planned step, left→right
        = t+1 … t+H, with the action for each step under it. This is the agent's
        imagination — what it expects, not what the game will show."""
        ax = self.ax_plan
        ax.clear()
        ax.axis("off")
        ax.set_title("IMAGINED PLAN — decoded CEM rollout (t+1 … t+H)",
                     color=FG, fontsize=10, family="monospace")
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
                    ha="center", va="top", color=FG, fontsize=9, family="monospace")

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