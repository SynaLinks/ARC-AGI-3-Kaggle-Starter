"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

The default body below is a port of the Stochastic Goose / random_agent
sample — a known-good baseline that produces a valid submission and
proves your end-to-end pipeline works. Replace `choose_action` with your
real strategy.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
import os
import random
import time
from typing import Any, Optional

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CategoricalCEMSolver

from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent


# ARC-AGI-3 16-colour palette (cell id → RGB), matching the engine's renderer
# (`key_colors` in agents/templates/reasoning_agent.py). We render frames to RGB
# so an RGB-pretrained world model can consume them directly.
ARC_PALETTE = torch.tensor(
    [
        (255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102),
        (51, 51, 51),    (0, 0, 0),       (229, 58, 163),  (255, 123, 204),
        (249, 60, 49),   (30, 147, 255),  (136, 216, 241), (255, 220, 0),
        (255, 133, 27),  (146, 18, 49),   (79, 204, 48),   (163, 86, 214),
    ],
    dtype=torch.float32,
) / 255.0
NUM_COLORS = ARC_PALETTE.shape[0]  # 16

# Pretrained LeWM (LeWorldModel, JEPA-from-pixels) checkpoint to take the frozen
# encoder from. Accepts a HuggingFace repo id, a local folder, or a `.pt` file
# (see stable_worldmodel.wm.load_pretrained). Set this to your checkpoint; if it
# can't be loaded the agent falls back to random actions.
# Overridable via env so local targets (`make monitor`) can point at the
# checked-in copy under agent/models/ without changing the Kaggle default.
LEWM_CHECKPOINT = os.environ.get("LEWM_CHECKPOINT", "galilai-group/lewm")  # TODO: point at the actual checkpoint

# Pretrained ViT/DINO backbones expect ImageNet-normalised RGB in [0, 1].
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Learned goal generator (k-ahead subgoal). The goal head summarises the last
# GOAL_CONTEXT latents of the current episode and predicts the latent roughly
# GOAL_HORIZON steps ahead — the subgoal the CEM steers toward. Shared by the
# agent (inference) and pretrain.py, whose training target is the human latent
# GOAL_HORIZON steps later along the same trajectory.
GOAL_CONTEXT = 8
GOAL_HORIZON = 5


def frame_data_to_tensor(frame_data: FrameData) -> torch.Tensor:
    """Render an ARC-AGI-3 frame to an RGB tensor of shape ``(1, 3, H, W)``.

    ``frame_data.frame`` is a *stack* of 2D integer grids (``list[list[list[int]]]``);
    we take the most recent grid (the current screen), map each cell id through
    `ARC_PALETTE` to RGB in ``[0, 1]``, and add the leading batch dim that flows
    straight into ``WorldModel.encode``.
    """
    grids = frame_data.frame
    if not grids:
        # No observation yet → an all-background frame so encode() still works.
        grid = torch.zeros(1, 1, dtype=torch.long)
    else:
        grid = torch.tensor(grids[-1], dtype=torch.long)   # (H, W)
    return grid_to_rgb(grid)


def grid_to_rgb(grid: torch.Tensor) -> torch.Tensor:
    """Cell-id grid ``(H, W)`` → RGB tensor ``(1, 3, H, W)`` in [0, 1] through
    the ARC palette. Shared by the live frame path and goal-grid encoding."""
    grid = grid.clamp(0, NUM_COLORS - 1)
    rgb = ARC_PALETTE[grid]                                 # (H, W, 3) in [0, 1]
    return rgb.permute(2, 0, 1).unsqueeze(0)                # (1, 3, H, W)


def frame_data_to_grid(frame_data: FrameData) -> torch.Tensor:
    """The most recent grid as raw cell ids, shape ``(1, H, W)`` long — the
    classification target for the decoder probe."""
    grids = frame_data.frame
    if not grids:
        return torch.zeros(1, 1, 1, dtype=torch.long)
    return torch.tensor(grids[-1], dtype=torch.long).clamp_(0, NUM_COLORS - 1).unsqueeze(0)


def cell_class_weights(targets: torch.Tensor) -> torch.Tensor:
    """Inverse-sqrt-frequency weight per colour class within a batch of cell-id
    targets. A small moving object is a sliver of the cells, so unweighted
    losses let the decoder paint pure background and call it a day — this makes
    rare colours (the objects) matter. Shared by pretrain.py and the online
    decoder step."""
    counts = torch.bincount(targets.flatten(), minlength=NUM_COLORS).float()
    weight = (counts.sum() / counts.clamp_min(1.0)).sqrt()
    return weight / weight.mean()


class GoalHead(nn.Module):
    """Learned goal generator: summarise the recent latent history with a GRU
    and predict a *subgoal* latent ~GOAL_HORIZON steps ahead (as a residual from
    the current latent). Conditioning on the in-episode history — rather than a
    single Markov state — is what lets it specialise to the current game and so
    generalise to *unseen* games, instead of memorising specific trajectories."""

    def __init__(self, latent_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """``context``: ``(B, L, latent_dim)`` latents oldest→newest. Returns the
        predicted subgoal latent ``(B, latent_dim)`` = current latent + Δ."""
        out, _ = self.gru(context)
        return context[:, -1] + self.head(out[:, -1])


class WorldModel(nn.Module):
    """Latent world model driving *goal-directed* planning. Three learned parts:

      - ``encode``: frame tensor → latent ``z`` (also used to encode the goal
        grid into a goal latent).
      - ``predict_next`` (dynamics): ``(z, action) → ẑ'`` predicted next latent.
      - ``decode`` (probe): latent → imagined frame, for the monitor only.

    The CEM steers toward a **goal latent** (`goal`, an encoding of the goal
    grid the agent wants to reach, set by the agent before each solve). The
    cost of a plan is the summed latent distance ``‖ẑ_h − goal‖`` of the states
    it rolls through — so *distance to the goal is the (negative) reward* and
    the solver, which minimises cost, drives the imagined trajectory toward the
    goal. Actions the engine declared unavailable are penalised so no illegal
    plan survives the elite set.

    Implements the stable-worldmodel ``Costable`` protocol via ``get_cost`` so it
    plugs straight into ``CategoricalCEMSolver``.
    """

    # Cost added per rollout step that uses an action the engine declared
    # unavailable — large enough to push any such plan out of the elite set.
    UNAVAILABLE_COST = 1e3

    def __init__(self, n_actions: int = 6, hidden: int = 256,
                 fallback_latent_dim: int = 128) -> None:
        super().__init__()
        self.n_actions = n_actions
        # Goal latent the planner steers toward, shape (1, latent_dim) or
        # (latent_dim,). None ⇒ no goal yet; the agent then acts randomly.
        self.goal: Optional[torch.Tensor] = None
        # When True (monitor runs only), get_cost stashes the imagined latents
        # of its most recent solve in `_candidate_zs` so the monitor can draw
        # the CEM's exploration fan on the PCA panel.
        self.debug_candidates = False
        self._candidate_zs: list[torch.Tensor] = []

        # Encoder: the frozen pretrained LeWM vision encoder (a ViT/DINO-style
        # backbone). We use its CLS token directly — per SSL/JEPA practice the
        # projector only shapes the pretraining loss space and backbone features
        # transfer better downstream, so the checkpoint's projector is dropped.
        # The trainable parts are the dynamics (+ decoder probe).
        self.encoder = None
        latent_dim = fallback_latent_dim
        try:
            import stable_worldmodel as swm

            lewm = swm.wm.load_pretrained(LEWM_CHECKPOINT)
            self.encoder = lewm.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            latent_dim = self._infer_latent_dim()
        except Exception as exc:  # no checkpoint / no transformers / offline
            # encode() will signal not-ready and the agent falls back to random.
            self._encoder_error = exc

        self.latent_dim = latent_dim

        # Dynamics is an MLP over concat([z, action]). nn.Linear acts on the
        # last dim only, so it transparently handles the planner's (B, N, ·)
        # tensors and training's (B, ·) tensors.
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )

        # Pixel decoder: latent z → 16-way colour logits per cell of a 64×64
        # imagined frame. The JEPA latents were never trained for
        # reconstruction, so this is a learned *probe* of the frozen latent
        # space — trained (cross-entropy over the discrete ARC palette, see
        # `cell_class_weights`) on the (z, grid) pairs the agent observes, and
        # used to render the CEM plan. Classification instead of RGB
        # regression: MSE hedges by averaging colours, which erases small
        # moving objects; argmax over logits cannot blur.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32×32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, NUM_COLORS, 4, stride=2, padding=1),  # 64×64
        )

        # Learned goal generator: recent latent history → a k-ahead subgoal
        # latent the CEM steers toward. Warm-started by pretrain.py; until then
        # `has_goal_head` stays False and the agent falls back to acting
        # randomly rather than steering toward an untrained (noise) goal.
        self.goal_head = GoalHead(latent_dim)
        self.has_goal_head = False

        self._load_pretrained_heads()

    def _load_pretrained_heads(self) -> None:
        """Warm-start the trainable parts from `pretrain.py` output (human
        replay dataset) when present: PRETRAINED_HEADS env var, else
        agent/models/pretrained_heads.pt. Silently skipped when absent or
        shape-incompatible — online learning then starts from scratch. Only the
        dynamics and decoder are used now; any other keys a checkpoint carries
        (older objective heads) are ignored."""
        path = os.environ.get("PRETRAINED_HEADS")
        if not path:
            try:
                base = os.path.dirname(os.path.abspath(__file__))
            except NameError:  # spliced into the Kaggle notebook, no __file__
                base = os.path.join(os.getcwd(), "agent")
            path = os.path.join(base, "models", "pretrained_heads.pt")
        if not os.path.exists(path):
            return
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            meta = blob["meta"]
            if meta["latent_dim"] != self.latent_dim or meta["n_actions"] != self.n_actions:
                print(f"[world-model] pretrained heads at {path} don't match "
                      f"(latent {meta['latent_dim']} vs {self.latent_dim}, "
                      f"actions {meta['n_actions']} vs {self.n_actions}), skipped")
                return
            self.dynamics.load_state_dict(blob["dynamics"])
            try:
                self.decoder.load_state_dict(blob["decoder"])
            except RuntimeError:
                # Checkpoint predates the 16-way classification decoder (was
                # RGB regression) — probe starts fresh, dynamics still load.
                print(f"[world-model] decoder in {path} has the old RGB shape, skipped")
            # The goal head is only present in checkpoints trained after it was
            # added; without it the agent can't steer and stays on the random
            # baseline, so `has_goal_head` gates its use.
            if "goal_head" in blob:
                self.goal_head.load_state_dict(blob["goal_head"])
                self.has_goal_head = True
            print(f"[world-model] warm-started dynamics{' + goal head' if self.has_goal_head else ''} "
                  f"from {path} ({meta['transitions']} human transitions, {meta['epochs']} epochs)")
        except Exception as exc:
            print(f"[world-model] failed to load pretrained heads: {exc!r}")

    @torch.no_grad()
    def _infer_latent_dim(self) -> int:
        """Probe the pretrained encoder with a dummy frame to get the latent dim."""
        z = self.encode(torch.zeros(1, 3, 64, 64))
        return z.shape[-1]

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """RGB frame ``(B, 3, H, W)`` in [0, 1] → latent ``(B, latent_dim)``:
        the frozen LeWM ViT backbone's CLS token concatenated with the mean of
        its patch tokens — global (CLS) *and* spatial (patch-mean) structure, so
        the latent is 2× the backbone width and carries the object layout the
        grid games turn on, not just a global summary. `_infer_latent_dim`
        derives `latent_dim` from this, so the whole model resizes with it."""
        if self.encoder is None:
            raise NotImplementedError("pretrained LeWM encoder unavailable")
        frame = (frame - IMAGENET_MEAN.to(frame)) / IMAGENET_STD.to(frame)
        with torch.no_grad():
            hidden = self.encoder(frame, interpolate_pos_encoding=True).last_hidden_state
            cls, patch_mean = hidden[:, 0], hidden[:, 1:].mean(dim=1)
            return torch.cat([cls, patch_mean], dim=-1)  # (B, 2 * encoder_dim)

    def decode_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, latent_dim)`` → per-cell colour logits
        ``(B, NUM_COLORS, 64, 64)`` — the decoder probe's training output."""
        return self.decoder(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, latent_dim)`` → imagined RGB frame ``(B, 3, 64, 64)``
        in [0, 1]: per-cell argmax through the ARC palette. Only as good as
        the trained decoder probe."""
        idx = self.decode_logits(latent).argmax(dim=1)      # (B, 64, 64)
        return ARC_PALETTE.to(latent.device)[idx].permute(0, 3, 1, 2)

    def predict_next(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Dynamics: predict the next latent ``ẑ'``, shape ``(..., latent_dim)``.
        Residual so the model predicts the *change* the action causes."""
        x = torch.cat([latent, action_onehot], dim=-1)
        return latent + self.dynamics(x)

    @staticmethod
    def prediction_error(predicted_latent: torch.Tensor, actual_latent: torch.Tensor) -> torch.Tensor:
        """Latent prediction error between the dynamics' prediction and the
        actual encoding one step later — the training target the dynamics
        minimise. Shape ``(...,)``."""
        return ((predicted_latent - actual_latent) ** 2).mean(dim=-1)

    def propose_goal(self, context: torch.Tensor) -> torch.Tensor:
        """Learned subgoal: recent latent history ``(B, L, latent_dim)`` (oldest
        → newest) → a goal latent ``(B, latent_dim)`` roughly GOAL_HORIZON steps
        ahead, which the CEM then steers toward."""
        return self.goal_head(context)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """stable-worldmodel ``Costable`` hook used by the CEM solver.

        Rolls each candidate action sequence forward in latent space and scores
        it by how close its **final** imagined state lands to the goal latent,
        ``‖ẑ_H − goal‖``. That distance **is** the cost the solver minimises, so
        the elite plans are the ones that reach the subgoal within the horizon —
        distance to goal is the (negative) reward.

        Actions the engine declared unavailable (``info_dict["available"]``,
        a 0/1 mask over the action set) add ``UNAVAILABLE_COST`` for every
        rollout step that uses one, so no such plan survives the elite refit.

        Args:
            info_dict: contains ``"latent"`` of shape ``(B, N, latent_dim)``
                (the solver expands the current latent over the ``N`` candidates)
                and optionally ``"available"`` of shape ``(B, N, n_actions)``.
            action_candidates: one-hot actions, shape ``(B, N, horizon, n_actions)``.

        Returns:
            cost of shape ``(B, N)`` — lower means the plan lands nearer the goal.
        """
        latent = info_dict["latent"]
        available = info_dict.get("available")
        goal = self.goal
        if goal is None:
            raise RuntimeError("WorldModel.goal must be set before planning")
        goal = goal.to(latent).reshape(-1)[: latent.shape[-1]]  # (latent_dim,)
        horizon = action_candidates.shape[2]
        penalty = torch.zeros(latent.shape[:2], device=latent.device, dtype=latent.dtype)
        for h in range(horizon):
            action_onehot = action_candidates[:, :, h, :]
            next_latent = self.predict_next(latent, action_onehot)
            if self.debug_candidates:
                if h == 0:  # new rollout → drop the previous solve's stash
                    self._candidate_zs = []
                self._candidate_zs.append(next_latent.detach())
            if available is not None:
                legal = (action_onehot * available).sum(dim=-1)  # (B, N) ∈ {0, 1}
                penalty = penalty + self.UNAVAILABLE_COST * (1.0 - legal)
            latent = next_latent
        # Distance the plan *ends up* from the subgoal, plus a penalty for any
        # illegal action used along the way. Distance to goal is the reward.
        return (latent - goal).norm(dim=-1) + penalty



class MyAgent(Agent):
    """Goal-directed agent: infers a goal grid, then plans toward it by
    minimising the world model's imagined latent distance to that goal.

    Falls back to uniformly-random actions until the world model is available,
    and while no goal has been inferred yet.
    """

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = 200

    # World model / planning hyper-parameters.
    # (Latent dim is taken from the pretrained LeWM encoder; this is only the
    # fallback used when the encoder can't be loaded.)
    FALLBACK_LATENT_DIM = 128
    PLAN_HORIZON = 5          # CEM rollout length
    NUM_SAMPLES = 256         # candidate action sequences per CEM iteration
    CEM_STEPS = 5             # CEM refinement iterations
    TOPK = 32                 # elites kept per iteration
    # Online learning: one SGD step per action on a minibatch drawn from a
    # sliding replay window, so the dynamics keep improving as the agent plays.
    TRAIN_WINDOW = 256        # transitions kept for replay
    TRAIN_BATCH = 128         # minibatch per SGD step
    TRAIN_MIN = 8             # start training once this many transitions exist
    EPSILON = 0.1             # ε-greedy: probability of a uniform random action,
                              # so off-plan actions keep feeding the dynamics
    # LR for the online dynamics (the WM predictor head) — the only part trained
    # online, so it can be aggressive to adapt the predictor within a game.
    LEARNING_RATE = 3e-3
    # Decoder probe (monitor runs only): one SGD step per action on a replay
    # of observed (z, frame) pairs, so the imagination strip sharpens live.
    DECODER_LR = 1e-3
    RECON_BUFFER = 256        # (z, frame) pairs kept for decoder replay
    RECON_BATCH = 16          # pairs per decoder SGD step

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Seed per game_id so replays from the same game are reproducible but
        # different games explore independently.
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        # Discrete action set the planner ranges over (RESET is handled directly).
        self._candidate_actions = [a for a in GameAction if a is not GameAction.RESET]
        n_actions = len(self._candidate_actions)

        self.world_model = WorldModel(
            n_actions=n_actions, fallback_latent_dim=self.FALLBACK_LATENT_DIM,
        )

        # stable-worldmodel CEM over discrete actions. We drive the solver
        # directly (no Gymnasium env loop); only `gym.spaces.Discrete` is used
        # to describe the action set to `configure`.
        self._solver = CategoricalCEMSolver(
            model=self.world_model,
            num_samples=self.NUM_SAMPLES,
            n_steps=self.CEM_STEPS,
            topk=self.TOPK,
            # Laplace smoothing on the elite refit: without it a hair-thin cost
            # edge collapses the distribution to one-hot on the 1st iteration
            # and the remaining CEM steps can never sample anything else.
            smoothing=0.05,
        )
        self._solver.configure(
            action_space=gym.spaces.Discrete(n_actions),
            n_envs=1,
            config=PlanConfig(horizon=self.PLAN_HORIZON, receding_horizon=1, action_block=1),
        )

        # Online learning state. Buffer rows: (latent_{t-1}, action_onehot_{t-1},
        # latent_t) — the dynamics regress latent_t from (latent_{t-1}, action).
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # (latent_{t-1}, action_onehot_{t-1}) carried to the next step so we can
        # form the transition once the next latent is observed.
        self._prev: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        # This episode's latents, oldest→newest — the goal head's context
        # window. Cleared on every RESET so a goal never spans a reset.
        self._latent_history: list[torch.Tensor] = []

        # Live arcade monitor, local dev only (`make monitor` sets ARC_MONITOR=1).
        # agent/monitor.py is not spliced into the Kaggle notebook, so on Kaggle
        # the env var is unset and this import never runs.
        self._monitor = None
        self._last_z = None     # latent of the latest frame, handed to the monitor
        self._last_plan = None  # decoded CEM rollout [(rgb, action_name, ẑ), …]
        self._last_score = None       # distance from the current state to the goal
        self._last_candidates = None  # (K, D) imagined latents from the last solve
        self._last_goal = None        # goal latent (D,) for the monitor's PCA panel
        # Decoder-probe training state (only fed while the monitor is on).
        self._recon_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._decoder_optimizer: Optional[torch.optim.Optimizer] = None
        if os.environ.get("ARC_MONITOR"):
            try:
                from agent.monitor import ArcadeMonitor

                self._monitor = ArcadeMonitor(
                    [a.name for a in self._candidate_actions] + ["RESET"],
                    game_id=self.game_id,
                )
                self.world_model.debug_candidates = True
            except Exception as exc:
                print(f"[monitor] disabled, ArcadeMonitor failed: {exc!r}")
                self._monitor = None


    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and retry.
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        # First call or after a death → reset the level. No transition spans a
        # reset, so drop the carried-over previous step.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev = None
            self._latent_history = []
            return self._monitored(latest_frame, GameAction.RESET)

        action = self._goal_directed_action(latest_frame)

        if action is None:
            # World model unavailable → random baseline so the agent still
            # returns valid actions and ships — drawn only from the
            # engine-declared available actions. LS20 gets a light ACTION4 bias
            # as an example of a per-game heuristic; `self.game_id` may include
            # a version suffix ("ls20-9607627b"), so match on prefix.
            action = self._random_action(latest_frame)
            action.reasoning = f"random fallback: {action.value}"

        if action.is_complex():
            # ACTION6 takes (x, y) coordinates on a 64×64 grid.
            # TODO: have the planner choose coordinates too; random for now.
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        return self._monitored(latest_frame, action)

    def _random_action(self, latest_frame: FrameData) -> GameAction:
        """A uniformly-random *available* action, with a light per-game bias
        (LS20 favours ACTION4 2×) as an example heuristic. Used as the baseline
        when the world model isn't available."""
        mask = self._available_mask(latest_frame)
        candidates = [a for i, a in enumerate(self._candidate_actions) if mask[0, i] > 0]
        if self.game_id.split("-")[0] == "ls20":
            weights = [2 if a is GameAction.ACTION4 else 1 for a in candidates]
            return random.choices(candidates, weights=weights, k=1)[0]
        return random.choice(candidates)

    def _monitored(self, latest_frame: FrameData, action: GameAction) -> GameAction:
        """Pass-through that reports (frame, latent, action) to the arcade
        monitor when it's on. Each latent is consumed once so the PCA cloud
        only gets points for frames the world model actually encoded."""
        if self._monitor is not None:
            rgb = frame_data_to_tensor(latest_frame)[0].permute(1, 2, 0).numpy()
            click = None
            if action.is_complex():
                data = getattr(action, "action_data", None)
                if data is not None and hasattr(data, "x"):
                    click = (int(data.x), int(data.y))
            self._monitor.update(rgb, self._last_z, action.name, plan=self._last_plan,
                                 score=self._last_score,
                                 candidates=self._last_candidates,
                                 goal=self._last_goal, click=click)
            self._last_z = None
            self._last_plan = None
            self._last_score = None
            self._last_candidates = None
            self._last_goal = None
        return action

    # ── Goal-directed loop ───────────────────────────────────────────────────
    def _goal_directed_action(self, latest_frame: FrameData) -> Optional[GameAction]:
        """Encode the frame, learn from the *previous* transition, propose the
        learned subgoal, then plan the next action by minimising the imagined
        latent distance to that subgoal.

        Returns ``None`` if the world model isn't available, so `choose_action`
        falls back to the random baseline.
        """
        try:
            frame = frame_data_to_tensor(latest_frame)
            latent = self.world_model.encode(frame)  # (1, D)
            self._last_z = latent[0].detach().cpu().numpy()
            if self._monitor is not None:
                self._train_decoder(latent.detach(), frame_data_to_grid(latest_frame))

            # 1) Learn the dynamics from the transition taken at t-1, now that
            #    we observe the actual latent at t.
            if self._prev is not None:
                latent_prev, action_prev = self._prev
                self._buffer.append((latent_prev.detach(), action_prev.detach(), latent.detach()))
                if len(self._buffer) > self.TRAIN_WINDOW:
                    self._buffer.pop(0)
                if len(self._buffer) >= self.TRAIN_MIN:
                    self._train_step()

            # Engine-declared unavailable actions are masked out of the CEM
            # (hard cost in `get_cost`) and out of the random choice below.
            mask = self._available_mask(latest_frame)
            avail_indices = [i for i in range(len(self._candidate_actions)) if mask[0, i] > 0]

            # 2) Propose the learned k-ahead subgoal for the current state,
            #    conditioned on this episode's latent history, and set it as the
            #    goal the CEM steers toward. None until the goal head is trained.
            self.world_model.goal = self._propose_goal(latent)

            # Distance from the current state to the goal — the PCA panel's
            # colour channel (and what the CEM minimises for the next state).
            if self._monitor is not None:
                self._last_score = self._goal_distance(latent)
                self._last_goal = (None if self.world_model.goal is None
                                   else self.world_model.goal.reshape(-1).cpu().numpy())

            # 3) Plan: CEM toward the goal — or, with no goal yet, act randomly
            #    while still collecting dynamics data.
            if self.world_model.goal is not None:
                outputs = self._solver.solve({"latent": latent.detach(), "available": mask})
                plan = [int(i) for i in outputs["actions"][0, :, 0]]  # (n_envs, horizon, action_block)
                action_idx = plan[0]  # receding horizon: execute only the first step

                # ε-greedy: sometimes act off-plan so every action keeps feeding
                # the dynamics. Also the safety net should CEM still argmax an
                # unavailable action.
                explored = random.random() < self.EPSILON
                if explored or action_idx not in avail_indices:
                    action_idx = random.choice(avail_indices)
                if self._monitor is not None:
                    self._last_plan = self._imagine_plan(latent.detach(), plan)
                    self._last_candidates = self._harvest_candidates()
                why = "ε-greedy exploration" if explored else "min distance-to-goal (CEM)"
            else:
                action_idx = random.choice(avail_indices)
                why = "no goal yet: random"

            # 4) Remember (latent, action) so the next step can form the
            #    transition and train the dynamics on it.
            action_onehot = self._one_hot(action_idx)
            self._prev = (latent.detach(), action_onehot)
        except NotImplementedError:
            return None

        action = self._candidate_actions[action_idx]
        action.reasoning = {"why": why, "action_idx": action_idx}
        return action

    def _propose_goal(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        """The learned k-ahead subgoal for the current state — a goal latent
        ``(1, latent_dim)`` from the goal head, conditioned on this episode's
        recent latent history (edge-padded early on). Conditioning on the
        in-episode history is what lets it specialise to the current game,
        unseen ones included, instead of memorising specific trajectories.

        Returns ``None`` until the goal head has been pretrained (`make
        pretrain`), so the agent falls back to acting randomly rather than
        steering toward an untrained (noise) goal.
        """
        self._latent_history.append(latent[0].detach())
        if not self.world_model.has_goal_head:
            return None
        window = self._latent_history[-GOAL_CONTEXT:]
        if len(window) < GOAL_CONTEXT:  # left-pad with the oldest latent seen
            window = [window[0]] * (GOAL_CONTEXT - len(window)) + window
        context = torch.stack(window).unsqueeze(0)  # (1, L, D)
        with torch.no_grad():
            return self.world_model.propose_goal(context).detach()

    def _train_step(self) -> None:
        """One inline SGD step training the latent dynamics to predict the next
        latent from (z, action), on a replay minibatch. The encoder is frozen
        and the decoder probe has its own optimizer, so only `dynamics` moves.
        Runs once per action; the window is kept, not cleared, so each
        transition trains many times."""
        params = [p for p in self.world_model.dynamics.parameters() if p.requires_grad]
        if not params:
            return
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(params, lr=self.LEARNING_RATE)

        batch = random.sample(self._buffer, min(self.TRAIN_BATCH, len(self._buffer)))
        latent_prev = torch.cat([b[0] for b in batch], dim=0)
        action_prev = torch.cat([b[1] for b in batch], dim=0)
        actual = torch.cat([b[2] for b in batch], dim=0)

        predicted = self.world_model.predict_next(latent_prev, action_prev)
        loss = self.world_model.prediction_error(predicted, actual).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    # ── Monitor helpers (monitor runs only, never on Kaggle) ────────────────
    @torch.no_grad()
    def _goal_distance(self, latent: torch.Tensor) -> Optional[float]:
        """Latent distance from the current state to the goal — the colour
        channel of the monitor's PCA panel, mirroring the per-step cost in
        `WorldModel.get_cost`. None when no goal is set yet."""
        goal = self.world_model.goal
        if goal is None:
            return None
        return float((latent.reshape(-1) - goal.reshape(-1)).norm())

    def _harvest_candidates(self, max_points: int = 192) -> Optional[Any]:
        """Subsample the imagined latents `get_cost` stashed during the last
        solve (all horizon steps × all candidates) — the grey exploration fan
        on the monitor's PCA panel."""
        stashed = self.world_model._candidate_zs
        if not stashed:
            return None
        flat = torch.cat([z.reshape(-1, z.shape[-1]) for z in stashed], dim=0)
        if flat.shape[0] > max_points:
            flat = flat[torch.randperm(flat.shape[0])[:max_points]]
        return flat.cpu().numpy()

    def _imagine_plan(self, latent: torch.Tensor, plan: list[int]) -> list[tuple[Any, str, Any]]:
        """Roll the CEM's chosen action sequence through the dynamics and
        decode each imagined latent to pixels: [(rgb H×W×3, action_name, ẑ), …].
        This is the agent's *imagination* — what it expects to see, not what
        the game will actually show. ẑ lets the monitor draw the plan's path
        through latent space on the PCA panel."""
        with torch.no_grad():
            # The rollout is sequential by nature (ẑ_{h+1} needs ẑ_h)…
            z, zs = latent, []
            for idx in plan:
                z = self.world_model.predict_next(z, self._one_hot(idx))
                zs.append(z[0])
            # …but decoding isn't: one (H, D) batch through the decoder.
            stacked = torch.stack(zs)
            rgbs = self.world_model.decode(stacked).permute(0, 2, 3, 1).cpu().numpy()
        return [
            (rgbs[h], self._candidate_actions[idx].name, stacked[h].cpu().numpy())
            for h, idx in enumerate(plan)
        ]

    def _train_decoder(self, latent: torch.Tensor, grid: torch.Tensor) -> None:
        """One cheap SGD step teaching the decoder probe to invert the frozen
        encoder, on a replay of observed (z, cell-id grid) pairs. Called once
        per action, only while the monitor is on. Class-weighted cross-entropy
        over the 16-colour palette so small objects aren't averaged away."""
        target = F.interpolate(grid.unsqueeze(1).float(), size=(64, 64),
                               mode="nearest").squeeze(1).long()
        self._recon_buffer.append((latent, target))
        if len(self._recon_buffer) > self.RECON_BUFFER:
            self._recon_buffer.pop(0)

        if self._decoder_optimizer is None:
            self._decoder_optimizer = torch.optim.Adam(
                self.world_model.decoder.parameters(), lr=self.DECODER_LR
            )
        batch = random.sample(self._recon_buffer, min(self.RECON_BATCH, len(self._recon_buffer)))
        zs = torch.cat([b[0] for b in batch], dim=0)
        targets = torch.cat([b[1] for b in batch], dim=0)
        loss = F.cross_entropy(self.world_model.decode_logits(zs), targets,
                               weight=cell_class_weights(targets))
        self._decoder_optimizer.zero_grad()
        loss.backward()
        self._decoder_optimizer.step()

    def _available_mask(self, latest_frame: FrameData) -> torch.Tensor:
        """0/1 mask ``(1, n_actions)`` over `_candidate_actions` from the
        engine's per-frame ``available_actions`` declaration. All-ones when the
        engine doesn't say (empty list) — never mask on missing information."""
        declared = set(latest_frame.available_actions or [])
        if not declared:
            return torch.ones(1, len(self._candidate_actions))
        mask = torch.tensor(
            [[1.0 if a.value in declared else 0.0 for a in self._candidate_actions]]
        )
        # Defensive: if the declaration excludes everything we can send,
        # ignore it rather than making every plan equally terrible.
        return mask if mask.any() else torch.ones_like(mask)

    def _one_hot(self, action_idx: int) -> torch.Tensor:
        """One-hot encode an action index as ``(1, n_actions)``."""
        onehot = torch.zeros(1, len(self._candidate_actions))
        onehot[0, action_idx] = 1.0
        return onehot
