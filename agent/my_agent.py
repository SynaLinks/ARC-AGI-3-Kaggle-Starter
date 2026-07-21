"""Your ARC-AGI-3 agent. This is the *only* file you should normally edit.

`scripts/build_notebook.py` splices the contents of this file into the
Kaggle submission notebook, so your local dev loop and your Kaggle
submission stay in lock-step:

    [edit my_agent.py] → [make play-local] → [make submit]

Current state: a **goal-conditioned CEM planner** on a world-model base. The
world model (frozen pretrained encoder + online-trained dynamics + decoder probe)
keeps learning as the agent plays; a `CategoricalCEMSolver` (stable-worldmodel)
plans a short action sequence each step and executes the first, choosing the plan
whose *imagined trajectory gets closest to a goal latent* (distance-to-goal in the
standardised latent space). The goal is a settable slot on the world model
(`set_goal`) that a higher-level goal manager will drive; that manager is **not
built yet**, so until a goal is set the agent explores with random available
actions while the dynamics keep learning online. This is the clean goal-conditioned
planner base to add goal *management* on top of.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
import math
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


# ARC-AGI-3 16-colour palette (cell id → RGB), matching the engine's renderer.
# We render frames to RGB so a pretrained world model can consume them directly.
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
# (see stable_worldmodel.wm.load_pretrained). Overridable via env so local
# targets (`make monitor`) can point at the checked-in copy under agent/models/.
LEWM_CHECKPOINT = os.environ.get("LEWM_CHECKPOINT", "galilai-group/lewm")  # TODO: point at the actual checkpoint

# Pretrained ViT/DINO backbones expect ImageNet-normalised RGB in [0, 1].
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ── Behaviour knobs (env-overridable) ─────────────────────────────────────────
#   ARC_DEVICE          — 'cuda' / 'cpu' for the world model (default: cuda if available).
def _env_num(name: str, default: float) -> float:
    try:
        return type(default)(os.environ[name])
    except (KeyError, ValueError):
        return default


def frame_data_to_tensor(frame_data: FrameData) -> torch.Tensor:
    """Render an ARC-AGI-3 frame to an RGB tensor of shape ``(1, 3, H, W)``.

    ``frame_data.frame`` is a *stack* of 2D integer grids; we take the most
    recent grid (the current screen), map each cell id through `ARC_PALETTE` to
    RGB in ``[0, 1]``, and add the leading batch dim that flows into
    ``WorldModel.encode``.
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
    the ARC palette."""
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


class WorldModel(nn.Module):
    """Latent world model. Three parts:

      - ``encode``: frame tensor → latent ``z`` (frozen pretrained encoder);
      - ``predict_next`` (dynamics): ``(z, action) → ẑ'`` predicted next latent
        (trained online);
      - ``decode`` (probe): latent → imagined frame, for the monitor only.

    Implements the stable-worldmodel ``Costable`` protocol via ``get_cost`` so it
    plugs into ``CategoricalCEMSolver``. The cost is a **goal-reaching** objective:
    roll the plan through the dynamics and reward the plan whose imagined trajectory
    comes closest to ``self.goal_latent`` (distance in the standardised latent
    space). The goal is set by ``set_goal`` — a higher-level goal manager will drive
    it; with no goal set, ``get_cost`` is a no-op and the agent explores randomly.
    """

    # Cost added per rollout step using an action the engine declared unavailable
    # — large enough to push any such plan out of the elite set.
    UNAVAILABLE_COST = 1e3

    # Dynamics MLP geometry. A shallow 1×256 predictor on the raw latent
    # collapsed to identity (AUTORESEARCH 2026-07-11); a deeper 4×768 trunk with
    # LayerNorm, trained in the *standardised* latent space, keeps a real
    # action-conditioned signal. LayerNorm lets it train at the base LR.
    DYN_WIDTH = 768
    DYN_DEPTH = 4
    # Coordinate conditioning: complex actions (ACTION6) carry an (x, y) click.
    # The click is Fourier-encoded (COORD_FREQS bands per axis) plus a validity
    # flag and fed to the dynamics alongside the action one-hot, so the model can
    # localise a click's effect. COORD_DIM is the width this adds to the input.
    COORD_FREQS = 6
    COORD_DIM = 1 + 4 * COORD_FREQS  # valid flag + sin/cos × 2 axes × COORD_FREQS

    @staticmethod
    def _build_dynamics(latent_dim: int, n_actions: int, width: int, depth: int,
                        coord_dim: int = 0) -> nn.Module:
        """Residual dynamics trunk: (latent_dim + n_actions + coord_dim) →
        latent_dim, `depth` Linear+LayerNorm+ReLU blocks of `width`. The residual
        `z + trunk(·)` is applied in `predict_next`."""
        layers: list[nn.Module] = []
        c = latent_dim + n_actions + coord_dim
        for _ in range(depth):
            layers += [nn.Linear(c, width), nn.LayerNorm(width), nn.ReLU(inplace=True)]
            c = width
        layers += [nn.Linear(c, latent_dim)]
        return nn.Sequential(*layers)

    def __init__(self, n_actions: int = 6, fallback_latent_dim: int = 128) -> None:
        super().__init__()
        self.n_actions = n_actions
        # Planner objective: **goal-reaching**. The target is a latent stored here
        # and set via ``set_goal``; ``get_cost`` scores plans by how close their
        # imagined trajectory gets to it. ``None`` until a (future) higher-level
        # goal manager assigns one — the planner is then a no-op and the agent
        # explores randomly. Standardised-latent space, so distances are commensurate.
        self.goal_latent: Optional[torch.Tensor] = None

        # Encoder: the frozen pretrained LeWM vision encoder (a ViT/DINO-style
        # backbone). We use its CLS token concatenated with the patch-mean. The
        # trainable parts are the dynamics (+ decoder probe).
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

        # Per-dim standardisation of the frozen encoder's pooled latent, applied
        # as the final step of `encode`. The raw ViT latent has wildly uneven
        # per-dim scale, so an MSE dynamics loss is dominated by a few
        # high-variance (largely action-independent) dims and the predictor
        # collapses toward identity (AUTORESEARCH 2026-07-11). Buffers default to
        # identity and are overwritten by the corpus statistics pretrain.py stores.
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("latent_std", torch.ones(latent_dim))

        # Coordinate conditioning: mark which action indices carry an (x, y) click
        # (the complex actions), and the Fourier bands used to encode it. The
        # ordering matches `_candidate_actions` (agent) and pretrain's id−1 index.
        cand = [a for a in GameAction if a is not GameAction.RESET][:n_actions]
        coord_mask = torch.zeros(n_actions)
        for i, a in enumerate(cand):
            if a.is_complex():
                coord_mask[i] = 1.0
        self.register_buffer("coord_action_mask", coord_mask)
        self.register_buffer("_coord_freqs",
                             (2.0 ** torch.arange(self.COORD_FREQS)) * math.pi)

        # Dynamics: a deep residual MLP over concat([z, action, coord_features])
        # in the standardised latent space.
        self.dynamics = self._build_dynamics(
            latent_dim, n_actions, width=self.DYN_WIDTH, depth=self.DYN_DEPTH,
            coord_dim=self.COORD_DIM,
        )

        # Pixel decoder: latent z → 16-way colour logits per cell of a 64×64
        # imagined frame — a learned probe of the frozen latent space, trained
        # (class-weighted cross-entropy) on the (z, grid) pairs the agent observes
        # and used to render the monitor. Classification not RGB regression: MSE
        # hedges by averaging colours, which erases small moving objects.
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

        self._load_pretrained_heads()

    def _load_pretrained_heads(self) -> None:
        """Warm-start dynamics + decoder (+ standardisation stats) from
        `pretrain.py` output when present: PRETRAINED_HEADS env var, else
        agent/models/pretrained_heads.pt. Silently skipped when absent or
        shape-incompatible — online learning then starts from scratch. Any extra
        keys an older checkpoint carries (goal head, distance head) are ignored."""
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
            if meta.get("latent_dim") != self.latent_dim or meta["n_actions"] != self.n_actions:
                print(f"[world-model] pretrained heads at {path} don't match "
                      f"(latent {meta.get('latent_dim')} vs {self.latent_dim}, "
                      f"actions {meta['n_actions']} vs {self.n_actions}), skipped")
                return
            if meta.get("coord_dim") != self.COORD_DIM:
                print(f"[world-model] pretrained heads at {path} are not "
                      f"coordinate-conditioned (coord_dim {meta.get('coord_dim')} "
                      f"vs {self.COORD_DIM}), skipped — retrain with pretrain.py")
                return
            # Corpus standardisation stats: encode() applies these, so they must
            # load before (and match) the dynamics/decoder trained in that space.
            if "latent_mean" not in blob:
                print(f"[world-model] {path} predates latent standardisation, "
                      f"skipped (retrain with the current pretrain.py)")
                return
            self.latent_mean.copy_(blob["latent_mean"])
            self.latent_std.copy_(blob["latent_std"])
            self.dynamics.load_state_dict(blob["dynamics"])
            try:
                self.decoder.load_state_dict(blob["decoder"])
            except RuntimeError:
                # Checkpoint predates the 16-way classification decoder (was
                # RGB regression) — probe starts fresh, dynamics still load.
                print(f"[world-model] decoder in {path} has the old RGB shape, skipped")
            print(f"[world-model] warm-started dynamics + decoder from {path} "
                  f"({meta['transitions']} transitions, {meta['epochs']} epochs)")
        except Exception as exc:
            print(f"[world-model] failed to load pretrained heads: {exc!r}")

    def set_goal(self, latent: Optional[torch.Tensor]) -> None:
        """Set the target the planner drives toward. ``latent`` is a *standardised*
        latent (an ``encode`` output), shape ``(D,)`` or ``(1, D)`` — stored as
        ``(D,)`` on the model's device. ``None`` clears the goal. The higher-level
        goal manager (not built yet) calls this; until then it stays ``None`` and
        the agent explores randomly."""
        if latent is None:
            self.goal_latent = None
            return
        device = self.latent_mean.device
        self.goal_latent = latent.detach().reshape(-1).to(device)

    def clear_goal(self) -> None:
        """Drop the current goal (planner reverts to the random-exploration base)."""
        self.goal_latent = None

    @torch.no_grad()
    def _infer_latent_dim(self) -> int:
        """Probe the pretrained encoder with a dummy frame to get the latent dim."""
        z = self.encode(torch.zeros(1, 3, 64, 64))
        return z.shape[-1]

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """RGB frame ``(B, 3, H, W)`` in [0, 1] → latent ``(B, latent_dim)``: the
        frozen ViT backbone's CLS token concatenated with the mean of its patch
        tokens, then per-dim standardised."""
        if self.encoder is None:
            raise NotImplementedError("pretrained LeWM encoder unavailable")
        frame = (frame - IMAGENET_MEAN.to(frame)) / IMAGENET_STD.to(frame)
        with torch.no_grad():
            hidden = self.encoder(frame, interpolate_pos_encoding=True).last_hidden_state
            cls, patch_mean = hidden[:, 0], hidden[:, 1:].mean(dim=1)
            z = torch.cat([cls, patch_mean], dim=-1)  # (B, 2 * encoder_dim)
        mean = getattr(self, "latent_mean", None)
        if mean is not None:
            z = (z - mean.to(z)) / self.latent_std.to(z)
        return z

    def decode_logits(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, latent_dim)`` → per-cell colour logits
        ``(B, NUM_COLORS, 64, 64)`` — the decoder probe's training output."""
        return self.decoder(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent → imagined RGB frame ``(B, 3, 64, 64)`` in [0, 1]: per-cell
        argmax through the ARC palette. Only as good as the trained decoder probe."""
        idx = self.decode_logits(latent).argmax(dim=1)      # (B, 64, 64)
        return ARC_PALETTE.to(latent.device)[idx].permute(0, 3, 1, 2)

    def predict_next(self, latent: torch.Tensor, action_onehot: torch.Tensor,
                     coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dynamics: predict the next latent ``ẑ'``, shape ``(..., latent_dim)``.
        Residual so the model predicts the *change* the action causes.

        ``coords`` is the click ``(x, y)`` for a complex action, shape
        ``(..., 2)`` in ``[0, 63]`` (values ``< 0`` mean "no click"). ``None`` →
        a neutral centre click for complex actions, none for the rest, so
        planning and imagination run without a specific coordinate."""
        if coords is None:
            coords = self._default_coords(action_onehot)
        x = torch.cat([latent, action_onehot, self._coord_features(coords)], dim=-1)
        return latent + self.dynamics(x)

    def _default_coords(self, action_onehot: torch.Tensor) -> torch.Tensor:
        """Placeholder click for planning/imagination: centre ``(31, 31)`` for a
        complex action, ``(−1, −1)`` otherwise. Real clicks are passed explicitly
        (online training, the coordinate search)."""
        lead = action_onehot.shape[:-1]
        is_coord = (action_onehot * self.coord_action_mask.to(action_onehot)
                    ).sum(dim=-1, keepdim=True) > 0
        centre = torch.full((*lead, 2), 31.0, device=action_onehot.device)
        none = torch.full((*lead, 2), -1.0, device=action_onehot.device)
        return torch.where(is_coord, centre, none)

    def _coord_features(self, coords: torch.Tensor) -> torch.Tensor:
        """Click ``(..., 2)`` → ``(..., COORD_DIM)``: a validity flag plus Fourier
        features of the normalised coordinate, zeroed where the click is absent
        (any component ``< 0``)."""
        c = coords.float()
        valid = (c >= 0).all(dim=-1, keepdim=True).float()
        xy = c.clamp(min=0.0) / 63.0
        ang = xy.unsqueeze(-1) * self._coord_freqs.to(c)         # (..., 2, F)
        feats = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).flatten(-2)
        return torch.cat([valid, feats * valid], dim=-1)

    @staticmethod
    def prediction_error(predicted_latent: torch.Tensor, actual_latent: torch.Tensor) -> torch.Tensor:
        """Latent prediction error between the dynamics' prediction and the
        actual encoding one step later — the training target the dynamics
        minimise. Shape ``(...,)``."""
        return ((predicted_latent - actual_latent) ** 2).mean(dim=-1)

    @torch.no_grad()
    def _decode_grid(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents to argmax cell-id grids, preserving leading dims:
        ``(..., D) → (..., 64, 64)``."""
        lead = latent.shape[:-1]
        logits = self.decode_logits(latent.reshape(-1, latent.shape[-1]))
        return logits.argmax(dim=1).reshape(*lead, 64, 64)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """stable-worldmodel ``Costable`` hook for the CEM solver — the
        **goal-reaching** objective. Roll each candidate plan forward through the
        dynamics and score it by how close its imagined trajectory gets to
        ``self.goal_latent`` (squared latent distance, in the standardised space).

        We take the **minimum distance over the horizon**, not the frontier only:
        a plan is good if it reaches the goal at *any* point along the rollout, and
        the receding horizon (execute the first action, re-plan next step) means a
        plan that touches the goal early and drifts after is still the right first
        move. Distance is the same per-dim MSE the dynamics train on, so goal-space
        and training-space are commensurate.

        With no goal set (``self.goal_latent is None``) this returns a flat zero
        cost — the agent should be falling back to random exploration and not
        calling the planner, but this stays well-defined if it does.

        Args:
            info_dict: ``"latent"`` of shape ``(B, N, latent_dim)`` (current latent
                expanded over the ``N`` candidates) and optionally ``"available"``
                of shape ``(B, N, n_actions)``.
            action_candidates: one-hot actions ``(B, N, horizon, n_actions)``.

        Returns:
            cost ``(B, N)`` — lower is better. Every step using an
            engine-unavailable action adds ``UNAVAILABLE_COST``.
        """
        latent = info_dict["latent"]
        available = info_dict.get("available")
        horizon = action_candidates.shape[2]
        penalty = torch.zeros(latent.shape[:2], device=latent.device, dtype=latent.dtype)

        if self.goal_latent is None:
            return penalty

        goal = self.goal_latent.to(latent)                    # (D,)
        z = latent                                            # (B, N, D)
        best = torch.full(latent.shape[:2], float("inf"),
                          device=latent.device, dtype=latent.dtype)
        for h in range(horizon):
            action_onehot = action_candidates[:, :, h, :]     # (B, N, A)
            z = self.predict_next(z, action_onehot)           # (B, N, D)
            dist = ((z - goal) ** 2).mean(dim=-1)             # (B, N)
            best = torch.minimum(best, dist)
            if available is not None:
                legal = (action_onehot * available).sum(dim=-1)  # (B, N) ∈ {0, 1}
                penalty = penalty + self.UNAVAILABLE_COST * (1.0 - legal)
        return best + penalty


class MyAgent(Agent):
    """The world model trains online while a goal-conditioned CEM plans each step
    (executing the first action): it picks the action sequence whose imagined
    trajectory gets closest to the world model's goal latent. Until a higher-level
    goal manager sets a goal (`world_model.set_goal`), no goal is set and the agent
    explores with random available actions while the dynamics keep learning."""

    # Upper bound on actions per game; the framework also enforces global limits.
    MAX_ACTIONS = 2000

    # (Latent dim is taken from the pretrained encoder; this is only the fallback
    # used when the encoder can't be loaded.)
    FALLBACK_LATENT_DIM = 128
    # CEM planner (stable-worldmodel CategoricalCEMSolver).
    PLAN_HORIZON = 8          # rollout length
    NUM_SAMPLES = 256         # candidate action sequences per CEM iteration
    CEM_STEPS = 5             # CEM refinement iterations
    TOPK = 32                 # elites kept per iteration
    EPSILON = 0.1             # ε-greedy: take a random available action this often
                              # so planning never fully starves exploration
    # Online learning: one SGD step per action on a minibatch drawn from a
    # sliding replay window, so the dynamics keep improving as the agent plays.
    TRAIN_WINDOW = 256        # transitions kept for replay
    TRAIN_BATCH = 128         # minibatch per SGD step
    TRAIN_MIN = 8             # start training once this many transitions exist
    LEARNING_RATE = 3e-3      # online dynamics LR
    # Decoder probe (monitor runs only): one SGD step per action on a replay of
    # observed (z, frame) pairs, so the imagined frame sharpens live.
    DECODER_LR = 1e-3
    RECON_BUFFER = 256        # (z, frame) pairs kept for decoder replay
    RECON_BATCH = 16          # pairs per decoder SGD step

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Seed per game_id so runs of the same game are reproducible but
        # different games explore independently.
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        # Discrete action set (RESET is handled directly).
        self._candidate_actions = [a for a in GameAction if a is not GameAction.RESET]
        n_actions = len(self._candidate_actions)

        self._device = torch.device(
            os.environ.get("ARC_DEVICE",
                           "cuda" if torch.cuda.is_available() else "cpu"))
        self.world_model = WorldModel(
            n_actions=n_actions, fallback_latent_dim=self.FALLBACK_LATENT_DIM,
        ).to(self._device)

        # stable-worldmodel CEM over discrete actions. We drive the solver
        # directly (no Gymnasium env loop); only `gym.spaces.Discrete` describes
        # the action set to `configure`. get_cost implements the goal-reaching objective.
        self._solver = CategoricalCEMSolver(
            model=self.world_model,
            num_samples=self.NUM_SAMPLES,
            n_steps=self.CEM_STEPS,
            topk=self.TOPK,
            device=self._device,
            # Laplace smoothing on the elite refit: without it a hair-thin cost
            # edge collapses the distribution to one-hot on the 1st iteration.
            smoothing=0.05,
        )
        self._solver.configure(
            action_space=gym.spaces.Discrete(n_actions),
            n_envs=1,
            config=PlanConfig(horizon=self.PLAN_HORIZON, receding_horizon=1, action_block=1),
        )

        # Online learning state. Buffer rows: (grid_{t-1}, latent_{t-1},
        # action_onehot_{t-1}, coords_{t-1}, grid_t, latent_t). The dynamics
        # regress latent_t from (latent_{t-1}, action, click coords).
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._buffer: list[tuple[torch.Tensor, ...]] = []
        # (grid, latent, action_onehot, coords) carried to the next step so we
        # can form the transition once the next frame is observed.
        self._prev: Optional[tuple[torch.Tensor, ...]] = None

        # Coordinate search grid for complex (ACTION6) actions: an 8×8 lattice of
        # click anchors, each scored by one-step goal distance in `_choose_click`.
        anchors = [4 + 8 * k for k in range(8)]
        self._click_candidates = torch.tensor(
            [(x, y) for x in anchors for y in anchors],
            dtype=torch.long, device=self._device,
        )  # (64, 2)

        # Live arcade monitor, local dev only (`make monitor` sets ARC_MONITOR=1).
        # agent/monitor.py is not spliced into the Kaggle notebook.
        self._monitor = None
        self._last_plan = None    # decoded CEM plan rollout [(rgb, action_name), …]
        self._recon_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._decoder_optimizer: Optional[torch.optim.Optimizer] = None
        if os.environ.get("ARC_MONITOR"):
            try:
                from agent.monitor import ArcadeMonitor

                self._monitor = ArcadeMonitor(
                    [a.name for a in self._candidate_actions] + ["RESET"],
                    game_id=self.game_id,
                )
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
        # First call or after a death → reset. No transition spans a reset.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev = None
            return self._monitored(latest_frame, GameAction.RESET)

        action = self._act(latest_frame)

        if action is None:
            # World model unavailable → random baseline so the agent still
            # returns valid actions and ships.
            action = self._random_action(latest_frame)
            action.reasoning = f"random fallback: {action.value}"
            if action.is_complex():
                # No model to choose a click → random (x, y) on the 64×64 grid.
                action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        # For a model-chosen action, `_act` has already set the click coordinate
        # (via the coordinate search), so nothing more to do here.
        return self._monitored(latest_frame, action)

    def _random_action(self, latest_frame: FrameData) -> GameAction:
        """A uniformly-random *available* action, with a light per-game bias
        (LS20 favours ACTION4 2×) as an example heuristic."""
        mask = self._available_mask(latest_frame)
        candidates = [a for i, a in enumerate(self._candidate_actions) if mask[0, i] > 0]
        if self.game_id.split("-")[0] == "ls20":
            weights = [2 if a is GameAction.ACTION4 else 1 for a in candidates]
            return random.choices(candidates, weights=weights, k=1)[0]
        return random.choice(candidates)

    def _monitored(self, latest_frame: FrameData, action: GameAction) -> GameAction:
        """Pass-through that reports (frame, latent, action) to the arcade
        monitor when it's on. The latent is consumed once so the PCA cloud only
        gets points for frames the world model actually encoded."""
        if self._monitor is not None:
            rgb = frame_data_to_tensor(latest_frame)[0].permute(1, 2, 0).numpy()
            click = None
            if action.is_complex():
                data = getattr(action, "action_data", None)
                if data is not None and hasattr(data, "x"):
                    click = (int(data.x), int(data.y))
            self._monitor.update(rgb, action.name, plan=self._last_plan, click=click)
            self._last_plan = None
        return action

    # ── Plan loop ────────────────────────────────────────────────────────────
    def _act(self, latest_frame: FrameData) -> Optional[GameAction]:
        """Encode the frame, learn the dynamics from the previous transition
        (online), then — if a goal is set — plan with the goal-reaching CEM and
        execute its first action (ε-greedy random otherwise). With no goal set,
        explore with a random available action while the dynamics keep learning.

        Returns ``None`` if the world model isn't available, so `choose_action`
        falls back to the random baseline.
        """
        try:
            frame = frame_data_to_tensor(latest_frame).to(self._device)
            grid = frame_data_to_grid(latest_frame).to(self._device)  # (1, 64, 64)
            latent = self.world_model.encode(frame)  # (1, D)
            if self._monitor is not None:
                self._train_decoder(latent.detach(), grid)

            # Learn from the transition taken at t-1, now that we observe frame t.
            if self._prev is not None:
                grid_prev, latent_prev, action_prev, coords_prev = self._prev
                self._buffer.append((grid_prev, latent_prev.detach(),
                                     action_prev.detach(), coords_prev,
                                     grid, latent.detach()))
                if len(self._buffer) > self.TRAIN_WINDOW:
                    self._buffer.pop(0)
                if len(self._buffer) >= self.TRAIN_MIN:
                    self._train_step()

            mask = self._available_mask(latest_frame)
            avail_indices = [i for i in range(len(self._candidate_actions)) if mask[0, i] > 0]

            if self.world_model.goal_latent is None:
                # No goal assigned yet (the higher-level goal manager isn't wired
                # up) → explore with a random available action. The dynamics still
                # learn online, so the world model is ready the moment a goal lands.
                action_idx = random.choice(avail_indices)
                explored, why = True, "random (no goal set)"
                plan = None
            else:
                # Plan with the goal-reaching CEM, executing only the first action;
                # ε-greedy keeps a trickle of pure exploration and is the safety
                # net if the plan's first action is somehow unavailable.
                outputs = self._solver.solve({"latent": latent.detach(), "available": mask.to(self._device)})
                plan = [int(i) for i in outputs["actions"][0, :, 0]]  # (n_envs, horizon, action_block)
                action_idx = plan[0]  # receding horizon: execute only the first step
                explored = random.random() < self.EPSILON or action_idx not in avail_indices
                if explored:
                    action_idx = random.choice(avail_indices)
                why = "ε-greedy" if explored else "goal CEM"

            # Resolve the executed action; for a complex (ACTION6) action, pick
            # the click location — searched toward the goal with the coordinate-
            # conditioned dynamics, or random when exploring / when no goal is set.
            action = self._candidate_actions[action_idx]
            if action.is_complex():
                if explored or self.world_model.goal_latent is None:
                    x, y = random.randint(0, 63), random.randint(0, 63)
                else:
                    x, y = self._choose_click(latent.detach(), action_idx)
                action.set_data({"x": x, "y": y})
                coords_prev = torch.tensor([[x, y]], device=self._device)
            else:
                coords_prev = torch.full((1, 2), -1, device=self._device)

            # Monitor: decode the CEM plan's imagined rollout (t+1 … t+H) — what
            # the world model expects each planned step to look like.
            if self._monitor is not None and plan is not None:
                self._last_plan = self._imagine_plan(latent, plan)

            # Remember (grid, latent, action, coords) so the next step forms the
            # transition once the following frame is observed.
            action_onehot = self._one_hot(action_idx)
            self._prev = (grid, latent.detach(), action_onehot, coords_prev)
        except NotImplementedError:
            return None

        action.reasoning = {"why": why, "action_idx": action_idx}
        return action

    def _train_step(self) -> None:
        """One inline SGD step training the latent dynamics to predict the next
        latent from (z, action), on a replay minibatch. Only `dynamics` moves."""
        params = [p for p in self.world_model.dynamics.parameters() if p.requires_grad]
        if not params:
            return
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(params, lr=self.LEARNING_RATE)

        batch = random.sample(self._buffer, min(self.TRAIN_BATCH, len(self._buffer)))
        latent_prev = torch.cat([b[1] for b in batch], dim=0)
        action_prev = torch.cat([b[2] for b in batch], dim=0)
        coords_prev = torch.cat([b[3] for b in batch], dim=0)
        actual = torch.cat([b[5] for b in batch], dim=0)

        predicted = self.world_model.predict_next(latent_prev, action_prev, coords_prev)
        loss = self.world_model.prediction_error(predicted, actual).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    @torch.no_grad()
    def _choose_click(self, latent: torch.Tensor, action_idx: int) -> tuple[int, int]:
        """Pick the ``(x, y)`` for a complex action by scoring each candidate
        anchor with the coordinate-conditioned dynamics: roll one step per anchor
        and take the click whose imagined next latent lands closest to the goal —
        the one-step, coordinate-space analogue of the planner's goal objective.
        Only called when a goal is set."""
        goal = self.world_model.goal_latent
        cand = self._click_candidates                        # (M, 2)
        m = cand.shape[0]
        z = latent.expand(m, -1)                             # (M, D)
        a = self._one_hot(action_idx).expand(m, -1)          # (M, n_actions)
        z_next = self.world_model.predict_next(z, a, cand)   # (M, D)
        dist = ((z_next - goal.to(z_next)) ** 2).mean(dim=-1)  # (M,)
        best = int(dist.argmin())
        return int(cand[best, 0]), int(cand[best, 1])

    @torch.no_grad()
    def _imagine_plan(self, latent: torch.Tensor, plan: list[int]) -> list[tuple[Any, str]]:
        """Roll the CEM's chosen action sequence through the dynamics and decode
        each imagined latent to pixels: [(rgb 64×64×3, action_name), …], one per
        horizon step. This is the agent's *imagination* — what it expects each
        planned step to look like, not what the game will actually show. Monitor
        only."""
        z, zs = latent, []
        for idx in plan:
            z = self.world_model.predict_next(z, self._one_hot(idx))
            zs.append(z)
        rgbs = self.world_model.decode(torch.cat(zs, dim=0)).permute(0, 2, 3, 1).cpu().numpy()
        return [(rgbs[h], self._candidate_actions[idx].name) for h, idx in enumerate(plan)]

    def _train_decoder(self, latent: torch.Tensor, grid: torch.Tensor) -> None:
        """One cheap SGD step teaching the decoder probe to invert the frozen
        encoder, on a replay of observed (z, cell-id grid) pairs. Called once per
        action, only while the monitor is on."""
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
        """0/1 mask ``(1, n_actions)`` over `_candidate_actions` from the engine's
        per-frame ``available_actions`` declaration. All-ones when the engine
        doesn't say (empty list) — never mask on missing information."""
        declared = set(latest_frame.available_actions or [])
        if not declared:
            return torch.ones(1, len(self._candidate_actions))
        mask = torch.tensor(
            [[1.0 if a.value in declared else 0.0 for a in self._candidate_actions]]
        )
        # Defensive: if the declaration excludes everything we can send, ignore it.
        return mask if mask.any() else torch.ones_like(mask)

    def _one_hot(self, action_idx: int) -> torch.Tensor:
        """One-hot encode an action index as ``(1, n_actions)`` on the model device."""
        onehot = torch.zeros(1, len(self._candidate_actions), device=self._device)
        onehot[0, action_idx] = 1.0
        return onehot