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
    grid = grid.clamp_(0, NUM_COLORS - 1)
    rgb = ARC_PALETTE[grid]                                 # (H, W, 3) in [0, 1]
    return rgb.permute(2, 0, 1).unsqueeze(0)                # (1, 3, H, W)


class WorldModel(nn.Module):
    """Latent world model driving exploration. Four learned parts:

      - ``encode``: frame tensor → latent ``z``.
      - ``predict_next`` (dynamics): ``(z, action) → ẑ'`` predicted next latent.
      - ``predicted_surprise`` (head): ``(z, action) → ŝ``, the *expected future
        surprise* of that transition, **before** the outcome is observed.
      - ``predicted_noop`` (head): ``(z, action) → p̂``, the probability the
        action leaves the environment *unchanged* (walking into a wall, an
        action the game ignores). ARC frames are discrete grids, so the
        training label is exact and free: did the raw frame change?
      - ``predicted_reward`` (head): ``(z, action) → r̂``, the probability the
        action produces *game progress* — a level completion or a WIN. The
        label comes from the engine's ``levels_completed``/``state`` fields,
        so it is exact but extremely sparse; the head is learned offline by
        ``pretrain.py`` (human replays contain the successes) and kept frozen
        online.

    The CEM maximises one of three switchable objectives (``objective``,
    settable via the ARC_OBJECTIVE env var; each rollout step is also gated by
    ``(1 − p̂)`` so predicted no-ops contribute nothing):

      - ``"novelty"`` (default): distance from the imagined latent ẑ′ to the
        nearest *visited* latent this game (`visited`, set by the agent before
        each solve). Big z-movement pays only while it leaves known territory —
        the moment a state is reached it joins the visited set and stops
        paying, so A↔B oscillation and camping die out naturally, and plans
        become multi-step escapes from explored space.
      - ``"displacement"``: predicted ``‖ẑ′ − z‖``. Strong signal with no
        learning required, but never habituates — kept for comparison.
      - ``"surprise"``: the head's expected prediction error, gated by the
        displacement gate ``d/(d + τ)``. Habituates as the dynamics learn
        (`realized_surprise` is the wait-one-step training target), but the
        signal is weak until the dynamics converge.
      - ``"reward"``: the reward head's predicted probability of game progress,
        i.e. plan toward level completions the pretrained head recognises from
        human replays. Exploit-flavoured — where the head sees no reward in
        reach the score is flat and behaviour degrades to ε-greedy.

    Implements the stable-worldmodel ``Costable`` protocol via ``get_cost`` so it
    plugs straight into ``CategoricalCEMSolver`` (cost = −gated objective).
    """

    # Tie-break in the CEM cost: with all surprises at 0 a plan of no-ops and a
    # plan of learned-but-real moves would otherwise score the same.
    NOOP_TIE_BREAK = 1e-3
    # BCE positive-class weight for the no-op head: no-ops are ~5% of human
    # transitions (humans rarely press dead buttons), so unweighted BCE barely
    # beats always answering "it changed". ≈ (1 − 0.05) / 0.05. Shared by
    # pretrain.py and the online loop so fine-tuning continues the same loss.
    NOOP_POS_WEIGHT = 19.0
    # Cost added per rollout step that uses an action the engine declared
    # unavailable — large enough to push any such plan out of the elite set.
    UNAVAILABLE_COST = 1e3

    OBJECTIVES = ("novelty", "displacement", "surprise", "reward")

    def __init__(self, n_actions: int = 6, hidden: int = 256, fallback_latent_dim: int = 128,
                 objective: str = "novelty") -> None:
        super().__init__()
        if objective not in self.OBJECTIVES:
            raise ValueError(f"objective must be one of {self.OBJECTIVES}, got {objective!r}")
        self.n_actions = n_actions
        self.objective = objective
        # Latents visited this game, shape (V, latent_dim) — the reference set
        # for the "novelty" objective. The agent refreshes it before each solve.
        self.visited: Optional[torch.Tensor] = None

        # Encoder: the frozen pretrained LeWM vision encoder (a ViT/DINO-style
        # backbone). We use its CLS token directly — per SSL/JEPA practice the
        # projector only shapes the pretraining loss space and backbone features
        # transfer better downstream, so the checkpoint's projector is dropped.
        # The trainable parts are dynamics + surprise head (+ decoder probe).
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

        # Dynamics and surprise head are MLPs over concat([z, action]). nn.Linear
        # acts on the last dim only, so both transparently handle the planner's
        # (B, N, ·) tensors and training's (B, ·) tensors.
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        self.surprise_head = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # No-op head outputs a *logit* (trained with BCE-with-logits);
        # `predicted_noop` applies the sigmoid for planning.
        self.noop_head = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Reward head also outputs a logit; `predicted_reward` applies the
        # sigmoid. Trained only by pretrain.py (positives — level completions —
        # are far too rare in a single online game to learn from).
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

        # Pixel decoder: latent z → imagined 64×64 RGB frame. The JEPA latents
        # were never trained for reconstruction, so this is a learned *probe* of
        # the frozen latent space — trained online (monitor runs only) on the
        # (z, frame) pairs the agent observes, and used to render the CEM plan.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 32×32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 64×64
            nn.Sigmoid(),
        )

        # Running scale τ of realized latent displacements ‖z' − z‖ for
        # transitions that actually changed the frame — normalizes the
        # displacement gate in `get_cost`. Updated by EMA during training.
        self.register_buffer("disp_scale", torch.tensor(1.0))

        self._load_pretrained_heads()

    def _load_pretrained_heads(self) -> None:
        """Warm-start the trainable parts from `pretrain.py` output (human
        replay dataset) when present: PRETRAINED_HEADS env var, else
        agent/models/pretrained_heads.pt. Silently skipped when absent or
        shape-incompatible — online learning then starts from scratch."""
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
            self.surprise_head.load_state_dict(blob["surprise_head"])
            self.noop_head.load_state_dict(blob["noop_head"])
            self.decoder.load_state_dict(blob["decoder"])
            # Checkpoints written before the reward head existed lack this key;
            # the head then stays at init (≈ constant score, harmless unless
            # the "reward" objective is selected).
            if "reward_head" in blob:
                self.reward_head.load_state_dict(blob["reward_head"])
            if "disp_scale" in blob:
                self.disp_scale.fill_(float(blob["disp_scale"]))
            print(f"[world-model] warm-started heads from {path} "
                  f"({meta['transitions']} human transitions, {meta['epochs']} epochs)")
        except Exception as exc:
            print(f"[world-model] failed to load pretrained heads: {exc!r}")

    @torch.no_grad()
    def _infer_latent_dim(self) -> int:
        """Probe the pretrained encoder with a dummy frame to get the latent dim."""
        z = self.encode(torch.zeros(1, 3, 64, 64))
        return z.shape[-1]

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """RGB frame ``(B, 3, H, W)`` in [0, 1] → latent ``(B, latent_dim)``:
        the CLS token of the frozen pretrained LeWM ViT backbone (pre-projector
        backbone features, the space that transfers best downstream)."""
        if self.encoder is None:
            raise NotImplementedError("pretrained LeWM encoder unavailable")
        frame = (frame - IMAGENET_MEAN.to(frame)) / IMAGENET_STD.to(frame)
        with torch.no_grad():
            output = self.encoder(frame, interpolate_pos_encoding=True)
            return output.last_hidden_state[:, 0]  # (B, encoder_dim)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Latent ``(B, latent_dim)`` → imagined RGB frame ``(B, 3, 64, 64)``
        in [0, 1]. Only as good as the online-trained decoder probe."""
        return self.decoder(latent)

    def predict_next(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Dynamics: predict the next latent ``ẑ'``, shape ``(..., latent_dim)``.
        Residual so the model predicts the *change* the action causes."""
        x = torch.cat([latent, action_onehot], dim=-1)
        return latent + self.dynamics(x)

    def predicted_surprise(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Head: expected *future* surprise ``ŝ`` of the transition, shape
        ``(...,)``. ``softplus`` keeps it non-negative like the realised error."""
        x = torch.cat([latent, action_onehot], dim=-1)
        return F.softplus(self.surprise_head(x).squeeze(-1))

    def noop_logit(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Raw logit that the action changes nothing, shape ``(...,)`` —
        the BCE-with-logits training target of the no-op head."""
        x = torch.cat([latent, action_onehot], dim=-1)
        return self.noop_head(x).squeeze(-1)

    def predicted_noop(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Head: probability ``p̂`` that the action leaves the environment
        unchanged, shape ``(...,)``."""
        return torch.sigmoid(self.noop_logit(latent, action_onehot))

    def reward_logit(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Raw logit that the action produces game progress (level completion
        or WIN), shape ``(...,)`` — the BCE-with-logits training target of the
        reward head in pretrain.py."""
        x = torch.cat([latent, action_onehot], dim=-1)
        return self.reward_head(x).squeeze(-1)

    def predicted_reward(self, latent: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Head: probability ``r̂`` that the action completes a level or wins,
        shape ``(...,)``."""
        return torch.sigmoid(self.reward_logit(latent, action_onehot))

    @staticmethod
    def realized_surprise(predicted_latent: torch.Tensor, actual_latent: torch.Tensor) -> torch.Tensor:
        """Surprise you can only know after waiting one step: the latent
        prediction error between the dynamics' prediction at ``t-1`` and the
        actual encoding at ``t``. Target for both dynamics (minimise) and head
        (predict). Shape ``(...,)``."""
        return ((predicted_latent - actual_latent) ** 2).mean(dim=-1)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """stable-worldmodel ``Costable`` hook used by the CEM solver.

        Rolls each candidate action sequence forward in latent space and sums
        the no-op-gated predicted surprise ``(ŝ + ε)·(1 − p̂_noop)``, returning
        its **negative** so the solver — which minimises cost — maximises the
        expected surprise of transitions that actually change the environment.

        Actions the engine declared unavailable (``info_dict["available"]``,
        a 0/1 mask over the action set) are handled *before* any learning:
        each rollout step using one incurs ``UNAVAILABLE_COST``, so no such
        plan survives the elite refit.

        Args:
            info_dict: contains ``"latent"`` of shape ``(B, N, latent_dim)``
                (the solver expands the current latent over the ``N`` candidates)
                and optionally ``"available"`` of shape ``(B, N, n_actions)``.
            action_candidates: one-hot actions, shape ``(B, N, horizon, n_actions)``.

        Returns:
            cost of shape ``(B, N)`` — lower means more surprising.
        """
        latent = info_dict["latent"]
        available = info_dict.get("available")
        horizon = action_candidates.shape[2]
        total = torch.zeros(latent.shape[:2], device=latent.device, dtype=latent.dtype)
        for h in range(horizon):
            action_onehot = action_candidates[:, :, h, :]
            p_noop = self.predicted_noop(latent, action_onehot)
            next_latent = self.predict_next(latent, action_onehot)
            disp = (next_latent - latent).norm(dim=-1)  # predicted ‖ẑ′ − z‖

            if self.objective == "surprise":
                # Gated by displacement: d/(d+τ) ≈ 0 for predicted no-ops.
                surprise = self.predicted_surprise(latent, action_onehot)
                disp_gate = disp / (disp + self.disp_scale.clamp_min(1e-6))
                score = (surprise + self.NOOP_TIE_BREAK) * disp_gate
            elif self.objective == "reward":
                # Probability this step completes a level / wins, per the
                # pretrained reward head.
                score = self.predicted_reward(latent, action_onehot)
            elif self.objective == "novelty" and self.visited is not None:
                # Distance from the imagined state to the nearest latent ever
                # visited this game: paying only for leaving known territory.
                flat = next_latent.reshape(-1, next_latent.shape[-1])
                visited = self.visited.to(flat)
                score = torch.cdist(flat, visited).min(dim=-1).values.reshape(disp.shape)
            else:  # displacement — or novelty before any state is visited
                score = disp

            step_score = score * (1.0 - p_noop)
            if available is not None:
                legal = (action_onehot * available).sum(dim=-1)  # (B, N) ∈ {0, 1}
                step_score = step_score * legal - self.UNAVAILABLE_COST * (1.0 - legal)
            total = total + step_score
            latent = next_latent
        return -total



class MyAgent(Agent):
    """Curiosity-driven agent: acts to maximise the world model's surprise.

    Falls back to uniformly-random actions until the world model is trained.
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
    # sliding replay window. Training every step (instead of every N) is what
    # lets the no-op gate react within a few actions instead of per-game.
    TRAIN_WINDOW = 256        # transitions kept for replay
    TRAIN_BATCH = 128         # minibatch per SGD step
    TRAIN_MIN = 8             # start training once this many transitions exist
    EPSILON = 0.1             # ε-greedy: probability of a uniform random action,
                              # so off-plan actions keep getting training data
    LEARNING_RATE = 1e-3
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

        # Exploration objective (see WorldModel docstring); ARC_OBJECTIVE lets
        # runs be A/B compared: novelty (default) / displacement / surprise.
        self._objective = os.environ.get("ARC_OBJECTIVE", "novelty")
        self.world_model = WorldModel(
            n_actions=n_actions, fallback_latent_dim=self.FALLBACK_LATENT_DIM,
            objective=self._objective,
        )
        # Latents visited this game — the novelty objective's reference set.
        self._visited_zs: list[torch.Tensor] = []

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

        # Online learning state for the "wait for the surprise" loop.
        # Buffer rows: (latent_{t-1}, action_onehot_{t-1}, latent_t, noop_t)
        # where noop_t = 1.0 iff the raw frame didn't change.
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # (latent_{t-1}, action_onehot_{t-1}, frame_{t-1}) carried to the next
        # step so we can compare the dynamics' prediction against the freshly
        # observed latent, and the raw frame for the exact no-op label.
        self._prev: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

        # Live arcade monitor, local dev only (`make monitor` sets ARC_MONITOR=1).
        # agent/monitor.py is not spliced into the Kaggle notebook, so on Kaggle
        # the env var is unset and this import never runs.
        self._monitor = None
        self._last_z = None     # latent of the latest frame, handed to the monitor
        self._last_plan = None  # decoded CEM rollout [(rgb, action_name), …]
        # Decoder-probe training state (only fed while the monitor is on).
        self._recon_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._decoder_optimizer: Optional[torch.optim.Optimizer] = None
        if os.environ.get("ARC_MONITOR"):
            try:
                from agent.monitor import ArcadeMonitor

                self._monitor = ArcadeMonitor(
                    [a.name for a in self._candidate_actions] + ["RESET"]
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
        # First call or after a death → reset the level. No transition spans a
        # reset, so drop the carried-over previous step.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._prev = None
            return self._monitored(latest_frame, GameAction.RESET)

        action = self._curiosity_action(latest_frame)

        if action is None:
            # World model not implemented/trained yet → random baseline so the
            # agent still returns valid actions and ships — drawn only from the
            # engine-declared available actions. LS20 gets a light ACTION4 bias
            # as an example of a per-game heuristic; `self.game_id` may include
            # a version suffix ("ls20-9607627b"), so match on prefix.
            mask = self._available_mask(latest_frame)
            candidates = [a for i, a in enumerate(self._candidate_actions) if mask[0, i] > 0]
            if self.game_id.split("-")[0] == "ls20":
                weights = [2 if a is GameAction.ACTION4 else 1 for a in candidates]
                action = random.choices(candidates, weights=weights, k=1)[0]
            else:
                action = random.choice(candidates)
            action.reasoning = f"random fallback: {action.value}"

        if action.is_complex():
            # ACTION6 takes (x, y) coordinates on a 64×64 grid.
            # TODO: have the planner choose coordinates too; random for now.
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
        return self._monitored(latest_frame, action)

    def _monitored(self, latest_frame: FrameData, action: GameAction) -> GameAction:
        """Pass-through that reports (frame, latent, action) to the arcade
        monitor when it's on. Each latent is consumed once so the PCA cloud
        only gets points for frames the world model actually encoded."""
        if self._monitor is not None:
            rgb = frame_data_to_tensor(latest_frame)[0].permute(1, 2, 0).numpy()
            self._monitor.update(rgb, self._last_z, action.name, plan=self._last_plan)
            self._last_z = None
            self._last_plan = None
        return action

    # ── Curiosity loop ───────────────────────────────────────────────────────
    def _curiosity_action(self, latest_frame: FrameData) -> Optional[GameAction]:
        """Encode the frame, learn from the *previous* transition's realised
        surprise, then plan the next action by maximising predicted surprise.

        Returns ``None`` if the world model isn't ready, so `choose_action`
        falls back to the random baseline.
        """
        try:
            frame = frame_data_to_tensor(latest_frame)
            latent = self.world_model.encode(frame)  # (1, D)
            self._last_z = latent[0].detach().cpu().numpy()
            if self._monitor is not None:
                self._train_decoder(latent.detach(), frame)

            # 1) Wait for the surprise: now that we observe the actual latent at
            #    t, record the realised target for the transition taken at t-1 —
            #    and the exact no-op label (did the raw frame change at all?).
            if self._prev is not None:
                latent_prev, action_prev, frame_prev = self._prev
                unchanged = frame_prev.shape == frame.shape and torch.equal(frame_prev, frame)
                noop = torch.tensor([1.0 if unchanged else 0.0])
                self._buffer.append((latent_prev.detach(), action_prev.detach(), latent.detach(), noop))
                if len(self._buffer) > self.TRAIN_WINDOW:
                    self._buffer.pop(0)
                if len(self._buffer) >= self.TRAIN_MIN:
                    self._train_step()

            # 2) Plan: CEM maximises the head's predicted future surprise.
            # Engine-declared unavailable actions are masked out of the CEM
            # (hard cost in `get_cost`) and out of exploration below — no
            # learning is spent on what the engine already told us is dead.
            mask = self._available_mask(latest_frame)
            avail_indices = [i for i in range(len(self._candidate_actions)) if mask[0, i] > 0]

            # Novelty reference set: every latent seen this game, including
            # the current one (so standing still scores zero).
            self._visited_zs.append(latent[0].detach())
            self.world_model.visited = torch.stack(self._visited_zs[-4096:])

            outputs = self._solver.solve({"latent": latent.detach(), "available": mask})
            plan = [int(i) for i in outputs["actions"][0, :, 0]]  # (n_envs, horizon, action_block)
            action_idx = plan[0]  # receding horizon: execute only the first step

            # ε-greedy: sometimes act off-plan so every action keeps receiving
            # (surprise, no-op) training data — a pure argmax policy only ever
            # collects data for its current favourite and can never revise it.
            # Also the safety net should CEM still argmax an unavailable action.
            explored = random.random() < self.EPSILON
            if explored or action_idx not in avail_indices:
                action_idx = random.choice(avail_indices)

            if self._monitor is not None:
                self._last_plan = self._imagine_plan(latent.detach(), plan)

            # 3) Remember (latent, action, frame) so the next step can score the
            #    realised surprise and label whether this action was a no-op.
            action_onehot = self._one_hot(action_idx)
            self._prev = (latent.detach(), action_onehot, frame)
        except NotImplementedError:
            return None

        action = self._candidate_actions[action_idx]
        action.reasoning = {
            "why": "ε-greedy exploration" if explored else f"max {self._objective} (CEM)",
            "action_idx": action_idx,
        }
        return action

    def _train_step(self) -> None:
        """One inline SGD step on a replay minibatch: dynamics regresses the
        next latent (minimising surprise), the surprise head regresses the
        realised surprise, and the no-op head classifies whether the action
        changed the frame. Runs once per action; the window is kept, not
        cleared, so each transition trains many times."""
        # Train dynamics + surprise head + no-op head; the pretrained encoder is
        # frozen, the decoder probe has its own optimizer (`_train_decoder`),
        # and the reward head is pretrain-only (a single game has too few
        # level completions to learn from — online BCE would only push it to 0).
        frozen_ids = {id(p) for h in (self.world_model.decoder, self.world_model.reward_head)
                      for p in h.parameters()}
        params = [p for p in self.world_model.parameters()
                  if p.requires_grad and id(p) not in frozen_ids]
        if not params:
            return
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(params, lr=self.LEARNING_RATE)

        batch = random.sample(self._buffer, min(self.TRAIN_BATCH, len(self._buffer)))
        latent_prev = torch.cat([b[0] for b in batch], dim=0)
        action_prev = torch.cat([b[1] for b in batch], dim=0)
        actual = torch.cat([b[2] for b in batch], dim=0)
        noop = torch.cat([b[3] for b in batch], dim=0)  # (B,) 1.0 = frame unchanged

        predicted = self.world_model.predict_next(latent_prev, action_prev)
        dynamics_loss = self.world_model.realized_surprise(predicted, actual).mean()

        target = self.world_model.realized_surprise(predicted.detach(), actual)  # (B,)
        head_loss = ((self.world_model.predicted_surprise(latent_prev, action_prev) - target) ** 2).mean()

        noop_loss = F.binary_cross_entropy_with_logits(
            self.world_model.noop_logit(latent_prev, action_prev), noop,
            pos_weight=torch.tensor(WorldModel.NOOP_POS_WEIGHT),
        )

        # Track the typical realized displacement of frame-changing
        # transitions — normalizes the displacement gate in `get_cost`.
        changed = noop < 0.5
        if changed.any():
            d = (actual[changed] - latent_prev[changed]).norm(dim=-1).mean()
            self.world_model.disp_scale.mul_(0.95).add_(0.05 * d)

        loss = dynamics_loss + head_loss + noop_loss
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    # ── Plan imagination (monitor runs only, never on Kaggle) ───────────────
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

    def _train_decoder(self, latent: torch.Tensor, frame: torch.Tensor) -> None:
        """One cheap SGD step teaching the decoder probe to invert the frozen
        encoder, on a replay of observed (z, frame) pairs. Called once per
        action, only while the monitor is on."""
        target = F.interpolate(frame, size=(64, 64), mode="nearest")
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
        loss = F.mse_loss(self.world_model.decode(zs), targets)
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
