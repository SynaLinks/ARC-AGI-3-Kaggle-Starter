#!/usr/bin/env python
"""Pretrain the world-model heads from replays.

Two data sources → one artefact:

  data      1. Human replays — the arcprize.org public demo dataset
               (~340 step-by-step recordings over 25 games). Not in git —
               download arc_agi_3_public_demo_human_testing.zip (~106 MB,
               6.4 GB unzipped; some recordings exceed GitHub's file-size
               limit) linked from the ARC-AGI-3 human-performance blog post,
               https://arcprize.org/blog/arc-agi-3-human-dataset, and unzip it
               into dataset/ — it extracts to
               dataset/public_games-dataset/<game>/*.recording.jsonl,
               which is the default --data; pass --data for other layouts.
            2. Random exploration — this script also *plays* each game with
               uniformly-random available actions through the local arc-agi
               engine and records the traces (dataset/.explore/). Humans
               rarely press dead buttons, so their data is thin on no-ops;
               random play hits walls constantly and fills that gap.

  artefact  `agent/models/pretrained_heads.pt` — the latent dynamics, the
            goal head (recent latent history → a k-ahead subgoal latent) and
            the decoder probe. Auto-loaded by `WorldModel` at agent startup so
            the goal-directed planner starts with dynamics that know how each
            action moves the latent and a goal head that proposes where a
            competent player heads next — including on unseen games, via the
            in-context history it conditions on.

Usage:
    make pretrain                          # everything, sensible defaults
    .venv/bin/python pretrain.py --games vc33 --limit 2 --epochs 1
    .venv/bin/python pretrain.py --explore-steps 0   # human replays only

Frame encodings are cached under dataset/.enc_cache/ — the first run pays the
ViT encoding pass, later runs start instantly.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
# Default to the checked-in encoder checkpoint (same one `make monitor` uses).
os.environ.setdefault("LEWM_CHECKPOINT", str(ROOT / "agent" / "models" / "lewm-pusht"))
# Pretraining must start from freshly initialised heads — point the agent's
# auto-warm-start at a path that never exists so it can't load its own output.
os.environ["PRETRAINED_HEADS"] = str(ROOT / ".no-warm-start")

import torch
import torch.nn.functional as F

from arcengine import GameAction

from agent.my_agent import (
    ARC_PALETTE, GOAL_CONTEXT, GOAL_HORIZON, NUM_COLORS, WorldModel,
    cell_class_weights,
)

DATA_DEFAULT = ROOT / "dataset" / "public_games-dataset"
EXPLORE_DIR = ROOT / "dataset" / ".explore"
OUT_DEFAULT = ROOT / "agent" / "models" / "pretrained_heads.pt"
CACHE_DIR = ROOT / "dataset" / ".enc_cache"
N_ACTIONS = 7  # ACTION1..ACTION7; RESET (id 0) is a boundary, never a transition
# GameAction sets `_value_` in a custom __init__, so Enum lookup-by-value
# (GameAction(6)) raises — map action ids explicitly.
ID_TO_ACTION = {a.value: a for a in GameAction}


def _action_id(raw: object) -> int:
    """`action_input.id` appears as int (6) in some recordings and as string
    ('ACTION6', 'RESET') in others. Unknown forms → −1 (transition boundary)."""
    if isinstance(raw, int):
        return raw
    s = str(raw)
    if s == "RESET":
        return 0
    if s.startswith("ACTION") and s[len("ACTION"):].isdigit():
        return int(s[len("ACTION"):])
    return -1


def parse_recording(path: Path) -> dict | None:
    """One .recording.jsonl → aligned per-line arrays/lists.

    Lines whose frame stack is missing or not 64×64 get action id −1 so no
    transition or memory edge is ever built across them.
    """
    grids, actions, levels, wins, states, avail = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            try:
                data = json.loads(line)["data"]
            except (json.JSONDecodeError, KeyError):
                continue
            stack = data.get("frame") or []
            grid = stack[-1] if stack else None
            ok = grid is not None and len(grid) == 64 and len(grid[0]) == 64
            grids.append(
                torch.tensor(grid, dtype=torch.uint8) if ok
                else torch.zeros(64, 64, dtype=torch.uint8)
            )
            actions.append(_action_id(data["action_input"]["id"]) if ok else -1)
            levels.append(int(data.get("levels_completed") or 0))
            wins.append(int(data.get("win_levels") or 0))
            states.append(str(data.get("state") or "NOT_FINISHED"))
            avail.append([int(a) for a in (data.get("available_actions") or [])])
    if len(grids) < 2:
        return None
    return {
        "grids": torch.stack(grids),
        "actions": torch.tensor(actions, dtype=torch.int64),
        "levels": torch.tensor(levels, dtype=torch.int64),
        "wins": torch.tensor(wins, dtype=torch.int64),
        "states": states,
        "avail": avail,
    }


@torch.no_grad()
def encode_recording(model: WorldModel, path: Path, batch: int) -> dict | None:
    """Parse + encode one recording, with an on-disk cache keyed by the file.

    Old-format caches (pre-`levels`) are upgraded in place by re-parsing the
    JSON only — the expensive encoder pass is never repeated.
    """
    cache = CACHE_DIR / path.parent.name / (path.stem + ".pt")
    blob = None
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if blob.get("latent_dim") != model.latent_dim:
            blob = None
    if blob is not None and "levels" in blob:
        return blob

    parsed = parse_recording(path)
    if parsed is None:
        return None
    if blob is not None:  # cached z is valid — just merge in the new fields
        parsed["z"] = blob["z"]
    else:
        zs = []
        grids = parsed["grids"]
        for start in range(0, len(grids), batch):
            chunk = grids[start:start + batch].long().clamp_(0, NUM_COLORS - 1)
            rgb = ARC_PALETTE[chunk].permute(0, 3, 1, 2)  # (B, 3, 64, 64) in [0, 1]
            zs.append(model.encode(rgb))
        parsed["z"] = torch.cat(zs)
    parsed["latent_dim"] = model.latent_dim
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parsed, cache)
    return parsed


def build_transitions(blob: dict) -> tuple[torch.Tensor, ...]:
    """Consecutive-line pairs (i−1 → i) where action_i is a real ACTION1..7.

    RESET (id 0) and malformed lines (id −1) act as boundaries: a RESET
    (new level / retry) must not become a learnable "transition"."""
    actions = blob["actions"]
    idx = torch.arange(1, len(actions))
    valid = (actions[idx] >= 1) & (actions[idx] <= N_ACTIONS) & (actions[idx - 1] != -1)
    idx = idx[valid]
    z_prev, z_next = blob["z"][idx - 1], blob["z"][idx]
    onehot = F.one_hot(actions[idx] - 1, num_classes=N_ACTIONS).float()
    return z_prev, onehot, z_next


def build_goal_samples(blob: dict, context_len: int, horizon: int) -> tuple[torch.Tensor, ...]:
    """Per-episode ``(context → k-ahead subgoal)`` samples for the goal head.

    Episodes are the maximal runs of consecutive lines joined by real actions
    (ACTION1..7); a RESET (id 0) or malformed line (id −1) breaks the run — a
    goal must never span a reset. For each position ``t`` in an episode with a
    full ``horizon`` of future left, we emit:

      - ``context``: the ``context_len`` latents ending at ``t`` (edge-padded
        with the earliest latent early in the episode) — the goal head's input;
      - ``target``: ``z_{t+horizon}`` — where a competent player actually was
        ``horizon`` steps later, i.e. the subgoal to learn to propose;
      - ``weight``: 2.0 if game progress (levels_completed ↑, or WIN) happens in
        the next ``horizon`` steps, else 1.0 — biases the head toward subgoals
        that lead to reward without discarding the rest of the trajectory.

    Returns ``(context (n, L, D), target (n, D), weight (n,))`` — empty tensors
    when the recording is too short to yield any sample.
    """
    actions = blob["actions"]
    z = blob["z"]
    levels = blob["levels"]
    states = blob["states"]
    n = len(actions)
    latent_dim = z.shape[-1]

    # Reward event per line: level count ticked up vs the previous line, or the
    # game reached WIN on this line.
    reward = torch.zeros(n)
    for i in range(1, n):
        if levels[i].item() > levels[i - 1].item() or states[i] == "WIN":
            reward[i] = 1.0

    # Segment into episodes: extend the current run while the action is real,
    # otherwise close it and start a fresh run at the boundary line.
    episodes, run = [], [0]
    for i in range(1, n):
        if 1 <= actions[i].item() <= N_ACTIONS:
            run.append(i)
        else:
            if len(run) >= 2:
                episodes.append(run)
            run = [i]
    if len(run) >= 2:
        episodes.append(run)

    contexts, targets, weights = [], [], []
    for run in episodes:
        a, b = run[0], run[-1] + 1  # contiguous global index range [a, b)
        zep = z[a:b]                # (m, D)
        rep = reward[a:b]           # (m,)
        m = zep.shape[0]
        for t in range(m - horizon):
            lo = max(0, t - context_len + 1)
            win = zep[lo:t + 1]
            if win.shape[0] < context_len:  # left-pad with the earliest latent
                pad = win[:1].expand(context_len - win.shape[0], -1)
                win = torch.cat([pad, win], dim=0)
            contexts.append(win)
            targets.append(zep[t + horizon])
            weights.append(2.0 if bool(rep[t + 1:t + horizon + 1].any()) else 1.0)

    if not contexts:
        return (torch.empty(0, context_len, latent_dim),
                torch.empty(0, latent_dim), torch.empty(0))
    return torch.stack(contexts), torch.stack(targets), torch.tensor(weights)


# ── Random exploration ────────────────────────────────────────────────────────

def _frame_line(fd, action_id: int) -> str:
    """Serialize an engine FrameDataRaw as one human-dataset-format line."""
    data = {
        "game_id": fd.game_id,
        "frame": [g.tolist() for g in fd.frame],
        "state": fd.state.value if hasattr(fd.state, "value") else str(fd.state),
        "levels_completed": fd.levels_completed,
        "win_levels": fd.win_levels,
        "action_input": {"id": action_id, "data": {"game_id": fd.game_id}, "reasoning": None},
        "guid": fd.guid or "",
        "full_reset": bool(fd.full_reset),
        "available_actions": list(fd.available_actions),
    }
    return json.dumps({"timestamp": "", "data": data})


def explore_games(games: list[str], steps: int, episodes: int) -> list[Path]:
    """Ensure `episodes` random-walk traces of `steps` actions exist per game
    under dataset/.explore/<game>/, generating the missing ones with the local
    arc-agi engine. Deterministic per (game, episode) seed. Returns all paths."""
    paths, missing = [], []
    for game in games:
        for ep in range(episodes):
            p = EXPLORE_DIR / game / f"random-{ep:02d}-{steps}.recording.jsonl"
            paths.append(p)
            if not p.exists():
                missing.append((game, ep, p))
    if not missing:
        return paths

    import arc_agi
    from arc_agi import OperationMode

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    for game, ep, path in missing:
        env = arc.make(game)
        if env is None:
            print(f"  explore: could not create env for {game!r}, skipping")
            paths.remove(path)
            continue
        rng = random.Random(f"{game}-{ep}-{steps}")
        lines = []
        fd = env.reset()
        if fd is None:
            paths.remove(path)
            continue
        lines.append(_frame_line(fd, 0))
        noops = 0
        for _ in range(steps):
            state = fd.state.name if hasattr(fd.state, "name") else str(fd.state)
            if state in ("GAME_OVER", "WIN"):
                fd = env.reset()
                if fd is None:
                    break
                lines.append(_frame_line(fd, 0))
                continue
            candidates = [a for a in fd.available_actions if 1 <= a <= N_ACTIONS] or [1, 2, 3, 4]
            aid = rng.choice(candidates)
            action = ID_TO_ACTION[aid]
            data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)} if action.is_complex() else None
            nxt = env.step(action, data=data)
            if nxt is None:
                break
            if nxt.frame and fd.frame and len(nxt.frame[-1]) == len(fd.frame[-1]):
                import numpy as _np
                noops += int(_np.array_equal(nxt.frame[-1], fd.frame[-1]))
            fd = nxt
            lines.append(_frame_line(fd, aid))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print(f"  explore: {game} ep{ep} — {len(lines)} frames, {noops} no-ops")
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DEFAULT, help="public_games-dataset folder")
    p.add_argument("--games", default="all", help="comma-separated game ids, or 'all'")
    p.add_argument("--limit", type=int, default=0, help="max recordings (0 = no limit), for smoke tests")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256, help="training minibatch")
    p.add_argument("--encode-batch", type=int, default=128, help="frames per encoder forward")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dyn-lr-mult", type=float, default=3.0,
                   help="LR multiplier for the dynamics (WM predictor) head")
    p.add_argument("--explore-steps", type=int, default=400,
                   help="random-walk actions per exploration episode (0 = no exploration)")
    p.add_argument("--explore-episodes", type=int, default=2, help="exploration episodes per game")
    p.add_argument("--no-decoder", action="store_true", help="skip the decoder probe")
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    if not args.data.is_dir():
        raise SystemExit(
            f"Dataset not found at {args.data} — download "
            "arc_agi_3_public_demo_human_testing.zip (see "
            "https://arcprize.org/blog/arc-agi-3-human-dataset) and unzip it "
            "into dataset/."
        )

    games = sorted(d.name for d in args.data.iterdir()
                   if d.is_dir() and not d.name.startswith("_"))
    if args.games != "all":
        wanted = {g.strip() for g in args.games.split(",")}
        unknown = wanted - set(games)
        if unknown:
            raise SystemExit(f"Unknown game(s) {sorted(unknown)}; have {games}")
        games = sorted(wanted)

    model = WorldModel(n_actions=N_ACTIONS)
    if model.encoder is None:
        raise SystemExit(f"LeWM encoder failed to load: {getattr(model, '_encoder_error', '?')}")

    # ── Gather traces: human replays + random exploration ───────────────────
    # Each recording is tagged human/explore. Dynamics + decoder learn from
    # both (exploration adds wall-hit / dead-button coverage humans skip), but
    # the *goal head* trains on human replays only — it must learn where a
    # competent player heads next, not where a random walk drifts.
    recordings = [(f, True) for g in games
                  for f in sorted((args.data / g).glob("*.recording.jsonl"))]
    if args.explore_steps > 0:
        print(f"exploring: {args.explore_episodes} × {args.explore_steps} random actions per game")
        recordings += [(f, False) for f in
                       explore_games(games, args.explore_steps, args.explore_episodes)]
    if args.limit:
        recordings = recordings[: args.limit]
    print(f"{len(recordings)} recordings across {len(games)} games")

    torch.manual_seed(0)  # reproducible shuffles across pretraining runs
    zp, oh, zn, z_all, grids_all = [], [], [], [], []
    gc, gt, gw = [], [], []  # goal-head samples (human only): context, target, weight
    t0 = time.time()
    for i, (path, is_human) in enumerate(recordings, 1):
        blob = encode_recording(model, path, args.encode_batch)
        if blob is None:
            continue
        a, b, c = build_transitions(blob)
        zp.append(a); oh.append(b); zn.append(c)
        z_all.append(blob["z"]); grids_all.append(blob["grids"])
        n_goal = 0
        if is_human:
            c_ctx, c_tgt, c_w = build_goal_samples(blob, GOAL_CONTEXT, GOAL_HORIZON)
            gc.append(c_ctx); gt.append(c_tgt); gw.append(c_w)
            n_goal = len(c_ctx)
        print(f"  [{i}/{len(recordings)}] {path.parent.name}/{path.name}: "
              f"{len(blob['z'])} frames, {len(a)} transitions, {n_goal} goal "
              f"samples ({time.time() - t0:.0f}s)")

    zp, oh, zn = map(torch.cat, (zp, oh, zn))
    z_all, grids_all = torch.cat(z_all), torch.cat(grids_all)
    gc = torch.cat(gc) if gc else torch.empty(0)
    if len(gc):
        gt, gw = torch.cat(gt), torch.cat(gw)
    train_goal = len(gc) > 0
    if not train_goal:
        print("WARNING: no goal samples (recordings shorter than the goal "
              f"horizon of {GOAL_HORIZON}); goal head will not be trained/saved.")
    reward_frac = f"{float((gw > 1).float().mean()):.1%} reward-leading" if train_goal else "—"
    print(f"\n{len(zp)} transitions, {len(z_all)} frames, {len(gc)} goal samples "
          f"({reward_frac}), latent dim {model.latent_dim}")

    # ── Train the parts the agent uses: dynamics, goal head (+ decoder) ──────
    # The dynamics (the WM predictor) gets a higher LR than the other heads —
    # it's the part planning quality hinges on, and it must chase a moving
    # target (its own residual latents), so it benefits from adapting faster.
    other = ([model.goal_head] if train_goal else []) \
        + ([model.decoder] if not args.no_decoder else [])
    groups = [{"params": list(model.dynamics.parameters()), "lr": args.lr * args.dyn_lr_mult},
              {"params": [p for h in other for p in h.parameters()], "lr": args.lr}]
    optimizer = torch.optim.Adam([g for g in groups if g["params"]])
    print(f"dynamics lr {args.lr * args.dyn_lr_mult:g}, other heads lr {args.lr:g}")

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(zp))
        sums, seen = torch.zeros(3), 0
        for start in range(0, len(order), args.batch):
            sel = order[start:start + args.batch]
            b_zp, b_oh, b_zn = zp[sel], oh[sel], zn[sel]

            predicted = model.predict_next(b_zp, b_oh)
            dyn_loss = model.prediction_error(predicted, b_zn).mean()

            goal_loss = torch.tensor(0.0)
            if train_goal:
                # Goal samples are indexed independently of transitions (there
                # are fewer of them near episode ends) — draw a matched batch.
                gsel = torch.randint(0, len(gc), (len(sel),))
                pred_goal = model.propose_goal(gc[gsel])
                per = ((pred_goal - gt[gsel]) ** 2).mean(dim=-1)  # (B,)
                goal_loss = (gw[gsel] * per).mean()

            dec_loss = torch.tensor(0.0)
            if not args.no_decoder:
                # Decoder pairs are independent of transitions — sample frames.
                # Class-weighted cross-entropy over cell colours (not RGB MSE):
                # regression averages small moving objects into the background.
                fsel = torch.randint(0, len(z_all), (len(sel),))
                target = grids_all[fsel].long().clamp_(0, NUM_COLORS - 1)
                dec_loss = F.cross_entropy(model.decode_logits(z_all[fsel]), target,
                                           weight=cell_class_weights(target))

            loss = dyn_loss + goal_loss + dec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(sel)
            sums += n * torch.tensor([dyn_loss.item(), goal_loss.item(), dec_loss.item()])
            seen += n
        d, g, dc = (sums / seen).tolist()
        print(f"epoch {epoch:2}/{args.epochs}  dynamics {d:.4f}  "
              f"goal {g:.4f}  decoder {dc:.4f}")

    # ── Save in the shape WorldModel auto-loads at agent startup ────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "meta": {
            "latent_dim": model.latent_dim,
            "n_actions": N_ACTIONS,
            "lewm_checkpoint": os.environ["LEWM_CHECKPOINT"],
            "transitions": len(zp),
            "frames": len(z_all),
            "goal_samples": len(gc),
            "goal_context": GOAL_CONTEXT,
            "goal_horizon": GOAL_HORIZON,
            "epochs": args.epochs,
            "games": games,
        },
        "dynamics": model.dynamics.state_dict(),
        "decoder": model.decoder.state_dict(),
    }
    if train_goal:
        blob["goal_head"] = model.goal_head.state_dict()
    torch.save(blob, args.out)
    print(f"\nsaved pretrained dynamics{' + goal head' if train_goal else ''} "
          f"+ decoder → {args.out}")


if __name__ == "__main__":
    main()
