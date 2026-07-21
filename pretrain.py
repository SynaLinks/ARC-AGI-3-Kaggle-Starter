#!/usr/bin/env python
"""Pretrain the world model from random exploration (pooled `CLS ⊕ patch-mean` latent).

Trains only the pieces the agent needs: the latent **dynamics** (the
action-conditioned next-latent predictor) and the **decoder**.
So the planner needs the dynamics (rollout) and the decoder (novelty cost), and
nothing goal-related is learned offline.

Data source → artefact:

  data      Random exploration ONLY. This script *plays* each game with
            uniformly-random available actions through the local arc-agi engine
            and records the traces (dataset/.explore/). Human replays are
            deliberately NOT used: they are purposeful, near-solution play, so
            training the heads on them leaks task-specific structure and inflates
            apparent quality on the training games — "cheating" against the goal
            of generalising to unseen games. Random play is game-agnostic, the
            honest signal for a transferable world model. (The dataset folder,
            default dataset/public_games-dataset/, is still read — but only to
            enumerate which game ids to explore; pass --data for other layouts.)

  artefact  `agent/models/pretrained_heads.pt` — the latent dynamics, the
            decoder probe, and the per-dim latent standardisation stats. The
            raw pooled ViT latent has wildly uneven per-dim scale, so an MSE
            dynamics loss collapses the predictor to identity (AUTORESEARCH
            2026-07-11); standardising balances the loss and keeps a real
            action signal. Auto-loaded by `WorldModel` at agent startup.

There is **no encoder fine-tuning** here: offline FT overfits the training games
and hurts unseen ones (AUTORESEARCH 2026-07-12/18), so the encoder stays frozen
and only the heads are trained.

Usage:
    make pretrain                          # frozen-encoder base pretrain
    .venv/bin/python pretrain.py --games vc33 --limit 2 --epochs 1
    .venv/bin/python pretrain.py --explore-episodes 10   # more random data

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
    ARC_PALETTE, NUM_COLORS, WorldModel, cell_class_weights,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    xs, ys = [], []
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
            # Click coordinate of a complex action (ACTION6), else −1/−1. Older
            # traces that predate coordinate recording simply carry no x/y here.
            ad = (data.get("action_input") or {}).get("data") or {}
            has_xy = ok and isinstance(ad, dict) and "x" in ad and "y" in ad
            xs.append(int(ad["x"]) if has_xy else -1)
            ys.append(int(ad["y"]) if has_xy else -1)
    if len(grids) < 2:
        return None
    return {
        "grids": torch.stack(grids),
        "actions": torch.tensor(actions, dtype=torch.int64),
        "levels": torch.tensor(levels, dtype=torch.int64),
        "wins": torch.tensor(wins, dtype=torch.int64),
        "states": states,
        "avail": avail,
        "xs": torch.tensor(xs, dtype=torch.int64),
        "ys": torch.tensor(ys, dtype=torch.int64),
    }


@torch.no_grad()
def encode_recording(model: WorldModel, path: Path, batch: int) -> dict | None:
    """Parse + encode one recording, with an on-disk cache keyed by the file.

    Old-format caches (pre-`levels`, pre-`xs`) are upgraded in place by
    re-parsing the JSON only — the expensive encoder pass is never repeated.
    Cached latents are the *raw* pooled ViT output (pre-standardisation); the
    corpus stats are computed and applied after the whole corpus is gathered.
    """
    cache = CACHE_DIR / path.parent.name / (path.stem + ".pt")
    blob = None
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if blob.get("latent_dim") != model.latent_dim:
            blob = None
    if blob is not None and "levels" in blob and "xs" in blob:
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
            chunk = grids[start:start + batch].long().clamp_(0, NUM_COLORS - 1).to(DEVICE)
            rgb = ARC_PALETTE.to(DEVICE)[chunk].permute(0, 3, 1, 2)  # (B, 3, 64, 64)
            zs.append(model.encode(rgb).cpu())
        parsed["z"] = torch.cat(zs)
    parsed["latent_dim"] = model.latent_dim
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parsed, cache)
    return parsed


def build_transitions(blob: dict) -> tuple[torch.Tensor, ...]:
    """Consecutive-line pairs (i−1 → i) where action_i is a real ACTION1..7,
    returned as gathered latents (z_prev, action_onehot, z_next, coords).

    ``coords`` is the ``(x, y)`` of the action taken at step i, shape ``(T, 2)``
    — real values for ACTION6, ``(−1, −1)`` for the coordinate-free actions (and
    for older traces that predate coordinate recording). Not consumed by the
    current heads; carried so coordinate-conditioned dynamics can train on it.

    RESET (id 0) and malformed lines (id −1) act as boundaries: a RESET
    (new level / retry) must not become a learnable "transition"."""
    actions = blob["actions"]
    idx = torch.arange(1, len(actions))
    valid = (actions[idx] >= 1) & (actions[idx] <= N_ACTIONS) & (actions[idx - 1] != -1)
    idx = idx[valid]
    z_prev, z_next = blob["z"][idx - 1], blob["z"][idx]
    onehot = F.one_hot(actions[idx] - 1, num_classes=N_ACTIONS).float()
    # xs/ys are absent from caches that predate coordinate parsing → all −1.
    n = len(blob["actions"])
    xs = blob.get("xs", torch.full((n,), -1, dtype=torch.int64))
    ys = blob.get("ys", torch.full((n,), -1, dtype=torch.int64))
    coords = torch.stack([xs[idx], ys[idx]], dim=-1)
    return z_prev, onehot, z_next, coords


# ── Random exploration ────────────────────────────────────────────────────────

def _frame_line(fd, action_id: int, action_data: dict | None = None) -> str:
    """Serialize an engine FrameDataRaw as one human-dataset-format line.

    ``action_data`` is the ``(x, y)`` payload of a complex action (ACTION6); it
    is recorded into ``action_input.data`` so coordinate-conditioned dynamics can
    be trained later. Simple actions and RESET pass ``None`` (no coordinate)."""
    ai_data = {"game_id": fd.game_id}
    if action_data and "x" in action_data and "y" in action_data:
        ai_data["x"] = int(action_data["x"])
        ai_data["y"] = int(action_data["y"])
    data = {
        "game_id": fd.game_id,
        "frame": [g.tolist() for g in fd.frame],
        "state": fd.state.value if hasattr(fd.state, "value") else str(fd.state),
        "levels_completed": fd.levels_completed,
        "win_levels": fd.win_levels,
        "action_input": {"id": action_id, "data": ai_data, "reasoning": None},
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
            lines.append(_frame_line(fd, aid, data))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        print(f"  explore: {game} ep{ep} — {len(lines)} frames, {noops} no-ops")
    return paths


def gather_explore(games: list[str], steps: int, episodes: int) -> list[Path]:
    """Explore traces for `games`, or [] when exploration is disabled."""
    if steps <= 0:
        return []
    print(f"exploring: {episodes} × {steps} random actions per game")
    return explore_games(games, steps, episodes)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DEFAULT, help="public_games-dataset folder")
    p.add_argument("--games", default="all", help="comma-separated game ids, or 'all'")
    p.add_argument("--limit", type=int, default=0, help="max recordings (0 = no limit), for smoke tests")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=128, help="training minibatch")
    p.add_argument("--encode-batch", type=int, default=128, help="frames per encoder forward")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dyn-lr-mult", type=float, default=1.0,
                   help="LR multiplier for the dynamics (WM predictor) head; 1.0 "
                        "since the LayerNorm dynamics trains stably at the base LR")
    p.add_argument("--explore-steps", type=int, default=400,
                   help="random-walk actions per exploration episode (0 = no exploration)")
    p.add_argument("--explore-episodes", type=int, default=5, help="exploration episodes per game")
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
    model.to(DEVICE)  # encode + train on-device

    explore_paths = gather_explore(games, args.explore_steps, args.explore_episodes)
    if not explore_paths:
        raise SystemExit("no exploration traces gathered — set --explore-steps > 0 "
                         "(human replays are intentionally not used)")
    if args.limit:
        explore_paths = explore_paths[: args.limit]
    print(f"{len(explore_paths)} exploration recordings across {len(games)} games")

    torch.manual_seed(0)  # reproducible shuffles across pretraining runs
    zp_l, oh_l, zn_l, co_l = [], [], [], []  # transitions (+ click coords)
    zf_l, gf_l = [], []                       # frames (z, grids)
    t0 = time.time()
    for i, path in enumerate(explore_paths, 1):
        blob = encode_recording(model, path, args.encode_batch)
        if blob is None:
            continue
        a, b, c, co = build_transitions(blob)
        # Latents kept fp16 in RAM (halves the accumulation peak); standardisation
        # + upcast to fp32 happen per-minibatch in the training loop.
        zp_l.append(a.half()); oh_l.append(b); zn_l.append(c.half()); co_l.append(co)
        zf_l.append(blob["z"].half()); gf_l.append(blob["grids"])
        print(f"  [{i}/{len(explore_paths)}] {path.parent.name}/{path.name}: "
              f"{len(blob['z'])} frames, {len(a)} transitions "
              f"({time.time() - t0:.0f}s)")

    def _cat(xs):
        return torch.cat(xs) if xs else torch.empty(0)

    zp, oh, zn = _cat(zp_l), _cat(oh_l), _cat(zn_l)
    coords_all = _cat(co_l)  # (T, 2) click coords; −1 where the action has none
    z_all, grids_all = _cat(zf_l), _cat(gf_l)
    if len(zp) == 0:
        raise SystemExit("no transitions gathered from exploration traces")
    n_click = int((coords_all[:, 0] >= 0).sum()) if len(coords_all) else 0
    print(f"click coords recorded on {n_click}/{len(zp)} transitions "
          f"(ACTION6); available for coordinate-conditioned dynamics")

    # ── Per-dim standardisation ──────────────────────────────────────────────
    # Cached latents are raw (encode()'s buffers were identity). Compute corpus
    # mean/std over the frame pool, store them in the model, and map every latent
    # into the standardised space the heads train in. Stats in fp32; the big
    # latent tensors stay fp16 and are upcast + standardised per-minibatch in the
    # training loop, keeping peak memory ~half.
    latent_mean = z_all.float().mean(dim=0)
    latent_std = z_all.float().std(dim=0).clamp_min(1e-6)
    model.latent_mean.copy_(latent_mean.to(DEVICE))
    model.latent_std.copy_(latent_std.to(DEVICE))

    # Move training tensors to device (latents stay fp16).
    zp, oh, zn = zp.to(DEVICE), oh.to(DEVICE), zn.to(DEVICE)
    coords_all = coords_all.to(DEVICE)
    z_all, grids_all = z_all.to(DEVICE), grids_all.to(DEVICE)

    print(f"\n{len(zp)} transitions, {len(z_all)} frames, "
          f"latent dim {model.latent_dim} [device {DEVICE}]")

    # ── Train dynamics (+ decoder) ───────────────────────────────────────────
    other = [model.decoder] if not args.no_decoder else []
    groups = [{"params": list(model.dynamics.parameters()), "lr": args.lr * args.dyn_lr_mult},
              {"params": [p for h in other for p in h.parameters()], "lr": args.lr}]
    optimizer = torch.optim.Adam([g for g in groups if g["params"]])
    print(f"dynamics lr {args.lr * args.dyn_lr_mult:g}, decoder lr {args.lr:g}")

    lm, ls = model.latent_mean, model.latent_std  # fp32 (D,) on DEVICE

    def _stdb(x):  # fp16 latents (…, D) → standardised fp32, per minibatch
        return (x.float() - lm) / ls

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(zp), device=DEVICE)
        sums, seen = torch.zeros(2), 0
        for start in range(0, len(order), args.batch):
            sel = order[start:start + args.batch]
            b_zp, b_oh, b_zn = _stdb(zp[sel]), oh[sel], _stdb(zn[sel])

            predicted = model.predict_next(b_zp, b_oh, coords_all[sel])
            dyn_loss = model.prediction_error(predicted, b_zn).mean()

            dec_loss = torch.tensor(0.0, device=DEVICE)
            if not args.no_decoder:
                fsel = torch.randint(0, len(z_all), (len(sel),), device=DEVICE)
                target = grids_all[fsel].long().clamp_(0, NUM_COLORS - 1)
                dec_loss = F.cross_entropy(model.decode_logits(_stdb(z_all[fsel])), target,
                                           weight=cell_class_weights(target))

            loss = dyn_loss + dec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(sel)
            sums += n * torch.tensor([dyn_loss.item(), dec_loss.item()])
            seen += n
        d, dc = (sums / seen).tolist()
        print(f"epoch {epoch:2}/{args.epochs}  dynamics {d:.4f}  decoder {dc:.4f}")

    # ── Save in the shape WorldModel auto-loads at agent startup ────────────
    model.cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "meta": {
            "latent_dim": model.latent_dim,
            "n_actions": N_ACTIONS,
            "coord_dim": model.COORD_DIM,
            "lewm_checkpoint": os.environ["LEWM_CHECKPOINT"],
            "transitions": len(zp),
            "frames": len(z_all),
            "data": "random-exploration-only",
            "epochs": args.epochs,
            "games": games,
        },
        "dynamics": model.dynamics.state_dict(),
        "decoder": model.decoder.state_dict(),
        "latent_mean": model.latent_mean.cpu(),
        "latent_std": model.latent_std.cpu(),
    }
    torch.save(blob, args.out)
    print(f"\nsaved pretrained dynamics + decoder "
          f"(+ standardisation stats) → {args.out}")


if __name__ == "__main__":
    main()