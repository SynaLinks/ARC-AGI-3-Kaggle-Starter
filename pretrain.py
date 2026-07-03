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

  artefact  `agent/models/pretrained_heads.pt` — dynamics, surprise head,
            no-op head, reward head and decoder probe trained on all
            transitions with exactly the online losses (plus the
            displacement-gate scale τ). Auto-loaded by `WorldModel` at agent
            startup. The reward head (did this action complete a level / win?)
            is pretrain-only — its positives come from human replays.

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

from agent.my_agent import ARC_PALETTE, NUM_COLORS, WorldModel

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
    noop = (blob["grids"][idx] == blob["grids"][idx - 1]).flatten(1).all(dim=1).float()
    # Reward label for action_i: it produced game progress — levels_completed
    # ticked up, or the game entered WIN on this line. Exact but very sparse;
    # positives come almost entirely from human replays.
    won = torch.tensor([s == "WIN" for s in blob["states"]], dtype=torch.bool)
    reward = ((blob["levels"][idx] > blob["levels"][idx - 1]) | won[idx]).float()
    return z_prev, onehot, z_next, noop, reward


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
    recordings = [f for g in games for f in sorted((args.data / g).glob("*.recording.jsonl"))]
    if args.explore_steps > 0:
        print(f"exploring: {args.explore_episodes} × {args.explore_steps} random actions per game")
        recordings += explore_games(games, args.explore_steps, args.explore_episodes)
    if args.limit:
        recordings = recordings[: args.limit]
    print(f"{len(recordings)} recordings across {len(games)} games")

    zp, oh, zn, noop, rew, z_all, grids_all = [], [], [], [], [], [], []
    t0 = time.time()
    for i, path in enumerate(recordings, 1):
        blob = encode_recording(model, path, args.encode_batch)
        if blob is None:
            continue
        a, b, c, d, e = build_transitions(blob)
        zp.append(a); oh.append(b); zn.append(c); noop.append(d); rew.append(e)
        z_all.append(blob["z"]); grids_all.append(blob["grids"])
        print(f"  [{i}/{len(recordings)}] {path.parent.name}/{path.name}: "
              f"{len(blob['z'])} frames, {len(a)} transitions ({time.time() - t0:.0f}s)")

    zp, oh, zn, noop, rew = map(torch.cat, (zp, oh, zn, noop, rew))
    z_all, grids_all = torch.cat(z_all), torch.cat(grids_all)
    print(f"\n{len(zp)} transitions, {len(z_all)} frames, "
          f"noop share {noop.mean():.1%}, reward share {rew.mean():.2%} "
          f"({int(rew.sum())} positives), latent dim {model.latent_dim}")

    # Reward positives (level completions) are far rarer than no-ops — weight
    # the positive class by the actual imbalance so BCE doesn't collapse to
    # "never rewarded". Capped so a near-empty positive set can't blow up.
    reward_pos_weight = torch.tensor(
        float(((1.0 - rew.mean()) / rew.mean().clamp_min(1e-6)).clamp(1.0, 500.0)))

    # ── Train the same parts the agent trains online ─────────────────────────
    heads = [model.dynamics, model.surprise_head, model.noop_head, model.reward_head]
    if not args.no_decoder:
        heads.append(model.decoder)
    optimizer = torch.optim.Adam((q for h in heads for q in h.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(zp))
        sums, seen, correct = torch.zeros(5), 0, 0
        noop_hits, noop_total = 0, 0  # recall on the rare positive class
        rew_hits, rew_total = 0, 0
        for start in range(0, len(order), args.batch):
            sel = order[start:start + args.batch]
            b_zp, b_oh, b_zn, b_noop, b_rew = zp[sel], oh[sel], zn[sel], noop[sel], rew[sel]

            predicted = model.predict_next(b_zp, b_oh)
            dyn_loss = model.realized_surprise(predicted, b_zn).mean()
            target = model.realized_surprise(predicted.detach(), b_zn)
            sur_loss = ((model.predicted_surprise(b_zp, b_oh) - target) ** 2).mean()
            logits = model.noop_logit(b_zp, b_oh)
            noop_loss = F.binary_cross_entropy_with_logits(
                logits, b_noop, pos_weight=torch.tensor(WorldModel.NOOP_POS_WEIGHT))
            rew_logits = model.reward_logit(b_zp, b_oh)
            rew_loss = F.binary_cross_entropy_with_logits(
                rew_logits, b_rew, pos_weight=reward_pos_weight)

            dec_loss = torch.tensor(0.0)
            if not args.no_decoder:
                # Decoder pairs are independent of transitions — sample frames.
                fsel = torch.randint(0, len(z_all), (len(sel),))
                chunk = grids_all[fsel].long().clamp_(0, NUM_COLORS - 1)
                rgb = ARC_PALETTE[chunk].permute(0, 3, 1, 2)
                dec_loss = F.mse_loss(model.decode(z_all[fsel]), rgb)

            loss = dyn_loss + sur_loss + noop_loss + rew_loss + dec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = len(sel)
            sums += n * torch.tensor([dyn_loss.item(), sur_loss.item(),
                                      noop_loss.item(), rew_loss.item(), dec_loss.item()])
            correct += ((logits > 0).float() == b_noop).sum().item()
            noop_hits += ((logits > 0) & (b_noop > 0)).sum().item()
            noop_total += (b_noop > 0).sum().item()
            rew_hits += ((rew_logits > 0) & (b_rew > 0)).sum().item()
            rew_total += (b_rew > 0).sum().item()
            seen += n
        d, s, np_, rw, dc = (sums / seen).tolist()
        recall = noop_hits / max(noop_total, 1)
        rew_recall = rew_hits / max(rew_total, 1)
        print(f"epoch {epoch:2}/{args.epochs}  dynamics {d:.4f}  surprise {s:.4f}  "
              f"noop {np_:.4f} (acc {correct / seen:.1%}, recall {recall:.1%})  "
              f"reward {rw:.4f} (recall {rew_recall:.1%})  decoder {dc:.4f}")

    # Displacement-gate scale τ: typical realized latent displacement of
    # frame-changing transitions (get_cost normalizes ‖ẑ′ − z‖ by this).
    with torch.no_grad():
        changed = noop < 0.5
        if changed.any():
            model.disp_scale.fill_(float((zn[changed] - zp[changed]).norm(dim=-1).mean()))
    print(f"displacement scale τ = {float(model.disp_scale):.4f}")

    # ── Save in the shape WorldModel auto-loads at agent startup ────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "meta": {
            "latent_dim": model.latent_dim,
            "n_actions": N_ACTIONS,
            "lewm_checkpoint": os.environ["LEWM_CHECKPOINT"],
            "transitions": len(zp),
            "reward_positives": int(rew.sum()),
            "frames": len(z_all),
            "epochs": args.epochs,
            "games": games,
        },
        "dynamics": model.dynamics.state_dict(),
        "surprise_head": model.surprise_head.state_dict(),
        "noop_head": model.noop_head.state_dict(),
        "reward_head": model.reward_head.state_dict(),
        "decoder": model.decoder.state_dict(),
        "disp_scale": float(model.disp_scale),
    }, args.out)
    print(f"\nsaved pretrained heads → {args.out}")


if __name__ == "__main__":
    main()
