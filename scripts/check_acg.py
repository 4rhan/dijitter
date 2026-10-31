"""Check the ACG patch (scripts/acg_pi05.py) on a trained checkpoint, on dataset frames, same noise per frame.

  1. scale 1.0 reproduces plain pi0.5 exactly (and skips the second pass).
  2. scale 0.0 is the pure "incoherent" model (v_inc): it should be jerkier than plain pi0.5.
  3. scales > 1 steer away from it: chunks should get smoother, without moving far from plain pi0.5.
  4. mechanics: two model calls per denoising step; in the incoherent pass, exactly the chosen layers
     get a mask whose actions<->actions block is the identity, with the prefix columns untouched.

Layer sets: "half" (default, last half of the action expert), "all", or a range like "12-17".

Usage:
  python scripts/check_acg.py --checkpoint outputs/train/pi05_inhand_lora/checkpoints/008000/pretrained_model
  python scripts/check_acg.py --checkpoint ... --layer-sets half all 15-17
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acg_pi05  # noqa: E402
from abc_to_lerobot import REPO  # noqa: E402
from metrics import chunk_metrics  # noqa: E402
from serve_pi05 import Args as ServeArgs  # noqa: E402
from serve_pi05 import load  # noqa: E402


@dataclasses.dataclass
class Args:
    checkpoint: Path
    repo_id: str = "abc_inhand_transfer"
    root: Path = REPO / "data" / "lerobot"
    scales: tuple[float, ...] = (0.0, 1.0, 1.5, 2.0, 3.0)
    layer_sets: tuple[str, ...] = ("half", "all")
    episodes: tuple[int, ...] = (0, 10, 20, 30)
    frames_per_episode: int = 4
    seed: int = 0


def parse_layers(spec: str, model) -> list[int]:
    n = len(acg_pi05.expert_layers(model))
    if spec == "all":
        return list(range(n))
    if spec == "half":
        return acg_pi05.default_layers(model)
    lo, hi = (int(x) for x in spec.split("-"))
    return list(range(lo, hi + 1))


def jerk(chunks: np.ndarray) -> float:
    return float(np.mean([chunk_metrics(c[None], executed_steps=len(c))["intra_jerk_rms"] for c in chunks]))


def main(args: Args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    policy, pre, post, _ = load(ServeArgs(checkpoint=args.checkpoint, repo_id=args.repo_id, root=args.root))
    ds = LeRobotDataset(args.repo_id, root=args.root.expanduser() / args.repo_id)
    frames = []
    for ep in args.episodes:
        start = int(ds.meta.episodes["dataset_from_index"][ep])
        length = int(ds.meta.episodes["length"][ep])
        for t in np.linspace(0, length - 60, args.frames_per_episode).astype(int):
            item = ds[start + int(t)]
            frames.append({k: v for k, v in item.items() if k.startswith("observation.") or k == "task"})
    model = acg_pi05.find_pi05_model(policy)

    # Count model calls (patched on the class before ACG captures the bound method) and record the
    # masks the incoherent pass builds.
    calls = {"denoise": 0}
    plain_denoise = type(model).denoise_step

    def counting_denoise(self, **kw):
        calls["denoise"] += 1
        return plain_denoise(self, **kw)

    type(model).denoise_step = counting_denoise
    rewrites = []
    plain_identity = acg_pi05.identity_actions

    def recording_identity(mask):
        out = plain_identity(mask)
        rewrites.append((mask.detach(), out.detach()))
        return out

    acg_pi05.identity_actions = recording_identity

    @torch.inference_mode()
    def predict(frame, seed):
        torch.manual_seed(seed)
        return post(policy.predict_action_chunk(pre(dict(frame)))).float().cpu().numpy()[0]

    def run(scale, layers=None):
        acg_pi05.set_acg_scale(policy, scale, layers)
        calls["denoise"] = 0
        rewrites.clear()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = np.stack([predict(f, args.seed + i) for i, f in enumerate(frames)])
        torch.cuda.synchronize()
        return out, calls["denoise"] / len(frames), (time.perf_counter() - t0) / len(frames) * 1000

    base, base_calls, base_ms = run(None)
    base_jerk = jerk(base)
    steps = int(base_calls)
    print(f"plain pi0.5: {len(frames)} frames, {steps} model calls/chunk, {base_ms:.0f} ms/chunk, "
          f"intra-chunk JerkRMS {base_jerk:.1f}")

    failures = []
    for spec in args.layer_sets:
        layers = parse_layers(spec, model)
        print(f"\nperturbed layers {spec} = {layers[0]}..{layers[-1]} ({len(layers)} of {len(acg_pi05.expert_layers(model))})")
        print(f"{'scale':>6}{'calls':>7}{'ms':>7}{'max|acg-plain| rad':>21}{'mean|acg-plain|':>17}{'JerkRMS':>10}{'vs plain':>10}")
        for s in args.scales:
            out, n_calls, ms = run(s, layers)
            d = np.abs(out - base)
            j = jerk(out)
            print(f"{s:>6g}{n_calls:>7.0f}{ms:>7.0f}{d.max():>21.2e}{d.mean():>17.2e}{j:>10.1f}{j / base_jerk:>9.2f}x")
            expected_calls = steps if s == 1.0 else 2 * steps
            if n_calls != expected_calls:
                failures.append(f"{spec} scale {s:g}: {n_calls:.0f} model calls per chunk, expected {expected_calls}")
            if s == 1.0 and d.max() > 0:
                failures.append(f"{spec} scale 1.0 differs from plain by {d.max():.2e}")
            if s != 1.0:
                per_chunk = len(rewrites) / len(frames)
                if per_chunk != steps * len(layers):
                    failures.append(f"{spec} scale {s:g}: {per_chunk:.0f} mask rewrites per chunk, expected {steps * len(layers)}")
                before, after = rewrites[0]
                n = before.shape[-2]
                eye = torch.eye(n, dtype=torch.bool, device=after.device)
                block_ok = bool(torch.equal(after[0, 0, :, -n:] == 0, eye & (before[0, 0, :, -n:] == 0)))
                prefix_ok = bool(torch.equal(after[..., :-n], before[..., :-n]))
                if not (block_ok and prefix_ok):
                    failures.append(f"{spec} scale {s:g}: mask block identity {block_ok}, prefix untouched {prefix_ok}")

    acg_pi05.identity_actions = plain_identity
    acg_pi05.set_acg_scale(policy, None)
    if failures:
        raise SystemExit("ACG check FAILED:\n  " + "\n  ".join(failures))
    print("\nACG mechanics check passed (calls, identity at 1.0, per-layer mask rewrite)")


if __name__ == "__main__":
    main(tyro.cli(Args))
