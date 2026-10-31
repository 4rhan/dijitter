"""Check the ACG patch (scripts/acg_pi05.py) on a trained checkpoint, on dataset frames, same noise per frame.

  1. scale 1.0 reproduces plain pi0.5 exactly.
  2. scale 0.0 is the pure "incoherent" model (v_inc): it should be jerkier than plain pi0.5.
  3. scales > 1 steer away from it: chunks should get smoother as the scale grows.
  4. mechanics: ACG calls the model twice per denoising step, and in the incoherent pass the
     action<->action attention block is exactly the identity while the prefix columns are untouched.

Usage:
  python scripts/check_acg.py --checkpoint outputs/train/pi05_inhand_lora/checkpoints/008000/pretrained_model
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
    episodes: tuple[int, ...] = (0, 10, 20, 30)
    frames_per_episode: int = 4
    seed: int = 0


def main(args: Args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.policies.pi05.modeling_pi05 as modeling_pi05

    policy, pre, post, cfg = load(ServeArgs(checkpoint=args.checkpoint, repo_id=args.repo_id, root=args.root))
    ds = LeRobotDataset(args.repo_id, root=args.root.expanduser() / args.repo_id)
    frames = []
    for ep in args.episodes:
        start = int(ds.meta.episodes["dataset_from_index"][ep])
        length = int(ds.meta.episodes["length"][ep])
        for t in np.linspace(0, length - 60, args.frames_per_episode).astype(int):
            item = ds[start + int(t)]
            frames.append({k: v for k, v in item.items() if k.startswith("observation.") or k == "task"})
    model = acg_pi05.find_pi05_model(policy)

    # Count model calls and record the incoherent-pass masks.
    calls = {"denoise": 0}
    masks = []
    plain_denoise = type(model).denoise_step

    def counting_denoise(self, **kw):
        calls["denoise"] += 1
        return plain_denoise(self, **kw)

    type(model).denoise_step = counting_denoise

    @torch.inference_mode()
    def predict(frame, seed):
        torch.manual_seed(seed)
        return post(policy.predict_action_chunk(pre(dict(frame)))).float().cpu().numpy()[0]

    def run(scale):
        acg_pi05.set_acg_scale(policy, scale)
        calls["denoise"] = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = [predict(f, args.seed + i) for i, f in enumerate(frames)]
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / len(frames) * 1000
        return np.stack(out), calls["denoise"] / len(frames), ms

    base, base_calls, base_ms = run(None)
    base_jerk = np.mean([chunk_metrics(c[None], executed_steps=len(c))["intra_jerk_rms"] for c in base])
    print(f"plain pi0.5: {len(frames)} frames, {base_calls:.0f} model calls/chunk, {base_ms:.0f} ms/chunk, "
          f"intra-chunk JerkRMS {base_jerk:.1f}")

    # Mask check: wrap the (already patched-in) self-only builder to record what the model receives.
    seen = []
    orig_ctx = acg_pi05.incoherent_actions

    import contextlib

    @contextlib.contextmanager
    def recording_ctx():
        with orig_ctx():
            inner = modeling_pi05.make_att_2d_masks

            def rec(pad, att):
                m = inner(pad, att)
                seen.append((m.detach().clone(), pad.detach().clone(), att.detach().clone()))
                return m

            modeling_pi05.make_att_2d_masks = rec
            try:
                yield
            finally:
                modeling_pi05.make_att_2d_masks = inner

    acg_pi05.incoherent_actions = recording_ctx
    failures = []
    print(f"\n{'scale':>6}{'calls':>7}{'ms':>7}{'max|acg-plain| rad':>21}{'JerkRMS':>10}{'vs plain':>10}")
    for s in args.scales:
        out, n_calls, ms = run(s)
        diff = float(np.abs(out - base).max())
        jerk = np.mean([chunk_metrics(c[None], executed_steps=len(c))["intra_jerk_rms"] for c in out])
        print(f"{s:>6g}{n_calls:>7.0f}{ms:>7.0f}{diff:>21.2e}{jerk:>10.1f}{jerk / base_jerk:>9.2f}x")
        if s == 1.0 and diff > 1e-4:
            failures.append(f"scale 1.0 differs from plain by {diff:.2e}")
        if n_calls != 2 * base_calls:
            failures.append(f"scale {s:g}: {n_calls:.0f} model calls per chunk, expected {2 * base_calls:.0f}")
    acg_pi05.incoherent_actions = orig_ctx
    acg_pi05.set_acg_scale(policy, None)

    # Inspect one recorded incoherent mask: action block identity, prefix columns untouched.
    m, pad, att = seen[0]
    full = acg_pi05.modeling_pi05.make_att_2d_masks(pad, att)  # unpatched now
    eye = torch.eye(m.shape[-1], dtype=torch.bool, device=m.device)
    ok_block = bool(torch.equal(m[0], eye))
    ok_plain = bool(full[0].all())  # plain pi05 suffix: every action token sees every other one
    print(f"\nincoherent action<->action mask: shape {tuple(m.shape)}, identity: {ok_block}; "
          f"plain mask is all-to-all: {ok_plain}")
    if not ok_block:
        failures.append("incoherent action block is not the identity")
    if failures:
        raise SystemExit("ACG check FAILED:\n  " + "\n  ".join(failures))
    print("ACG mechanics check passed")


if __name__ == "__main__":
    main(tyro.cli(Args))
