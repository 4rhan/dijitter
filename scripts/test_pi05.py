"""Smoke-test LeRobot pi0.5 on our converted ABC dataset: load, predict action chunks, time them.

Builds the policy and pre/post-processors the same way `lerobot-train` does (dataset stats for
quantile normalization, our camera keys renamed to the ones pi05_base was trained with), then runs
`predict_action_chunk` on a few dataset frames.

pi05_base has never seen YAM, so its actions are not expected to match the demos; this checks the
pipeline (shapes, normalization, GPU memory, latency) before fine-tuning.

With --acg-scales, the same frames are re-run with ACG (scripts/acg_pi05.py) under the same noise
seed and compared to the baseline: scale 1.0 must reproduce it exactly, larger scales should differ
and give lower intra-chunk jerk, at about twice the latency.

Usage:
  python scripts/test_pi05.py --repo-id abc_inhand_transfer
  python scripts/test_pi05.py --acg-scales 1.0 2.0 3.0
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
from abc_to_lerobot import REPO  # noqa: E402
from metrics import chunk_metrics  # noqa: E402

# Our dataset keys -> the camera keys pi05_base expects.
RENAME_MAP = {
    "observation.images.top_rgb": "observation.images.base_0_rgb",
    "observation.images.left_rgb": "observation.images.left_wrist_0_rgb",
    "observation.images.right_rgb": "observation.images.right_wrist_0_rgb",
}


@dataclasses.dataclass
class Args:
    repo_id: str = "abc_inhand_transfer"
    root: Path = REPO / "data" / "lerobot"
    """Parent directory of the dataset (<root>/<repo_id>)."""
    policy: str = "lerobot/pi05_base"
    num_frames: int = 5
    """Frames to predict on (spread across episode 0)."""
    acg_scales: tuple[float, ...] = ()
    """ACG scales to compare against the baseline (e.g. 1.0 2.0 3.0)."""
    dtype: str = "bfloat16"
    device: str = "cuda"
    seed: int = 0


def load_policy(args: Args, ds_meta):
    """Policy + processors, built like lerobot_train.py does for a pretrained checkpoint."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.pretrained_path = args.policy
    cfg.device = args.device
    cfg.dtype = args.dtype
    policy = make_policy(cfg, ds_meta=ds_meta, rename_map=RENAME_MAP)
    policy.eval()

    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "normalizer_processor": {
                "stats": ds_meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": RENAME_MAP},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": ds_meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )
    return policy, pre, post


@torch.no_grad()
def predict_chunk(policy, pre, post, frame: dict, seed: int) -> np.ndarray:
    """One action chunk (chunk_size, action_dim) for a single dataset frame, with fixed noise seed."""
    batch = {k: v for k, v in frame.items() if k.startswith("observation.") or k == "task"}
    torch.manual_seed(seed)
    actions = policy.predict_action_chunk(pre(batch))
    return post(actions).float().cpu().numpy()[0]


def main(args: Args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root.expanduser() / args.repo_id)
    print(f"{args.repo_id}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")

    t0 = time.perf_counter()
    policy, pre, post = load_policy(args, ds.meta)
    n_params = sum(p.numel() for p in policy.parameters()) / 1e9
    print(f"loaded {args.policy} ({n_params:.2f}B params, {args.dtype}) in {time.perf_counter() - t0:.0f} s")
    if args.device == "cuda":
        print(f"GPU memory after load: {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

    ep_len = int(ds.meta.episodes["length"][0])
    chunk = policy.config.chunk_size
    starts = np.linspace(0, max(ep_len - chunk, 0), args.num_frames).astype(int)
    frames = [ds[int(t)] for t in starts]
    demos = [np.stack([ds[int(t) + k]["action"].numpy() for k in range(min(chunk, ep_len - t))]) for t in starts]

    baseline = run(policy, pre, post, frames, demos, starts, args, label="baseline")
    if args.acg_scales:
        from acg_pi05 import set_acg_scale

        print("\nACG vs baseline (same frames, same noise):")
        failures = []
        for scale in args.acg_scales:
            set_acg_scale(policy, scale)
            preds = run(policy, pre, post, frames, demos, starts, args, label=f"acg {scale:g}", verbose=False)
            diff = max(np.abs(p - b).max() for p, b in zip(preds, baseline, strict=True))
            print(f"    max |acg - baseline| = {diff:.2e} rad")
            if scale == 1.0 and diff > 1e-4:
                failures.append(f"scale 1.0 should reproduce the baseline, differs by {diff:.2e}")
            if scale != 1.0 and diff == 0.0:
                failures.append(f"scale {scale:g} has no effect: the incoherent pass is not perturbing the model")
        set_acg_scale(policy, None)
        if failures:
            raise SystemExit("ACG check failed:\n  " + "\n  ".join(failures))
        print("ACG check passed")
    if args.device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def run(policy, pre, post, frames, demos, starts, args: Args, label: str, verbose: bool = True) -> list[np.ndarray]:
    """Predict a chunk per frame; print per-frame (verbose) and summary stats; return the chunks."""
    preds, latencies, errs, jerks = [], [], [], []
    for t, frame, demo in zip(starts, frames, demos, strict=True):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred = predict_chunk(policy, pre, post, frame, args.seed)
        if args.device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t1)
        n = min(len(demo), len(pred))
        errs.append(np.abs(pred[:n] - demo[:n]).mean())
        jerks.append(chunk_metrics(pred[None], executed_steps=len(pred))["intra_jerk_rms"])
        preds.append(pred)
        if verbose:
            print(
                f"  frame {t:4d}: chunk {pred.shape}  |pred-demo| {errs[-1]:.3f} rad  "
                f"range [{pred.min():+.2f}, {pred.max():+.2f}]  intra JerkRMS {jerks[-1]:.1f}  "
                f"{latencies[-1] * 1000:.0f} ms"
            )
    warm = latencies[1:] or latencies
    print(
        f"  {label:>9}: |pred-demo| {np.mean(errs):.3f} rad  intra JerkRMS {np.mean(jerks):.1f}  "
        f"latency {np.mean(warm) * 1000:.0f} ms/chunk (first {latencies[0] * 1000:.0f} ms)"
    )
    return preds

if __name__ == "__main__":
    main(tyro.cli(Args))
