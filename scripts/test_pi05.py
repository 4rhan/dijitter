"""Smoke-test LeRobot pi0.5 on our converted ABC dataset: load, predict action chunks, time them.

Builds the policy and pre/post-processors the same way `lerobot-train` does (dataset stats for
quantile normalization, our camera keys renamed to the ones pi05_base was trained with), then runs
`predict_action_chunk` on a few dataset frames.

pi05_base has never seen YAM, so its actions are not expected to match the demos; this checks the
pipeline (shapes, normalization, GPU memory, latency) before fine-tuning and before ACG.

Usage:
  python scripts/test_pi05.py --repo-id abc_inhand_transfer
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
    latencies = []
    for i, t in enumerate(starts):
        frame = ds[int(t)]
        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        pred = predict_chunk(policy, pre, post, frame, args.seed)
        if args.device == "cuda":
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t1)

        gt = np.stack([ds[int(t) + k]["action"].numpy() for k in range(min(chunk, ep_len - t))])
        n = min(len(gt), len(pred))
        err = np.abs(pred[:n] - gt[:n]).mean()
        m = chunk_metrics(pred[None], executed_steps=len(pred))
        print(
            f"  frame {t:4d}: chunk {pred.shape}  |pred-demo| {err:.3f} rad  "
            f"range [{pred.min():+.2f}, {pred.max():+.2f}]  intra JerkRMS {m['intra_jerk_rms']:.1f}  "
            f"{latencies[-1] * 1000:.0f} ms"
        )

    warm = latencies[1:] or latencies
    print(f"latency: first {latencies[0] * 1000:.0f} ms, then {np.mean(warm) * 1000:.0f} ms/chunk")
    if args.device == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main(tyro.cli(Args))
