"""Check a converted LeRobot dataset against the ABC episodes it came from.

For every converted episode this verifies:
  * frame count matches the source episode
  * observation.state / action match states_actions.bin exactly (float32)
  * each camera's decoded frames match the source video (mean abs pixel error; small = video re-encode only)
  * the prompt is the source instruction
and writes outputs/check/<repo_id>_ep<k>.png: source (top row) vs converted (bottom row) for top/left/right cameras.

Episodes are matched in the same sorted order abc_to_lerobot.py converts them.

Usage:
  python scripts/check_lerobot.py --repo-id abc_inhand_transfer_test
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_to_lerobot import CAMERA_KEYS, REPO, load_episode, load_frames  # noqa: E402

OUT = REPO / "outputs" / "check"
PIXEL_TOL = 8.0  # mean abs error (0-255) allowed from H.264 re-encoding


@dataclasses.dataclass
class Args:
    repo_id: str = "abc_inhand_transfer_test"
    src: Path = REPO / "third_party" / "abc" / "cache" / "train_sim"
    out: Path = REPO / "data" / "lerobot"
    num_frames_checked: int = 5
    """Frames per episode compared against the source video (evenly spaced)."""


def main(args: Args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = args.out.expanduser() / args.repo_id
    ds = LeRobotDataset(args.repo_id, root=root)
    episodes = sorted(p.parent for p in args.src.expanduser().glob("episode_*/states_actions.bin"))
    print(f"{args.repo_id}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, fps={ds.meta.fps}")
    OUT.mkdir(parents=True, exist_ok=True)

    failures = 0
    for k in range(ds.meta.total_episodes):
        ep = episodes[k]
        meta, state, action = load_episode(ep)
        src_frames = load_frames(ep / meta["videos"]["combined"], meta["cameras"])

        start = int(ds.meta.episodes["dataset_from_index"][k])
        end = int(ds.meta.episodes["dataset_to_index"][k])
        T = end - start
        rows = ds.hf_dataset.select(range(start, end)).with_format("numpy")
        ds_state, ds_action = np.stack(rows["observation.state"]), np.stack(rows["action"])

        errs = []
        if T != len(state):
            errs.append(f"length {T} != source {len(state)}")
        n = min(T, len(state))
        state_err = float(np.abs(ds_state[:n] - state[:n]).max())
        action_err = float(np.abs(ds_action[:n] - action[:n]).max())
        if state_err > 1e-6 or action_err > 1e-6:
            errs.append(f"state err {state_err:.2e}, action err {action_err:.2e}")

        pix_err = {}
        grid_rows = None
        for t in np.linspace(0, n - 1, args.num_frames_checked).astype(int):
            item = ds[start + int(t)]
            if item["task"] != meta["instruction"]:
                errs.append(f"prompt {item['task']!r} != {meta['instruction']!r}")
            src_row, ds_row = [], []
            for cam, key in CAMERA_KEYS.items():
                conv = (item[key].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)  # (C,H,W) float -> HWC
                src = src_frames[cam][t]
                pix_err.setdefault(cam, []).append(float(np.abs(conv.astype(float) - src.astype(float)).mean()))
                src_row.append(src)
                ds_row.append(conv)
            if grid_rows is None:  # save the first checked frame as a side-by-side picture
                grid_rows = np.concatenate([np.concatenate(src_row, 1), np.concatenate(ds_row, 1)], 0)
        mean_pix = {cam: np.mean(v) for cam, v in pix_err.items()}
        if any(v > PIXEL_TOL for v in mean_pix.values()):
            errs.append(f"pixel err too high {mean_pix}")
        imageio.imwrite(OUT / f"{args.repo_id}_ep{k}.png", grid_rows)

        status = "OK  " if not errs else "FAIL"
        failures += bool(errs)
        pix = " ".join(f"{c}={v:.1f}" for c, v in mean_pix.items())
        print(f"[{status}] ep{k} {ep.name} T={T} state/action err={max(state_err, action_err):.1e} pixel err {pix}")
        for e in errs:
            print(f"        {e}")

    print(f"\n{ds.meta.total_episodes - failures}/{ds.meta.total_episodes} episodes OK; side-by-side images in {OUT}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(tyro.cli(Args))
