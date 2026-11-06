"""Check a running scripts/serve_karma.py before the robot: replay recorded frames through it exactly the
way karma sends them, and compare the returned chunks with what the demonstrator did next.

For random frames of the training dataset this sends the three views as JPEG q95 (karma's default),
the state with karma's 1=open gripper, and the dataset's task, then flips the returned gripper back
and measures the error against the next 50 recorded actions. A healthy server is far below the
"hold still" baseline (predicting the current state for the whole chunk); a camera, gripper or
normalisation mix-up shows up as an error near or above it.

  python scripts/check_karma_server.py --dataset-root data/lerobot/yam_two_cube
  python scripts/check_karma_server.py --server http://192.168.0.130:8202 --n 40

Frames come from the training set, so this checks plumbing, not generalisation.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import time
from pathlib import Path

import numpy as np
import tyro

GRIPPER = [6, 13]
NAMES = [f"{side}_{j}" for side in ("L", "R") for j in ("j1", "j2", "j3", "j4", "j5", "j6", "grip")]
WIRE = {"top_cam": "observation.images.top", "left_cam": "observation.images.left_wrist",
        "right_cam": "observation.images.right_wrist"}


@dataclasses.dataclass
class Args:
    dataset_root: Path = Path("data/lerobot/yam_two_cube")
    server: str = "http://127.0.0.1:8202"
    n: int = 24
    """Frames to replay."""
    norm_tag: str = "pi05_two_cube"
    jpeg_quality: int = 95
    seed: int = 0


def numpy_json(o):
    if isinstance(o, np.ndarray):
        from numpy.lib.format import dtype_to_descr

        o = np.ascontiguousarray(o)
        return {"__numpy__": base64.b64encode(o.tobytes()).decode(), "dtype": dtype_to_descr(o.dtype), "shape": o.shape}
    raise TypeError(type(o))


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=quality)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def main(args: Args) -> None:
    import requests
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    url = args.server.rstrip("/") + "/act"
    health = requests.get(url, timeout=10).json()
    print("health:", health)
    for key, want in (("status", "ok"), ("norm_tag", args.norm_tag), ("state_dim", 14), ("num_cameras", 3)):
        if health.get(key) != want:
            raise SystemExit(f"FAIL: health {key}={health.get(key)!r}, karma would want {want!r}")

    root = args.dataset_root.resolve()
    meta_fps = json.loads((root / "meta" / "info.json").read_text())["fps"]
    ds = LeRobotDataset(root.name, root=root, video_backend="pyav",
                        delta_timestamps={"action": [i / meta_fps for i in range(50)]})
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(ds.num_frames, size=args.n, replace=False)

    errors, hold_errors, latencies, grip_agree = [], [], [], []
    for index in picks:
        sample = ds[int(index)]
        images = {wire: (sample[key].permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                  for wire, key in WIRE.items()}
        state = sample["observation.state"].numpy().astype(np.float32)
        wire_state = state.copy()
        wire_state[GRIPPER] = 1.0 - wire_state[GRIPPER]  # karma's wire: 1=open
        payload = {**{wire: jpeg(image, args.jpeg_quality) for wire, image in images.items()},
                   "instruction": sample["task"], "state": wire_state, "timestamp": time.time(),
                   "num_steps": 10, "normalization_tag": args.norm_tag, "enable_cuda_graph": True}
        started = time.perf_counter()
        response = requests.post(url, data=json.dumps(payload, default=numpy_json),
                                 headers={"Content-Type": "application/json"}, timeout=60)
        latencies.append(time.perf_counter() - started)
        if response.status_code != 200:
            raise SystemExit(f"FAIL: HTTP {response.status_code}: {response.text}")
        chunk = np.asarray(response.json()["actions"], dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != 14:
            raise SystemExit(f"FAIL: chunk shape {chunk.shape}")
        chunk[:, GRIPPER] = 1.0 - chunk[:, GRIPPER]  # back to the dataset's 0=open
        truth = sample["action"].numpy()
        valid = ~sample["action_is_pad"].numpy()
        n = min(len(chunk), len(truth))
        valid = valid[:n]
        errors.append(np.abs(chunk[:n] - truth[:n])[valid].mean(0))
        hold_errors.append(np.abs(state[None, :] - truth[:n])[valid].mean(0))
        grip_agree.append(np.mean((chunk[:n, GRIPPER] > 0.5) == (truth[:n, GRIPPER] > 0.5)))

    err, hold = np.mean(errors, 0), np.mean(hold_errors, 0)
    np.set_printoptions(precision=3, suppress=True)
    print(f"\n{args.n} frames, round trip median {np.median(latencies) * 1000:.0f} ms")
    print("joint      " + " ".join(f"{n:>7}" for n in NAMES))
    print("policy MAE " + " ".join(f"{v:7.3f}" for v in err))
    print("hold MAE   " + " ".join(f"{v:7.3f}" for v in hold))
    ratio = err.mean() / max(hold.mean(), 1e-9)
    print(f"mean policy/hold error ratio {ratio:.2f}; gripper open/closed agreement {np.mean(grip_agree):.0%}")
    if ratio < 0.6 and np.mean(grip_agree) > 0.8:
        print("PASS: the server predicts the recorded motion far better than holding still")
    else:
        print("CHECK: policy error is close to holding still, or the gripper disagrees; look for a camera, "
              "gripper or checkpoint mix-up before running the robot")


if __name__ == "__main__":
    main(tyro.cli(Args))
