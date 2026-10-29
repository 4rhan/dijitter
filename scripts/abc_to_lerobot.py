"""Convert ABC sim episodes (cache/{train,val}_sim) to a LeRobot v3.0 dataset for openpi's pi05_yam_lora.

ABC episode layout (see abc/abc_minimal/episode_io.py):
  states_actions.bin                float64 (T, 28): 14 recorded dofs, then 14 commanded dofs
  combined_camera-images-rgb.mp4    cameras stacked vertically, each (H, W) = (168, 224)
  episode_metadata.json             instruction, cameras, num_steps, fps

LeRobot keys match LeRobotYamDataConfig in yam-abc-reproduce's openpi config:
  observation.images.{top,left,right}_rgb, observation.state (14,), action (14,)
Actions stay absolute; the openpi config converts arm joints to deltas itself.

Usage:
  python scripts/abc_to_lerobot.py --repo-id abc_inhand_transfer --max-episodes 20
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import shutil
import time
from pathlib import Path

import av
import numpy as np
import tyro

REPO = Path(__file__).resolve().parents[1]
STATE_DIM = 14
FPS = 30
NAMES = [
    *[f"left_joint_{i}" for i in range(6)],
    "left_gripper",
    *[f"right_joint_{i}" for i in range(6)],
    "right_gripper",
]
# ABC camera name -> LeRobot key expected by the openpi YAM config.
CAMERA_KEYS = {
    "top": "observation.images.top_rgb",
    "left": "observation.images.left_rgb",
    "right": "observation.images.right_rgb",
}


@dataclasses.dataclass
class Args:
    src: Path = REPO / "third_party" / "abc" / "cache" / "train_sim"
    """Directory of ABC episode_* folders."""
    repo_id: str = "abc_inhand_transfer"
    """LeRobot dataset name (passed to openpi as --data.repo-id)."""
    out: Path = REPO / "data" / "lerobot"
    """Parent directory; the dataset is written to <out>/<repo_id>."""
    task: str = ""
    """Only convert episodes whose task_name matches (empty = all)."""
    max_episodes: int | None = None
    """Convert only the first N episodes (for a quick check)."""
    overwrite: bool = False


def load_episode(ep: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta = json.loads((ep / "episode_metadata.json").read_text())
    table = np.fromfile(ep / "states_actions.bin", dtype=np.float64).reshape(-1, 2 * STATE_DIM)
    return meta, table[:, :STATE_DIM].astype(np.float32), table[:, STATE_DIM:].astype(np.float32)


def load_frames(video: Path, cameras: list[str]) -> dict[str, np.ndarray]:
    """Decode the combined video and split it into per-camera (T, H, W, 3) uint8 arrays."""
    with av.open(str(video)) as container:
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    h = frames.shape[1] // len(cameras)
    return {cam: frames[:, i * h : (i + 1) * h] for i, cam in enumerate(cameras)}


def create_dataset(repo_id: str, root: Path, height: int, width: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        key: {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
        for key in CAMERA_KEYS.values()
    }
    features["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": NAMES}
    features["action"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": NAMES}
    kwargs = {}
    # H.264 encodes far faster than LeRobot's AV1 default on a laptop CPU, when this version allows choosing.
    if "vcodec" in inspect.signature(LeRobotDataset.create).parameters:
        kwargs["vcodec"] = "h264"
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=root,
        robot_type="yam",
        use_videos=True,
        image_writer_threads=4,
        **kwargs,
    )


def main(args: Args) -> None:
    src = args.src.expanduser()
    root = args.out.expanduser() / args.repo_id
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    episodes = sorted(p.parent for p in src.glob("episode_*/states_actions.bin"))
    if args.task:
        episodes = [e for e in episodes if json.loads((e / "episode_metadata.json").read_text())["task_name"] == args.task]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise SystemExit(f"no episodes found under {src}")
    print(f"converting {len(episodes)} episodes from {src} -> {root}")

    ds = None
    total_frames, t0 = 0, time.perf_counter()
    for n, ep in enumerate(episodes, 1):
        meta, state, action = load_episode(ep)
        cameras = meta["cameras"]
        frames = load_frames(ep / meta["videos"]["combined"], cameras)
        T = min(len(state), *(len(v) for v in frames.values()))
        if T != len(state):
            print(f"  [warn] {ep.name}: {len(state)} steps but {T} video frames; truncating")
        if ds is None:
            h, w = frames[cameras[0]].shape[1:3]
            ds = create_dataset(args.repo_id, root, h, w)
        prompt = meta["instruction"]
        for t in range(T):
            frame = {"observation.state": state[t], "action": action[t], "task": prompt}
            for cam, key in CAMERA_KEYS.items():
                frame[key] = frames[cam][t]
            ds.add_frame(frame)
        ds.save_episode()
        total_frames += T
        elapsed = time.perf_counter() - t0
        eta = elapsed / n * (len(episodes) - n)
        print(f"  [{n}/{len(episodes)}] {ep.name} T={T}  elapsed {elapsed / 60:.1f} min  eta {eta / 60:.1f} min", flush=True)

    if hasattr(ds, "finalize"):
        ds.finalize()  # LeRobot v3 buffers parquet/video writers until finalize
    print(f"done: {len(episodes)} episodes, {total_frames} frames -> {root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
