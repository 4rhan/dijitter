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
  python scripts/abc_to_lerobot.py --max-episodes 100 --workers 8   # shards in parallel, then merged

LeRobot's writer spends ~30 s of single-threaded Python per episode (per-frame image stats in the
encoder threads), so --workers converts contiguous shards in separate processes and merges them in
order with LeRobot's aggregate_datasets. check_lerobot.py verifies the result episode by episode.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import inspect
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import av
import numpy as np
import tyro

REPO = Path(__file__).resolve().parents[1]
STATE_DIM = 14
FPS = 30
MAX_EPISODE_FRAMES = 4000  # longest ABC sim episode is 1030 steps
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
    vcodec: str = "h264"
    """Video codec passed to LeRobot. "h264" runs on the CPU; "h264_nvenc" hung with LeRobot 0.6.1
    streaming encoding on the GPU box."""
    workers: int = 1
    """Parallel converter processes (each takes a contiguous shard of the episodes)."""


def load_episode(ep: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta = json.loads((ep / "episode_metadata.json").read_text())
    table = np.fromfile(ep / "states_actions.bin", dtype=np.float64).reshape(-1, 2 * STATE_DIM)
    return meta, table[:, :STATE_DIM].astype(np.float32), table[:, STATE_DIM:].astype(np.float32)


def load_frames(video: Path, cameras: list[str]) -> dict[str, np.ndarray]:
    """Decode the combined video and split it into per-camera (T, H, W, 3) uint8 arrays."""
    with av.open(str(video)) as container:
        container.streams.video[0].thread_type = "AUTO"  # multi-threaded decode
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
    h = frames.shape[1] // len(cameras)
    return {cam: frames[:, i * h : (i + 1) * h] for i, cam in enumerate(cameras)}


def create_dataset(repo_id: str, root: Path, height: int, width: int, vcodec: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        key: {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
        for key in CAMERA_KEYS.values()
    }
    features["observation.state"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": NAMES}
    features["action"] = {"dtype": "float32", "shape": (STATE_DIM,), "names": NAMES}
    # H.264 encodes far faster than LeRobot's AV1 default. The codec option moved between versions:
    # 0.5 takes create(vcodec=...); 0.6 takes an RGBEncoderConfig and can stream frames straight into
    # the encoder instead of writing PNGs first.
    params = inspect.signature(LeRobotDataset.create).parameters
    kwargs = {"image_writer_threads": 4}
    if "rgb_encoder" in params:
        from lerobot.configs.video import RGBEncoderConfig

        nvenc = vcodec.endswith("_nvenc")
        kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec=vcodec, crf=None if nvenc else 30)
        kwargs["streaming_encoding"] = True
        # The streaming encoder DROPS a frame when its queue is full (it is built for live recording),
        # and we feed frames far faster than real time. A queue longer than any episode never fills.
        kwargs["encoder_queue_maxsize"] = MAX_EPISODE_FRAMES
        kwargs["image_writer_threads"] = 0
    elif "vcodec" in params:
        kwargs["vcodec"] = vcodec
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=features,
        root=root,
        robot_type="yam",
        use_videos=True,
        **kwargs,
    )


def convert(episodes: list[Path], repo_id: str, root: Path, vcodec: str, tag: str = "") -> int:
    """Write `episodes` (in order) to a new LeRobot dataset at `root`; returns the frame count."""
    ds = None
    total_frames, t0 = 0, time.perf_counter()
    for n, ep in enumerate(episodes, 1):
        meta, state, action = load_episode(ep)
        cameras = meta["cameras"]
        frames = load_frames(ep / meta["videos"]["combined"], cameras)
        T = min(len(state), *(len(v) for v in frames.values()))
        if T != len(state):
            print(f"  [warn] {ep.name}: {len(state)} steps but {T} video frames; truncating")
        if T > MAX_EPISODE_FRAMES:
            raise SystemExit(f"{ep.name}: {T} frames > MAX_EPISODE_FRAMES={MAX_EPISODE_FRAMES}; raise it")
        if ds is None:
            h, w = frames[cameras[0]].shape[1:3]
            ds = create_dataset(repo_id, root, h, w, vcodec)
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
        print(f"  {tag}[{n}/{len(episodes)}] {ep.name} T={T}  elapsed {elapsed / 60:.1f} min  eta {eta / 60:.1f} min", flush=True)

    if hasattr(ds, "finalize"):
        ds.finalize()  # LeRobot v3 buffers parquet/video writers until finalize
    return total_frames


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
    workers = max(1, min(args.workers, len(episodes)))
    print(f"converting {len(episodes)} episodes from {src} -> {root} ({workers} worker(s))")
    t0 = time.perf_counter()

    if workers == 1:
        total_frames = convert(episodes, args.repo_id, root, args.vcodec)
    else:
        from lerobot.datasets.aggregate import aggregate_datasets

        shard_dir = root.parent / f"{args.repo_id}.shards"
        shutil.rmtree(shard_dir, ignore_errors=True)
        shards = [list(s) for s in np.array_split(np.array(episodes, dtype=object), workers)]
        ids = [f"{args.repo_id}_shard{i}" for i in range(workers)]
        roots = [shard_dir / i for i in ids]
        # spawn: the LeRobot writer starts threads, which do not survive a fork
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(workers, mp_context=ctx) as pool:
            jobs = [
                pool.submit(convert, shard, i, r, args.vcodec, f"shard{k} ")
                for k, (shard, i, r) in enumerate(zip(shards, ids, roots, strict=True))
            ]
            total_frames = sum(j.result() for j in jobs)
        print(f"merging {workers} shards -> {root}")
        aggregate_datasets(ids, args.repo_id, roots=roots, aggr_root=root)
        shutil.rmtree(shard_dir)
    print(f"done: {len(episodes)} episodes, {total_frames} frames -> {root} in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(tyro.cli(Args))
