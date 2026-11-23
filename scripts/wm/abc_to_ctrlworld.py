"""Convert ABC YAM sim episodes into Ctrl-World's training format (github.com/Robert-gyj/Ctrl-World).

Ctrl-World (DROID) stores per episode: 3 camera mp4s at 5 Hz (192x320), their SVD-VAE latents, and the
robot state at 15 Hz (3 state samples per video frame). Its "action" conditioning is the robot state of
every history/future frame: DROID uses 6-D cartesian pose + gripper; for the YAM we use the 14-D
policy-space joint state, written as 13 "cartesian" dims + 1 "gripper" dim so the stock loader concatenates
it back into the same 14-D vector.

Source: ABC episodes (30 Hz): combined_camera-images-rgb.mp4 (top/left/right stacked vertically,
224x168 each) and states_actions.bin (float64, [T, 28] = 14-D state, 14-D action).

  python scripts/wm/abc_to_ctrlworld.py --src third_party/abc/cache/train_sim --out ~/wm_data/yam \
      --svd ~/wm_models/stable-video-diffusion-img2vid --val-every 20

Writes out/{annotation,videos,latent_videos}/{train,val}/<id> and out/../meta/<name>/{train,val}_sample.json + stat.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import numpy as np
import torch

PROMPT = "sim_inhand transfer the item to other side"
SRC_HZ, STATE_HZ, VIDEO_HZ = 30, 15, 5
CAMS = ("top", "left", "right")


def load_episode(ep: Path):
    raw = np.fromfile(ep / "states_actions.bin", dtype=np.float64).reshape(-1, 28)
    video = mediapy.read_video(str(ep / "combined_camera-images-rgb.mp4"))  # (T, 504, 224, 3)
    meta = json.loads((ep / "episode_metadata.json").read_text())
    cams = meta.get("cameras", list(CAMS))
    h = video.shape[1] // len(cams)
    views = [video[:, i * h:(i + 1) * h] for i in range(len(cams))]
    return raw[:, :14], views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="directory of ABC episode_* folders")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--svd", type=Path, required=True, help="stable-video-diffusion-img2vid folder (for its VAE)")
    ap.add_argument("--val-every", type=int, default=20, help="every n-th episode goes to val")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--success", type=int, default=1, help="success label written for these episodes")
    ap.add_argument("--start-id", type=int, default=0, help="first episode id (to append rollouts after demos)")
    a = ap.parse_args()

    from diffusers import AutoencoderKLTemporalDecoder

    vae = AutoencoderKLTemporalDecoder.from_pretrained(a.svd, subfolder="vae").to("cuda").eval()
    eps = sorted(p.parent for p in a.src.expanduser().glob("episode_*/states_actions.bin"))[: a.max_episodes]
    out = a.out.expanduser()
    for k, ep in enumerate(eps):
        eid = a.start_id + k
        split = "val" if k % a.val_every == a.val_every - 1 else "train"
        ann_path = out / "annotation" / split / f"{eid}.json"
        if ann_path.exists():
            continue
        state, views = load_episode(ep)
        n = min(len(state), len(views[0]))
        state15 = state[:n:SRC_HZ // STATE_HZ]
        nframes = int(np.floor((len(state15) - 1) / (STATE_HZ // VIDEO_HZ))) + 1
        vids, lats = [], []
        for c, v in enumerate(views):
            frames = torch.tensor(v[:n:SRC_HZ // VIDEO_HZ][:nframes]).permute(0, 3, 1, 2).float() / 255.0 * 2 - 1
            x = torch.nn.functional.interpolate(frames, size=(192, 320), mode="bilinear", align_corners=False)
            vid_dir, lat_dir = out / "videos" / split / str(eid), out / "latent_videos" / split / str(eid)
            vid_dir.mkdir(parents=True, exist_ok=True); lat_dir.mkdir(parents=True, exist_ok=True)
            rgb = ((x / 2 + 0.5).clamp(0, 1) * 255).permute(0, 2, 3, 1).numpy().astype(np.uint8)
            mediapy.write_video(str(vid_dir / f"{c}.mp4"), rgb, fps=VIDEO_HZ)
            with torch.no_grad():
                lat = torch.cat([vae.encode(x[i:i + 64].cuda()).latent_dist.sample().mul_(vae.config.scaling_factor).cpu()
                                 for i in range(0, len(x), 64)])
            torch.save(lat, lat_dir / f"{c}.pt")
            vids.append({"video_path": f"videos/{split}/{eid}/{c}.mp4"})
            lats.append({"latent_video_path": f"latent_videos/{split}/{eid}/{c}.pt"})
        ann = {
            "texts": [PROMPT], "episode_id": str(eid), "source": ep.name, "success": a.success,
            "video_length": nframes, "videos": vids, "latent_videos": lats,
            "observation.state.joint_position": state15.tolist(),
            "observation.state.cartesian_position": state15[:, :13].tolist(),
            "observation.state.gripper_position": state15[:, 13].tolist(),
        }
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps(ann))
        if k % 50 == 0:
            print(f"[{k + 1}/{len(eps)}] {ep.name} -> {split}/{eid}: {nframes} frames", flush=True)

    # meta: one training sample per start frame (every 2nd frame), and 14-D state percentiles
    meta = out.parent / "meta" / out.name  # Ctrl-World reads <meta_path>/<dataset_name>/{split}_sample.json, stat.json
    meta.mkdir(parents=True, exist_ok=True)
    allstate = []
    for split in ("train", "val"):
        samples = []
        for f in sorted((out / "annotation" / split).glob("*.json")):
            ann = json.loads(f.read_text())
            allstate.append(np.array(ann["observation.state.joint_position"]))
            for fid in range(0, ann["video_length"] - 4, 2):
                samples.append({"episode_id": ann["episode_id"], "frame_ids": [fid]})
        (meta / f"{split}_sample.json").write_text(json.dumps(samples))
        print(f"{split}: {len(samples)} samples")
    s = np.concatenate(allstate)
    (meta / "stat.json").write_text(json.dumps({"state_01": np.percentile(s, 1, axis=0).tolist(),
                                                "state_99": np.percentile(s, 99, axis=0).tolist()}))
    print(f"meta -> {meta}")


if __name__ == "__main__":
    main()
