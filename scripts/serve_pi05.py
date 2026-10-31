"""Policy server for a LeRobot pi0.5 LoRA checkpoint (optionally with ACG), for scripts/eval_sim.py.

Speaks the same websocket protocol as scripts/mock_policy_server.py and openpi's server (metadata frame
on connect, then msgpack-numpy request/response):
  client -> server: {"images": {top/left/right: (H, W, 3) uint8}, "state": (14,), "prompt": str, "frame_index": int}
  server -> client: {"actions": (50, 14), "server_timing": {"infer_ms": ...}}

The checkpoint is loaded the way LeRobot loads one: pi05_base from the HF cache plus the LoRA adapter
(merged into the weights for inference), and the checkpoint's own saved pre/post-processors, which
hold the training stats, the camera renaming and (for RELATIVE=1 runs) the relative<->absolute steps.

Usage:
  python scripts/serve_pi05.py --checkpoint outputs/train/pi05_inhand_lora/checkpoints/last/pretrained_model
  python scripts/serve_pi05.py --checkpoint ... --acg-scale 2.0
  python scripts/serve_pi05.py --checkpoint ... --acg-scale 1.25 --acg-layers 15-17
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_to_lerobot import CAMERA_KEYS, REPO  # noqa: E402
from test_pi05 import RENAME_MAP  # noqa: E402


@dataclasses.dataclass
class Args:
    checkpoint: Path
    """A checkpoint's pretrained_model/ directory (adapter + config + processors)."""
    repo_id: str = "abc_inhand_transfer"
    root: Path = REPO / "data" / "lerobot"
    """Parent directory of the training dataset (only its metadata is read)."""
    acg_scale: float | None = None
    """ACG guidance scale (scripts/acg_pi05.py); None = plain pi0.5."""
    acg_layers: str = "half"
    """Action-expert layers ACG perturbs: "half" (last half), "all", or a range like "15-17"."""
    num_steps: int | None = None
    """Flow-matching denoising steps (default: the checkpoint's, 10)."""
    seed: int = 0
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000


def load(args: Args):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies import make_policy, make_pre_post_processors

    ckpt = str(args.checkpoint.expanduser().resolve())
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root.expanduser() / args.repo_id)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = ckpt
    cfg.device = args.device
    if args.num_steps is not None:
        cfg.num_inference_steps = args.num_steps
    policy = make_policy(cfg, ds_meta=meta, rename_map=RENAME_MAP)
    if hasattr(policy, "merge_and_unload"):  # PEFT wrapper -> plain PI05Policy with LoRA folded in
        policy = policy.merge_and_unload()
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return policy, pre, post, cfg


class Pi05Server:
    def __init__(self, args: Args):
        self.policy, self.pre, self.post, self.cfg = load(args)
        if args.acg_scale is not None:
            import acg_pi05

            layers = acg_pi05.parse_layers(args.acg_layers, acg_pi05.find_pi05_model(self.policy))
            acg_pi05.set_acg_scale(self.policy, args.acg_scale, layers)
        torch.manual_seed(args.seed)

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        batch = {
            key: torch.from_numpy(np.asarray(obs["images"][cam])).permute(2, 0, 1).float() / 255.0
            for cam, key in CAMERA_KEYS.items()
        }
        batch["observation.state"] = torch.as_tensor(np.asarray(obs["state"], dtype=np.float32))
        batch["task"] = obs["prompt"]
        actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        return {"actions": actions.float().cpu().numpy()[0]}


async def serve(args: Args) -> None:
    import websockets.asyncio.server as ws_server
    from openpi_client import msgpack_numpy

    server = Pi05Server(args)
    # Warm-up: the first call compiles kernels and allocates the KV cache.
    h, w = 168, 224
    dummy = {"images": {c: np.zeros((h, w, 3), np.uint8) for c in CAMERA_KEYS}, "state": np.zeros(14, np.float32),
             "prompt": "warm up", "frame_index": 0}
    t0 = time.monotonic()
    shape = server.infer(dummy)["actions"].shape
    print(f"warm-up chunk {shape} in {(time.monotonic() - t0) * 1000:.0f} ms", flush=True)
    if args.device == "cuda":
        print(f"GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak", flush=True)

    packer = msgpack_numpy.Packer()
    metadata = {
        "backend": "lerobot_pi05",
        "checkpoint": str(args.checkpoint),
        "acg_scale": args.acg_scale,
        "acg_layers": args.acg_layers if args.acg_scale is not None else None,
        "num_steps": server.cfg.num_inference_steps,
        "state_dim": 14,
    }

    async def handler(websocket):
        await websocket.send(packer.pack(metadata))
        async for message in websocket:
            t0 = time.monotonic()
            action = server.infer(msgpack_numpy.unpackb(message))
            action["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
            await websocket.send(packer.pack(action))

    async with ws_server.serve(handler, args.host, args.port, compression=None, max_size=None) as ws:
        print(f"pi05 server (acg_scale={args.acg_scale}, layers={args.acg_layers}) on ws://{args.host}:{args.port}", flush=True)
        await ws.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve(tyro.cli(Args)))
