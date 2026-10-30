"""A stand-in for the π0.5 policy server, for testing scripts/eval_sim.py without a GPU.

Speaks the same websocket protocol as openpi's WebsocketPolicyServer (metadata frame on connect,
then msgpack-numpy request/response), so eval_sim.py cannot tell it apart from the real server.

Modes:
  hold     every chunk repeats the current state (arms stay still): checks the loop runs.
  replay   returns the recorded demo actions of --episode-dir from the client's frame_index on;
           run eval_sim.py with --replay-episode <same dir>. Checks that the sim reproduces the
           demo, that the success evaluator fires, and gives a smoothness reference for the
           metrics on the exact same code path as π0.5.
  noisy    replay plus Gaussian noise on every action (--noise-std rad): a deliberately jittery policy,
           to confirm the metrics separate smooth from jerky.

Usage:
  python scripts/mock_policy_server.py --mode replay --episode-dir third_party/abc/cache/val_sim/episode_<id>
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import tyro
import websockets.asyncio.server as ws_server
from openpi_client import msgpack_numpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_to_lerobot import load_episode  # noqa: E402


@dataclasses.dataclass
class Args:
    mode: str = "hold"
    """hold | replay | noisy"""
    episode_dir: Path | None = None
    horizon: int = 50
    """Chunk length returned per call (π0.5 YAM predicts 50)."""
    noise_std: float = 0.01
    seed: int = 0
    host: str = "0.0.0.0"
    port: int = 8000


class MockPolicy:
    def __init__(self, args: Args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.actions = None
        if args.mode in ("replay", "noisy"):
            if args.episode_dir is None:
                raise SystemExit(f"--mode {args.mode} needs --episode-dir")
            _, _, self.actions = load_episode(args.episode_dir.expanduser())
        elif args.mode != "hold":
            raise SystemExit(f"unknown --mode {args.mode!r}")

    def infer(self, obs: dict) -> dict:
        H = self.args.horizon
        if self.actions is None:
            chunk = np.repeat(np.asarray(obs["state"], dtype=np.float32)[None], H, axis=0)
        else:
            # states[t] is the response to actions[t], so the next command after observing frame t is actions[t + 1].
            idx = np.clip(np.arange(H) + int(obs.get("frame_index", 0)) + 1, 0, len(self.actions) - 1)
            chunk = self.actions[idx].copy()
            if self.args.mode == "noisy":
                chunk += self.rng.normal(0.0, self.args.noise_std, size=chunk.shape).astype(np.float32)
        return {"actions": chunk}


async def serve(args: Args) -> None:
    policy = MockPolicy(args)
    packer = msgpack_numpy.Packer()
    metadata = {"backend": "mock", "mode": args.mode, "state_dim": 14}

    async def handler(websocket):
        await websocket.send(packer.pack(metadata))
        async for message in websocket:
            t0 = time.monotonic()
            action = policy.infer(msgpack_numpy.unpackb(message))
            action["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
            await websocket.send(packer.pack(action))

    async with ws_server.serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        print(f"mock policy server ({args.mode}) on ws://{args.host}:{args.port}", flush=True)
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(serve(tyro.cli(Args)))
