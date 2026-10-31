"""Closed-loop eval of a policy server in the ABC YAM sim (CPU MuJoCo), with smoothness logging.

Speaks the yam-abc-reproduce wire contract, so the same client works against
  * the real π0.5 server (third_party/yam-abc-reproduce/.../deploy/servers/openpi_server.py), and
  * scripts/mock_policy_server.py (hold / demo-replay), for testing this loop without a GPU.

  client -> server: {"images": {top/left/right: (H, W, 3) uint8}, "state": (14,), "prompt": str, "frame_index": int}
  server -> client: {"actions": (H, 14), "server_timing": {"infer_ms": ...}}

Per run it writes outputs/eval/<tag>/:
  summary.json          success rate, latency, smoothness (executed + per-chunk), args
  episode_<k>.npz       chunks (N, H, 14), chunk_frames (N,), executed (T, 14), states (T, 14), latency_ms, infer_ms
  episode_<k>.mp4       (with --save-video) top|left|right camera strip

Usage:
  python scripts/eval_sim.py --tag baseline --num-episodes 10            # against a server on localhost:8000
  python scripts/eval_sim.py --host <gpu-box-ip> --tag acg_s2 --num-episodes 50
  python scripts/eval_sim.py --replay-episode <abc episode dir> --tag mock_replay   # with mock server in replay mode
  python scripts/eval_sim.py --viser --tag watch --num-episodes 3   # live 3D view at http://<host>:8080, real time
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_to_lerobot import REPO  # noqa: E402
from metrics import chunk_metrics, trajectory_metrics  # noqa: E402

ABC = REPO / "third_party" / "abc"
sys.path.insert(0, str(ABC))
CAMERAS = ("top", "left", "right")
# The instruction stored in every converted training episode; π0.5 must see the same text at eval.
TRAIN_PROMPT = "sim_inhand transfer the item to other side"


@dataclasses.dataclass
class Args:
    host: str = "localhost"
    port: int = 8000
    task: str = "inhand_transfer"
    """abc_sim task (see abc_sim/task_specs.py)."""
    prompt: str = TRAIN_PROMPT
    num_episodes: int = 10
    seed: int = 0
    """Episode k uses seed + k, so runs with the same seed see the same worlds."""
    execute_steps: int = 25
    """Actions executed from each predicted chunk before re-querying (π0.5 predicts 50)."""
    max_steps: int = 1200
    """Per-episode cap on executed actions (30 Hz; demos are ~400-550 steps)."""
    camera_height: int = 168
    camera_width: int = 224
    replay_episode: Path | None = None
    """Reset to this recorded ABC episode's scene instead of a random world (for the mock replay server)."""
    tag: str = "run"
    out: Path = REPO / "outputs" / "eval"
    save_video: bool = False
    viser: bool = False
    """Show the rollout live in a browser (viser, like abc's viz_policy.py), paced to real time."""
    viser_port: int = 8080


class LiveViewer:
    """Browser 3D view of the env being stepped (same mjviser scene abc's viz_policy.py uses)."""

    def __init__(self, port: int):
        import asyncio
        import math

        import viser

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.status = self.server.gui.add_text("status", initial_value="starting", disabled=True)
        self.model = None
        self.scene = None
        self.next_t = 0.0

        @self.server.on_client_connect
        async def _(client) -> None:
            await asyncio.sleep(0.1)
            client.camera.position = np.array([-0.42, 0.0, 1.66])
            client.camera.look_at = np.array([0.45, 0.0, 0.87])
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])
            client.camera.fov = math.radians(50.0)

        print(f"live view: http://<this machine>:{self.server.get_port()}", flush=True)

    def show(self, env, status: str) -> None:
        from mjviser import ViserMujocoScene

        env = getattr(env, "unwrapped", env)
        if env.model is not self.model:  # abc_sim recompiles the MjModel on every reset
            self.server.scene.reset()
            self.scene = ViserMujocoScene(self.server, env.model, num_envs=1)
            self.scene.camera_tracking_enabled = False
            self.server.scene.remove_by_name("/fixed_bodies/world/table_plane")
            self.model = env.model
        self.scene.update_from_mjdata(env.data)
        self.status.value = status
        # Real time: one control step (timestep x decimation) of wall clock per action.
        period = float(env.model.opt.timestep) * getattr(env, "_control_decimation", 1)
        now = time.perf_counter()
        if self.next_t > now:
            time.sleep(self.next_t - now)
        self.next_t = max(now, self.next_t) + period


def make_env(args: Args):
    import abc_sim

    spec = abc_sim.get_task_spec(args.task)
    kwargs = dict(
        task=spec.env_task,
        prompt=args.prompt,
        render_cameras=True,
        camera_backend="mujoco",
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        max_episode_steps=args.max_steps,
        terminate_on_success=True,
    )
    env = abc_sim.make_env(**kwargs)
    if args.replay_episode is None:
        return env, None

    from abc_minimal.episode_io import load_episode

    # The task randomizer is seed-driven and each episode records its seed, so the same task
    # (and its success evaluator) can rebuild the recorded world; see reset().
    ep = args.replay_episode.expanduser()
    metadata, states, _ = load_episode(ep)
    world_seed = int(json.loads((ep / "randomization.json").read_text())["seed"])
    return env, (ep, metadata, states, world_seed)


def reset(env, replay, seed: int):
    if replay is None:
        obs, _ = env.reset(seed=seed, randomize=True)
        return obs
    from abc_minimal.episode_io import seed_initial_state

    ep, metadata, states, world_seed = replay
    env.reset(seed=world_seed, randomize=True)
    print(f"replay {ep.name}: world seed {world_seed}, start {seed_initial_state(env, ep, metadata, states)}")
    return env.get_obs()


def to_request(obs: dict, prompt: str, frame_index: int) -> dict:
    return {
        "images": {cam: np.ascontiguousarray(obs["images"][cam].transpose(1, 2, 0)) for cam in CAMERAS},
        "state": np.asarray(obs["state"], dtype=np.float32),
        "prompt": prompt,
        "frame_index": frame_index,
    }


def run_episode(env, replay, policy, args: Args, seed: int, viewer: LiveViewer | None = None):
    obs = reset(env, replay, seed)
    if viewer is not None:
        viewer.show(env, f"seed {seed}: starting")
    chunks, chunk_frames, latency_ms, infer_ms = [], [], [], []
    executed, states, frames = [], [], []
    success, truncated, t = False, False, 0
    while t < args.max_steps and not success:
        t0 = time.perf_counter()
        resp = policy.infer(to_request(obs, args.prompt, t))
        latency_ms.append((time.perf_counter() - t0) * 1000)
        infer_ms.append(float(resp.get("server_timing", {}).get("infer_ms", np.nan)))
        chunk = np.asarray(resp["actions"], dtype=np.float32)
        chunks.append(chunk)
        chunk_frames.append(t)

        n = min(args.execute_steps, len(chunk), args.max_steps - t)
        for i in range(n):
            last = i == n - 1
            step_obs, _, terminated, truncated, info = env.step(chunk[i], render_obs=last or args.save_video)
            executed.append(chunk[i])
            states.append(np.asarray(env.get_state(), dtype=np.float32))
            if viewer is not None:
                viewer.show(env, f"seed {seed}: step {t + 1}/{args.max_steps}  chunk {len(chunks)}"
                            f"{'  SUCCESS' if info.get('task_success', False) else ''}")
            if args.save_video and step_obs is not None:
                frames.append(np.concatenate([step_obs["images"][c].transpose(1, 2, 0) for c in CAMERAS], axis=1))
            if last and step_obs is not None:
                obs = step_obs
            t += 1
            if info.get("task_success", False):
                success = True
            if terminated or truncated:
                if step_obs is None:
                    obs = env.get_obs()
                break
        if truncated:
            break

    return {
        "success": bool(success),
        "steps": t,
        "chunks": np.stack(chunks),
        "chunk_frames": np.asarray(chunk_frames),
        "executed": np.stack(executed),
        "states": np.stack(states),
        "latency_ms": np.asarray(latency_ms),
        "infer_ms": np.asarray(infer_ms),
        "frames": frames,
    }


def main(args: Args) -> None:
    from openpi_client import websocket_client_policy

    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"server metadata: {policy.get_server_metadata()}")
    env, replay = make_env(args)
    viewer = LiveViewer(args.viser_port) if args.viser else None

    per_episode = []
    try:
        for k in range(args.num_episodes):
            seed = args.seed + k
            r = run_episode(env, replay, policy, args, seed, viewer)
            m = {
                "episode": k,
                "seed": seed,
                "success": r["success"],
                "steps": r["steps"],
                "latency_ms_mean": float(r["latency_ms"].mean()),
                "infer_ms_mean": float(np.nanmean(r["infer_ms"])) if np.isfinite(r["infer_ms"]).any() else None,
                **trajectory_metrics(r["executed"]),
                **chunk_metrics(r["chunks"], args.execute_steps),
            }
            per_episode.append(m)
            np.savez_compressed(out / f"episode_{k}.npz", **{key: r[key] for key in
                                ("chunks", "chunk_frames", "executed", "states", "latency_ms", "infer_ms")})
            if args.save_video and r["frames"]:
                import imageio.v2 as imageio

                imageio.mimsave(out / f"episode_{k}.mp4", r["frames"], fps=30)
            print(f"[{k + 1}/{args.num_episodes}] seed={seed} success={m['success']} steps={m['steps']} "
                  f"latency={m['latency_ms_mean']:.0f}ms jerk={m['jerk_rms']:.2f} atv={m['atv']:.5f} "
                  f"switch={m.get('switch_jump', float('nan')):.4f}", flush=True)
    finally:
        env.close()

    keys = ("latency_ms_mean", "jerk_rms", "atv", "intra_jerk_rms", "intra_atv", "switch_jump")
    summary = {
        "tag": args.tag,
        "episodes": len(per_episode),
        "success_rate": float(np.mean([m["success"] for m in per_episode])),
        **{f"{key}": float(np.nanmean([m.get(key, np.nan) for m in per_episode])) for key in keys},
        "args": {k: str(v) for k, v in dataclasses.asdict(args).items()},
        "per_episode": per_episode,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{args.tag}: success {summary['success_rate']:.0%} over {len(per_episode)} episodes | "
          f"latency {summary['latency_ms_mean']:.0f} ms | JerkRMS {summary['jerk_rms']:.2f} | "
          f"ATV {summary['atv']:.5f} | switch jump {summary['switch_jump']:.4f}  -> {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
