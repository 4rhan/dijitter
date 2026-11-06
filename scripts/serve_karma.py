"""HTTP policy server for real-robot rollouts with karma (github.com/SRA-VJTI/karma) and a LeRobot
pi0.5 LoRA checkpoint trained on karma recordings (e.g. Archaive16/yam-two-cube-place-01-clean).

Karma never loads a model: `karma inference/rollout/hitl --server http://HOST:PORT` talks to a server
like this one (docs/model-servers.md and docs/inference.md in karma):
  GET  /act -> {"status": "ok", "norm_tag": ..., "state_dim": 14, "num_cameras": 3}
              karma refuses to energize the arms unless norm_tag matches its --norm-tag
  POST /act <- json_numpy {"top_cam", "left_cam", "right_cam": JPEG bytes (1-D uint8) or HxWx3 RGB,
                           "state": (14,) float32, "instruction": str, "num_steps", ...}
            -> {"actions": (50, 14) absolute joint targets, "dt_ms": GPU time}

The one conversion: karma's wire gripper is 1=open (native YAM), but karma writes datasets with
0=open (record.py), so the checkpoint learned 0=open. The gripper columns (6, 13) are flipped on the
state coming in and on the actions going out. Joints are radians on both sides, nothing to convert.

Images go through the checkpoint's own saved processors exactly as in training (camera renaming,
quantile stats, resize-with-pad to 224). Karma's cameras default to 848x480, the training size.

  python scripts/serve_karma.py --checkpoint outputs/train/pi05_two_cube_lora/checkpoints/last/pretrained_model
  python scripts/serve_karma.py --mock      # no model, no GPU: every chunk holds the current pose
  curl http://HOST:8202/act                 # health, from the robot PC

--mock answers with the measured state repeated, so a karma rollout against it energizes, streams
cameras and holds still. Use it to test the robot PC <-> server link before running the policy.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import tyro

REPO = Path(__file__).resolve().parent.parent
STATE_DIM = 14
GRIPPER_COLUMNS = [6, 13]
# karma wire key -> karma camera role; karma records each role as observation.images.<role>.
WIRE_CAMERAS = {"top_cam": "top", "left_cam": "left_wrist", "right_cam": "right_wrist"}


@dataclasses.dataclass
class Args:
    checkpoint: Path | None = None
    """A checkpoint's pretrained_model/ directory. Not needed with --mock."""
    norm_tag: str = "pi05_two_cube"
    """What the health check reports; pass the same value to karma as --norm-tag."""
    prompt: str | None = None
    """Use this task text for every request instead of karma's instruction (default: the dataset's
    task when it has exactly one, so a typo at karma's prompt cannot change the conditioning)."""
    use_instruction: bool = False
    """Condition on karma's instruction as typed, even when the dataset has a single task."""
    num_steps: int | None = None
    """Flow-matching denoising steps (default: the checkpoint's, 10). karma's num_steps is ignored."""
    flip_gripper: bool = True
    """Convert karma's 1=open gripper to the dataset's 0=open and back. Only off for a checkpoint
    trained on data that kept 1=open."""
    dataset_root: Path | None = None
    """Training dataset directory, for its metadata (default: read from the checkpoint's train_config)."""
    save_frames: Path | None = None
    """Save the decoded camera frames of every 30th request here, to check framing and colours."""
    mock: bool = False
    host: str = "0.0.0.0"
    port: int = 8202
    device: str = "cuda"
    seed: int = 0


# ---- json_numpy wire format (what karma's client sends), without the dependency -----------------

def _numpy_hook(obj: dict):
    if "__numpy__" in obj:
        from numpy.lib.format import descr_to_dtype

        array = np.frombuffer(base64.b64decode(obj["__numpy__"]), descr_to_dtype(obj["dtype"]))
        return array.reshape(obj["shape"]) if obj["shape"] else array[0]
    return obj


def loads(body: bytes | str) -> dict:
    return json.loads(body, object_hook=_numpy_hook)


def decode_image(frame) -> np.ndarray:
    """1-D uint8 = JPEG/PNG bytes, HxWx3 = raw RGB (karma --raw-frames)."""
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[2] == 3:
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))
    if array.ndim == 1 and array.dtype == np.uint8:
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(array.tobytes())).convert("RGB"), dtype=np.uint8)
    raise ValueError(f"camera frame must be encoded bytes (1-D uint8) or HxWx3, got {array.shape} {array.dtype}")


def flip_gripper(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float32, copy=True)
    out[..., GRIPPER_COLUMNS] = 1.0 - out[..., GRIPPER_COLUMNS]
    return out


# ---- policy ---------------------------------------------------------------------------------------

class MockPolicy:
    """Holds the current pose: a chunk of 50 copies of the (dataset-frame) state."""

    tasks: list[str] = []
    image_size = None
    info = {"backend": "mock"}

    def infer(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        return np.repeat(state[None, :], 50, axis=0)


class Pi05Policy:
    def __init__(self, args: Args):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        from lerobot.policies import make_policy, make_pre_post_processors

        self.torch = torch
        ckpt = args.checkpoint.expanduser().resolve()
        train_cfg = json.loads((ckpt / "train_config.json").read_text())
        repo_id = train_cfg["dataset"]["repo_id"]
        root = args.dataset_root or Path(train_cfg["dataset"]["root"] or f"data/lerobot/{repo_id}")
        root = root if root.is_absolute() else REPO / root
        self.rename_map = train_cfg.get("rename_map") or {}
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        self.tasks = list(meta.tasks.index) if hasattr(meta.tasks, "index") else []
        cameras = [key for key in meta.camera_keys]
        self.camera_keys = {wire: f"observation.images.{role}" for wire, role in WIRE_CAMERAS.items()}
        missing = sorted(set(self.camera_keys.values()) - set(cameras))
        if missing:
            raise SystemExit(f"the training dataset has cameras {cameras}, karma sends {sorted(self.camera_keys.values())}; "
                             f"missing in the dataset: {missing}")
        first = meta.features[cameras[0]]["shape"]
        self.image_size = (first[0], first[1]) if first[-1] == 3 else (first[1], first[2])

        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        cfg.pretrained_path = str(ckpt)
        cfg.device = args.device
        if args.num_steps is not None:
            cfg.num_inference_steps = args.num_steps
        policy = make_policy(cfg, ds_meta=meta, rename_map=self.rename_map or None)
        if hasattr(policy, "merge_and_unload"):  # PEFT wrapper -> plain PI05Policy with LoRA folded in
            policy = policy.merge_and_unload()
        policy.eval()
        self.policy = policy
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides={
                "device_processor": {"device": args.device},
                "rename_observations_processor": {"rename_map": self.rename_map},
            },
        )
        torch.manual_seed(args.seed)
        self.info = {
            "backend": "lerobot_pi05",
            "checkpoint": str(ckpt),
            "dataset": repo_id,
            "num_steps": cfg.num_inference_steps,
            "chunk_size": cfg.chunk_size,
            "relative_actions": bool(getattr(cfg, "use_relative_actions", False)),
        }

    def infer(self, images: dict[str, np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            batch = {
                self.camera_keys[wire]: torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                for wire, image in images.items()
            }
            batch["observation.state"] = torch.as_tensor(state, dtype=torch.float32)
            batch["task"] = task
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        return actions.float().cpu().numpy()[0]


# ---- server ---------------------------------------------------------------------------------------

class Server:
    def __init__(self, args: Args):
        self.args = args
        self.policy = MockPolicy() if args.mock else Pi05Policy(args)
        self.lock = threading.Lock()
        self.requests = 0
        self.warned: set[str] = set()
        tasks = self.policy.tasks
        self.fixed_task = args.prompt or (tasks[0] if len(tasks) == 1 and not args.use_instruction else None)
        if args.save_frames:
            args.save_frames.mkdir(parents=True, exist_ok=True)

    def warn_once(self, key: str, message: str) -> None:
        if key not in self.warned:
            self.warned.add(key)
            print(f"WARNING: {message}", flush=True)

    def health(self) -> dict:
        return {"status": "ok", "norm_tag": self.args.norm_tag, "state_dim": STATE_DIM,
                "num_cameras": len(WIRE_CAMERAS), **self.policy.info}

    def act(self, payload: dict) -> dict:
        missing = [key for key in (*WIRE_CAMERAS, "state") if key not in payload]
        if missing:
            raise ValueError(f"missing required field: {missing[0]!r}")
        images = {wire: decode_image(payload[wire]) for wire in WIRE_CAMERAS}
        for wire, image in images.items():
            if self.policy.image_size and image.shape[:2] != tuple(self.policy.image_size):
                self.warn_once(f"size_{wire}", f"{wire} is {image.shape[1]}x{image.shape[0]}, training frames were "
                               f"{self.policy.image_size[1]}x{self.policy.image_size[0]}")
        wire_state = np.asarray(payload["state"], dtype=np.float32).reshape(-1)
        if wire_state.shape != (STATE_DIM,) or not np.all(np.isfinite(wire_state)):
            raise ValueError(f"state must be {STATE_DIM} finite values, got {wire_state.shape}")
        state = flip_gripper(wire_state) if self.args.flip_gripper else wire_state
        instruction = str(payload.get("instruction", "")).strip()
        task = self.fixed_task or instruction
        if self.fixed_task and instruction and instruction.lower() != self.fixed_task.lower():
            self.warn_once(f"task_{instruction}", f"karma's instruction {instruction!r} is ignored; the policy gets {self.fixed_task!r}")
        if not task:
            raise ValueError("empty instruction")

        with self.lock:
            started = time.perf_counter()
            actions = self.policy.infer(images, state, task)
            dt_ms = (time.perf_counter() - started) * 1000
            self.requests += 1
            index = self.requests
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != STATE_DIM or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"policy returned an invalid chunk {actions.shape}")
        wire_actions = flip_gripper(actions) if self.args.flip_gripper else actions
        if self.args.save_frames and index % 30 == 1:
            from PIL import Image

            for wire, image in images.items():
                Image.fromarray(image).save(self.args.save_frames / f"req{index:05d}_{wire}.jpg", quality=90)
        if index == 1 or index % 50 == 0:
            g = wire_state[GRIPPER_COLUMNS]
            print(f"[{time.strftime('%H:%M:%S')}] request {index}: {dt_ms:.0f} ms, task {task!r}, "
                  f"wire gripper L/R {g[0]:.2f}/{g[1]:.2f} (1=open), first action gripper "
                  f"{wire_actions[0, 6]:.2f}/{wire_actions[0, 13]:.2f}", flush=True)
        return {"actions": wire_actions.astype(np.float64).tolist(), "dt_ms": dt_ms}


def make_handler(server: Server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: dict | str) -> None:
            data = (json.dumps(body) if isinstance(body, dict) else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json" if isinstance(body, dict) else "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") != "/act":
                return self._send(404, "use /act")
            self._send(200, server.health())

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/act":
                return self._send(404, "use /act")
            try:
                payload = loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, server.act(payload))
            except ValueError as err:
                print(f"bad request: {err}", flush=True)
                self._send(400, str(err))
            except Exception as err:  # noqa: BLE001 - report it to karma instead of dropping the connection
                print(f"inference failed: {type(err).__name__}: {err}", flush=True)
                self._send(500, f"{type(err).__name__}: {err}")

        def log_message(self, *args):  # the per-request line above is enough
            pass

    return Handler


def main(args: Args) -> None:
    if not args.mock and args.checkpoint is None:
        sys.exit("--checkpoint is required (or --mock)")
    server = Server(args)
    h, w = server.policy.image_size or (480, 848)
    t0 = time.perf_counter()
    blank = np.zeros((h, w, 3), np.uint8)
    chunk = server.policy.infer({k: blank for k in WIRE_CAMERAS}, np.zeros(STATE_DIM, np.float32), server.fixed_task or "warm up")
    print(f"warm-up: chunk {np.asarray(chunk).shape} in {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)
    if not args.mock:
        torch = server.policy.torch
        if args.device == "cuda":
            print(f"GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB peak", flush=True)
    print(f"task: {server.fixed_task!r}" if server.fixed_task else "task: karma's instruction, as typed", flush=True)
    print(f"karma: --server http://<this host>:{args.port} --norm-tag {args.norm_tag}", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(server))
    print(f"serving {server.policy.info.get('backend')} on http://{args.host}:{args.port}/act", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(tyro.cli(Args))
