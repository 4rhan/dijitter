"""Smoothness metrics for 14-D YAM action trajectories, plus a CLI to compute them on ABC demo episodes.

JerkRMS: root-mean-square of the third finite difference of the joint trajectory, in rad/s^3
         (per joint, then averaged over joints). High = jerky.
ATV:     action total variation, the mean absolute step-to-step change per joint, in rad/step
         (sum_t |a_{t+1} - a_t| / (T - 1), averaged over joints). High = jittery or back-and-forth.

Both use the 12 arm joints only by default: the grippers are near-binary open/close signals
whose jumps would dominate and hide arm jitter. Pass --include-grippers to change that.

Usage (runs on recorded demos; uses the commanded actions by default):
  python scripts/metrics.py --max-episodes 200
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_to_lerobot import REPO, load_episode  # noqa: E402

ARM_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
FPS = 30.0


def jerk_rms(actions: np.ndarray, fps: float = FPS) -> float:
    """actions: (T, D). RMS of d^3a/dt^3 over time, averaged over dims."""
    if len(actions) < 4:
        return float("nan")
    jerk = np.diff(actions, n=3, axis=0) * fps**3
    return float(np.sqrt((jerk**2).mean(axis=0)).mean())


def atv(actions: np.ndarray) -> float:
    """actions: (T, D). Mean |a_{t+1} - a_t| over time, averaged over dims."""
    if len(actions) < 2:
        return float("nan")
    return float(np.abs(np.diff(actions, axis=0)).mean())


def trajectory_metrics(actions: np.ndarray, include_grippers: bool = False, fps: float = FPS) -> dict[str, float]:
    a = actions if include_grippers else actions[:, ARM_INDICES]
    return {"jerk_rms": jerk_rms(a, fps), "atv": atv(a)}


def chunk_metrics(
    chunks: np.ndarray, executed_steps: int, include_grippers: bool = False, fps: float = FPS
) -> dict[str, float]:
    """Smoothness of a policy's predicted action chunks.

    chunks: (N, H, D), the full chunk predicted at each inference call, in call order.
    executed_steps: how many actions of each chunk were executed before the next call.

    intra_*: JerkRMS / ATV inside each predicted chunk (what ACG targets), averaged over chunks.
    switch_jump: mean |first action of chunk k+1 - action it replaces in chunk k|, i.e. the
                 discontinuity introduced at every chunk switch (rad, averaged over joints).
    """
    c = chunks if include_grippers else chunks[..., ARM_INDICES]
    out = {
        "intra_jerk_rms": float(np.mean([jerk_rms(x, fps) for x in c])),
        "intra_atv": float(np.mean([atv(x) for x in c])),
    }
    if len(c) > 1 and executed_steps < c.shape[1]:
        out["switch_jump"] = float(np.abs(c[1:, 0] - c[:-1, executed_steps]).mean())
    return out


@dataclasses.dataclass
class Args:
    src: Path = REPO / "third_party" / "abc" / "cache" / "train_sim"
    """Directory of ABC episode_* folders."""
    max_episodes: int | None = None
    use_state: bool = False
    """Measure the recorded joint states instead of the commanded actions."""
    include_grippers: bool = False
    out: Path = REPO / "outputs" / "metrics"


def main(args: Args) -> None:
    episodes = sorted(p.parent for p in args.src.glob("episode_*/states_actions.bin"))[: args.max_episodes]
    if not episodes:
        raise SystemExit(f"no episodes under {args.src}")
    rows = []
    for ep in episodes:
        _, state, action = load_episode(ep)
        traj = state if args.use_state else action
        rows.append({"episode": ep.name, "T": len(traj), **trajectory_metrics(traj, args.include_grippers)})

    jerk = np.array([r["jerk_rms"] for r in rows])
    tv = np.array([r["atv"] for r in rows])
    summary = {
        "source": str(args.src),
        "signal": "state" if args.use_state else "action",
        "include_grippers": args.include_grippers,
        "episodes": len(rows),
        "jerk_rms_mean": float(jerk.mean()),
        "jerk_rms_median": float(np.median(jerk)),
        "atv_mean": float(tv.mean()),
        "atv_median": float(np.median(tv)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"demos_{summary['signal']}{'_grip' if args.include_grippers else ''}"
    (args.out / f"{name}.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2))

    print(f"{len(rows)} episodes, signal={summary['signal']}, grippers={'yes' if args.include_grippers else 'no'}")
    print(f"  JerkRMS  mean {summary['jerk_rms_mean']:.2f}  median {summary['jerk_rms_median']:.2f}  rad/s^3")
    print(f"  ATV      mean {summary['atv_mean']:.5f}  median {summary['atv_median']:.5f}  rad/step")
    print(f"  per-episode results -> {args.out / (name + '.json')}")


if __name__ == "__main__":
    main(tyro.cli(Args))
