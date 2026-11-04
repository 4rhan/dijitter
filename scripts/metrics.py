"""Action smoothness as defined in the ACG paper (arXiv 2510.22201, Sec. V-C, Eqs. 12-13), plus a CLI.

ATV:     (1 / (M (T - 1))) sum_t sum_j |a^j_{t+1} - a^j_t|, on the actions a. Eq. 12 has no time step,
         so this is per step, although the paper labels its column rad/s.
JerkRMS: root mean square over time of the third derivative of the motor angles s, per motor. The
         paper does not say how it differentiates; here it is the third finite difference at 30 Hz.

As in the paper: both are averaged across all motors (14 here, grippers included) and use only the
first 64 timesteps, because later trajectories diverge depending on success or failure. Note that
the YAM gripper values run 0-1 rather than in radians.

chunk_metrics (intra-chunk and chunk-switch numbers) is not from the paper: it is a diagnostic of ours.

Usage:
  python scripts/metrics.py --max-episodes 200          # the training demos
  python scripts/metrics.py --eval-root outputs/eval    # recompute saved eval runs from their episode files
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
WINDOW = 64  # timesteps the paper measures over


def jerk_rms(traj: np.ndarray, fps: float = FPS) -> float:
    """traj: (T, D). RMS of the third derivative over time, per dim, then averaged over dims."""
    if len(traj) < 4:
        return float("nan")
    jerk = np.diff(traj, n=3, axis=0) * fps**3
    return float(np.sqrt((jerk**2).mean(axis=0)).mean())


def atv(actions: np.ndarray) -> float:
    """actions: (T, D). Mean |a_{t+1} - a_t| over time and dims."""
    if len(actions) < 2:
        return float("nan")
    return float(np.abs(np.diff(actions, axis=0)).mean())


def trajectory_metrics(actions: np.ndarray, states: np.ndarray, fps: float = FPS) -> dict[str, float]:
    """The paper's metrics. actions, states: (T, D), aligned in time."""
    return {"jerk_rms": jerk_rms(states[:WINDOW], fps), "atv": atv(actions[:WINDOW])}


def chunk_metrics(
    chunks: np.ndarray, executed_steps: int, include_grippers: bool = False, fps: float = FPS
) -> dict[str, float]:
    """Our own diagnostic, not a paper metric: smoothness of a policy's predicted action chunks.

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
    eval_root: Path | None = None
    """Instead of demos: recompute every eval run under this directory from its episode_*.npz."""
    out: Path = REPO / "outputs" / "metrics"


def eval_table(root: Path) -> None:
    print(f"{'run':<30}{'episodes':>9}{'ATV':>10}{'JerkRMS':>10}")
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = []
        for path in run.glob("episode_*.npz"):
            with np.load(path) as ep:
                rows.append(trajectory_metrics(ep["executed"], ep["states"]))
        if rows:
            print(f"{run.name:<30}{len(rows):>9}{np.mean([r['atv'] for r in rows]):>10.5f}"
                  f"{np.mean([r['jerk_rms'] for r in rows]):>10.1f}")


def main(args: Args) -> None:
    if args.eval_root is not None:
        eval_table(args.eval_root)
        return
    episodes = sorted(p.parent for p in args.src.glob("episode_*/states_actions.bin"))[: args.max_episodes]
    if not episodes:
        raise SystemExit(f"no episodes under {args.src}")
    rows = []
    for ep in episodes:
        _, state, action = load_episode(ep)
        rows.append({"episode": ep.name, "T": len(action), **trajectory_metrics(action, state)})

    jerk = np.array([r["jerk_rms"] for r in rows])
    tv = np.array([r["atv"] for r in rows])
    summary = {
        "source": str(args.src),
        "episodes": len(rows),
        "jerk_rms_mean": float(jerk.mean()),
        "jerk_rms_median": float(np.median(jerk)),
        "atv_mean": float(tv.mean()),
        "atv_median": float(np.median(tv)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "demos.json").write_text(json.dumps({"summary": summary, "episodes": rows}, indent=2))

    print(f"{len(rows)} episodes, first {WINDOW} timesteps, all motors")
    print(f"  JerkRMS (states)  mean {summary['jerk_rms_mean']:.2f}  median {summary['jerk_rms_median']:.2f}")
    print(f"  ATV (actions)     mean {summary['atv_mean']:.5f}  median {summary['atv_median']:.5f}")
    print(f"  per-episode results -> {args.out / 'demos.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
