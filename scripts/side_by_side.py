"""Stack eval videos of the same worlds from several runs, one row per run, labelled with the outcome.

Each run is an eval_sim.py output dir made with --save-video. Episodes are matched by world seed, so
re-run the same seeds in every run (eval_sim.py --seeds ...). A run that ends early holds its last
frame, with its label showing SUCCESS or FAIL and the step it ended on.

  python scripts/side_by_side.py --run baseline=outputs/eval/vid_baseline --run acg=outputs/eval/vid_acg \
      --out outputs/eval/side_by_side/acg_vs_baseline

Writes seed_<s>.mp4 per world and all.mp4 with every world one after another.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREEN, RED, GREY = (40, 170, 70), (200, 50, 50), (60, 60, 60)


def font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def load_run(spec: str) -> tuple[str, Path, dict[int, dict]]:
    name, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"--run needs NAME=DIR, got {spec!r}")
    d = Path(path)
    eps = json.loads((d / "summary.json").read_text())["per_episode"]
    return name, d, {e["seed"]: e for e in eps}


def label_bar(width: int, text: str, color, height: int) -> np.ndarray:
    img = Image.new("RGB", (width, height), color)
    ImageDraw.Draw(img).text((10, height // 2), text, fill=(255, 255, 255), font=font(height * 3 // 5), anchor="lm")
    return np.asarray(img)


def make_video(rows, seed: int, writer_paths: list, scale: int, fps: int, bar: int):
    """rows: [(name, frames, episode dict)]. Appends frames to every writer in writer_paths."""
    n = max(len(f) for _, f, _ in rows)
    h, w = rows[0][1][0].shape[:2]
    w2, h2 = w * scale, h * scale
    for t in range(n):
        parts = []
        for name, frames, ep in rows:
            done = t >= len(frames) - 1
            frame = frames[min(t, len(frames) - 1)]
            if scale != 1:
                frame = np.asarray(Image.fromarray(frame).resize((w2, h2), Image.NEAREST))
            if done:
                verdict = "SUCCESS" if ep["success"] else "FAIL"
                color = GREEN if ep["success"] else RED
                text = f"{name}   {verdict} at step {ep['steps']}"
            else:
                color, text = GREY, f"{name}   step {t + 1}"
            parts += [label_bar(w2, text, color, bar), frame]
        parts.insert(0, label_bar(w2, f"world seed {seed}", (20, 20, 20), bar))
        img = np.concatenate(parts, axis=0)
        for wr in writer_paths:
            wr.append_data(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=EVAL_DIR, one per row (2 or more)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="*", help="only these seeds (default: every seed all runs share)")
    ap.add_argument("--scale", type=int, default=2, help="upscale the 168-px camera strips (default 2)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bar", type=int, default=40, help="label bar height in pixels")
    a = ap.parse_args()

    runs = [load_run(s) for s in a.run]
    shared = set.intersection(*(set(eps) for _, _, eps in runs))
    seeds = [s for s in (a.seeds or sorted(shared)) if s in shared]
    if not seeds:
        raise SystemExit("no world seed is in every run; re-run the same seeds with eval_sim.py --seeds")
    a.out.mkdir(parents=True, exist_ok=True)
    all_writer = imageio.get_writer(a.out / "all.mp4", fps=a.fps, macro_block_size=8)
    try:
        for seed in seeds:
            rows = []
            for name, d, eps in runs:
                ep = eps[seed]
                video = d / f"episode_{ep['episode']}.mp4"
                if not video.exists():
                    raise SystemExit(f"{video} missing: run eval_sim.py with --save-video")
                rows.append((name, [np.asarray(f) for f in imageio.mimread(video, memtest=False)], ep))
            with imageio.get_writer(a.out / f"seed_{seed}.mp4", fps=a.fps, macro_block_size=8) as one:
                make_video(rows, seed, [one, all_writer], a.scale, a.fps, a.bar)
            print(f"seed {seed}: " + "  ".join(f"{n} {'SUCCESS' if e['success'] else 'FAIL'} ({e['steps']} steps)"
                                               for n, _, e in rows), flush=True)
    finally:
        all_writer.close()
    print(f"wrote {len(seeds)} worlds to {a.out}/ (seed_<s>.mp4 and all.mp4)")


if __name__ == "__main__":
    main()
