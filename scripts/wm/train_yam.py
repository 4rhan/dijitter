"""Fine-tune Ctrl-World (github.com/Robert-gyj/Ctrl-World) on YAM sim data from scripts/wm/abc_to_ctrlworld.py.

Starts from the released DROID checkpoint (yjguo/Ctrl-World): every weight is kept except the action
projector's first layer, which is re-initialised for the YAM's 14-D joint state (DROID: 7-D pose).

  python scripts/wm/train_yam.py --ctrlworld ~/Ctrl-World --models ~/wm_models --data ~/wm_data --out ~/wm_runs/yam \
      --steps 30000 --batch 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrlworld", type=Path, required=True, help="Ctrl-World repo checkout")
    ap.add_argument("--models", type=Path, required=True, help="folder with stable-video-diffusion-img2vid, clip-vit-base-patch32, Ctrl-World")
    ap.add_argument("--data", type=Path, required=True, help="folder holding <name>/ and meta/<name>/")
    ap.add_argument("--name", default="yam")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--from-scratch-svd", action="store_true", help="start from plain SVD, not the DROID Ctrl-World weights")
    a = ap.parse_args()

    repo = a.ctrlworld.expanduser().resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from config import wm_args
    from models.ctrl_world import CrtlWorld

    import scripts.train_wm as tw  # noqa: E402  (Ctrl-World's own training loop)

    args = wm_args()
    m = a.models.expanduser()
    args.svd_model_path = str(m / "stable-video-diffusion-img2vid")
    args.clip_model_path = str(m / "clip-vit-base-patch32")
    args.dataset_root_path = str(a.data.expanduser())
    args.dataset_meta_info_path = str(a.data.expanduser() / "meta")
    args.dataset_names = args.dataset_cfgs = a.name
    args.prob = [1.0]
    args.action_dim = 14
    args.tag = a.name
    args.output_dir = str(a.out.expanduser())
    args.wandb_run_name = a.name
    args.max_train_steps = a.steps
    args.train_batch_size = a.batch
    args.learning_rate = a.lr
    args.checkpointing_steps = a.save_every
    args.validation_steps = a.val_every
    args.video_num = 2
    os.makedirs(args.output_dir, exist_ok=True)

    resume = sorted(Path(args.output_dir).glob("checkpoint-*.pt"), key=lambda p: int(p.stem.split("-")[1]))
    if resume:
        args.ckpt_path = str(resume[-1])
        print(f"resuming weights from {resume[-1]} (the step counter restarts at 0)", flush=True)
    else:
        init = Path(args.output_dir) / "init_from_droid.pt"
        if not init.exists():
            model = CrtlWorld(args)
            sd = model.state_dict()
            if not a.from_scratch_svd:
                cands = sorted((m / "Ctrl-World").glob("*.pt")) + sorted((m / "Ctrl-World").glob("*.pth"))
                if not cands:
                    raise SystemExit(f"no Ctrl-World checkpoint (*.pt) under {m / 'Ctrl-World'}")
                droid = torch.load(cands[0], map_location="cpu")
                kept = skipped = 0
                for k, v in droid.items():
                    if k in sd and sd[k].shape == v.shape:
                        sd[k] = v
                        kept += 1
                    else:
                        skipped += 1
                        print(f"  re-initialised: {k} {tuple(v.shape)} -> {tuple(sd[k].shape) if k in sd else 'absent'}")
                print(f"loaded {kept} tensors from {cands[0].name}, re-initialised {skipped}", flush=True)
            torch.save(sd, init)
            del model
        args.ckpt_path = str(init)
    tw.main(args)


if __name__ == "__main__":
    main()
