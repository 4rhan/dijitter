# π0.5 + ACG on YAM (sim) — TODO

Scope: YAM in simulation only (ABC MuJoCo sim), LoRA fine-tune only (`pi05_yam_lora`).
LIBERO (`pi05_libero`) is used only as an ACG sanity check on a pretrained checkpoint.
Real robot comes later, in the `karma` repo.

Layout: `third_party/abc` (sim + data) · `third_party/yam-abc-reproduce` (openpi π0.5) · `third_party/ACG` (reference) · `scripts/` (our code) · `data/`, `outputs/` (gitignored)

## Day 1: laptop setup (2026-10-29)
- [x] Clone the repos and set up the CPU env
- [x] Download sim data: `sim_inhand_transfer_the_item_to_other_side` (1009 train / 52 val)
- [x] Replay episodes in viser (`viz_episode.py`)
- [x] Write the ABC → LeRobot converter (`scripts/abc_to_lerobot.py`) and checker (`scripts/check_lerobot.py`)
- [x] Test conversion on 2 episodes: state/action exact, pixel err ~2/255
- [x] Organise into one repo with `third_party/` submodules
- [x] Write the sim eval client `scripts/eval_sim.py` (success, latency, every chunk, smoothness) and `scripts/mock_policy_server.py` (hold / replay / noisy)
- [x] Write jitter metrics `scripts/metrics.py`: JerkRMS + ATV on trajectories, plus per-chunk (intra-chunk JerkRMS/ATV, chunk-switch jump)
- [x] Add ACG to π0.5 `sample_actions()` (`openpi/models/pi0.py`, π0/π0.5 share the file): `acg_scale` kwarg (None = off), incoherent pass masks action↔action attention in all layers
- [x] Add `--acg-scale` / `--num-steps` to the YAM policy server (`yam_abc_reproduce/deploy/servers/openpi_server.py`)
- [x] Add π0.5 LoRA config `pi05_yam_lora_inhand` (repo `abc_inhand_transfer`, 10k steps)
- [x] ACG unit tests `openpi/models/acg_test.py` (mask pattern; scale 1.0 == baseline; scale 3.0 differs; works under jit)
- [x] Laptop env sync (`uv sync`), then on the laptop:
  - [x] `python scripts/metrics.py --max-episodes 200` → demo jitter reference: actions JerkRMS 312.5 / ATV 0.0064; states JerkRMS 140.7 / ATV 0.0060 (teleop commands are steppy)
  - [x] mock loop: `mock_policy_server.py --mode replay` + `eval_sim.py --replay-episode` → demo reproduces in sim, evaluator fires (success at step 291, 19 ms round trip)
  - [x] mock loop `--mode noisy` (0.01 rad): JerkRMS 297 → 1268, ATV 0.0061 → 0.0142, switch jump 0 → 0.011; `--mode hold` in random worlds: fails, truncates correctly
- [ ] → moved to Day 2: ACG tests, fork + push
- [ ] (optional) `viz_episode.py --mode physics`: confirm recorded actions re-simulate the demo
- [ ] (later) ACG per-layer variant: perturb only middle layers like the paper (needs a per-layer mask in `gemma.py`)

## Day 2: GPU box (2026-10-30)
Get code on the box
- [ ] Copy the repo over (fork not pushed yet): `rsync -a --exclude data --exclude outputs --exclude .venv ~/post-vla/dijitter <box>:~/`
- [ ] In `third_party/yam-abc-reproduce`: `uv sync --extra gui --extra convert --group openpi`
- [ ] Run ACG tests: `uv run pytest third_party/policy/openpi/src/openpi/models/acg_test.py`

LIBERO sanity check: ACG on a known-good checkpoint (`pi05_libero`, no training)
- [ ] Clone the sim: `cd third_party/policy/openpi && git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero`
- [ ] Client venv (py3.8), per `examples/libero/README.md` "Without Docker"; headless box → `MUJOCO_GL=egl`
- [ ] Baseline server: `uv run python third_party/policy/openpi/scripts/serve_policy.py --env LIBERO` (pulls `gs://openpi-assets/checkpoints/pi05_libero`)
- [ ] Client: `python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 5` → note success (paper ~98%) + latency
- [ ] ACG server: same + `--acg-scale 2.0` (`serve_policy.py` now forwards `--acg-scale` / `--num-steps`); rerun client, compare success
- [ ] Pass: ACG success ≈ baseline, no crash, latency ~2× (two passes per denoising step)

YAM sim (data prep can run on CPU while LIBERO runs)
- [ ] Download sim data on the box (`prepare.py --sim-data sim_inhand_transfer_the_item_to_other_side`)
- [ ] Convert 100 episodes → `data/lerobot/abc_inhand_transfer`, then `check_lerobot.py` (expect 100/100 OK)
- [ ] Point openpi at it (`HF_LEROBOT_HOME=$PWD/data/lerobot`)
- [ ] `compute_norm_stats.py --config-name pi05_yam_lora_inhand` (also a dry run of the data pipeline)
- [ ] Stop the LIBERO server (frees GPU memory), then LoRA fine-tune 10k steps (`train.py pi05_yam_lora_inhand --exp-name=inhand_sim`), let it run overnight
- [ ] Serve: `openpi_server.py --config pi05_yam_lora_inhand --checkpoint <ckpt> --image-key-map top=image,left=left_wrist,right=right_wrist --flatten-prefix observation/ --state-key observation/state [--acg-scale 2.0]`
- [ ] Laptop: `eval_sim.py --host <box-ip> --tag baseline` (needs port 8000 reachable, e.g. `ssh -L 8000:localhost:8000 <box>`)
- [ ] Fork `yam-abc-reproduce`, repoint the submodule, push (then the box can `git pull` instead of rsync)

## Day 3+: experiments
- [ ] Baseline: success rate + latency over N seeds, ACG off
- [ ] Baseline jitter: JerkRMS / ATV
- [ ] ACG on: sweep scale (e.g. 1.5, 2, 3) and perturbed-layer choice
- [ ] Compare smoothness vs success (want: smoother, same success)
- [ ] Open-loop smoothness on the 52 held-out `val_sim` episodes (ACG on/off)
- [ ] If ACG looks wrong: unit-test the mask (incoherent pass ≠ normal pass; scale=1 == baseline exactly)
- [ ] Stress tests: lighting, camera pose, object position (randomization in `abc_sim`) — ACG on/off
- [ ] If baseline success is low: convert more episodes and resume training
