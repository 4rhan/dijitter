# π0.5 + ACG on YAM (sim) — TODO

Scope: YAM in simulation only (ABC MuJoCo sim), LoRA fine-tune only (`pi05_yam_lora`).
Real robot comes later, in the `karma` repo.

Layout: `third_party/abc` (sim + data) · `third_party/yam-abc-reproduce` (openpi π0.5) · `third_party/ACG` (reference) · `scripts/` (our code) · `data/`, `outputs/` (gitignored)

## Day 1: laptop setup (2026-10-29)
- [x] Clone the repos and set up the CPU env
- [x] Download sim data: `sim_inhand_transfer_the_item_to_other_side` (1009 train / 52 val)
- [x] Replay episodes in viser (`viz_episode.py`)
- [x] Write the ABC → LeRobot converter (`scripts/abc_to_lerobot.py`) and checker (`scripts/check_lerobot.py`)
- [x] Test conversion on 2 episodes: state/action exact, pixel err ~2/255
- [x] Organise into one repo with `third_party/` submodules
- [ ] (optional) `viz_episode.py --mode physics`: confirm recorded actions re-simulate the demo
- [ ] Write the sim eval client `scripts/eval_pi05.py`: ABC sim (CPU) ↔ openpi websocket server; logs success, latency, every action chunk
- [ ] Write jitter metrics `scripts/metrics.py`: JerkRMS + ATV from the logged actions
- [ ] Add ACG to π0.5 (openpi `src/openpi/models/pi0.py`, which holds π0 and π0.5; we use the `pi05=True` path) `sample_actions()`: second pass with action↔action attention masked (each action token sees prefix + itself), `v = v_cond + (s-1)(v_cond - v_incoh)`, `acg_scale` flag (1.0 = off)
- [ ] Add a π0.5 LoRA config (based on `pi05_yam_lora`) pointing at `abc_inhand_transfer` (repo-id, prompt, camera keys)

## Day 2: GPU box
- [ ] Clone `yam-abc-reproduce` (HTTPS submodule rewrite), `uv sync --extra gui --extra convert --group openpi`
- [ ] Download sim data on the box (`prepare.py --sim-data sim_inhand_transfer_the_item_to_other_side`)
- [ ] Convert 100 episodes → `data/lerobot/abc_inhand_transfer`, then `check_lerobot.py` (expect 100/100 OK)
- [ ] Point openpi at it (`HF_LEROBOT_HOME=$PWD/data/lerobot`)
- [ ] `compute_norm_stats.py --config-name pi05_yam_lora`
- [ ] LoRA fine-tune, ~10k steps (`train.py pi05_yam_lora --exp-name=inhand_sim`)
- [ ] Serve the checkpoint (`serve_policy.py policy:checkpoint ...`)

## Day 3+: experiments
- [ ] Baseline: success rate + latency over N seeds, ACG off
- [ ] Baseline jitter: JerkRMS / ATV
- [ ] ACG on: sweep scale (e.g. 1.5, 2, 3) and perturbed-layer choice
- [ ] Compare smoothness vs success (want: smoother, same success)
- [ ] Open-loop smoothness on the 52 held-out `val_sim` episodes (ACG on/off)
- [ ] If ACG looks wrong: unit-test the mask (incoherent pass ≠ normal pass; scale=1 == baseline exactly)
- [ ] Stress tests: lighting, camera pose, object position (randomization in `abc_sim`) — ACG on/off
- [ ] If baseline success is low: convert more episodes and resume training
