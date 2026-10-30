# π0.5 + ACG on YAM (sim) — TODO

Scope: YAM in simulation only (ABC MuJoCo sim), LoRA fine-tune only.
Stack (from 2026-10-30): LeRobot 0.6.1 PyTorch π0.5, starting from `lerobot/pi05_base` (already on the GPU box). No other checkpoints, no LIBERO.
The Day 1 JAX/openpi ACG edits (in the `yam-abc-reproduce` submodule) are dropped; ACG gets ported to LeRobot `modeling_pi05.py`.
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
- [ ] ~~JAX ACG tests, fork `yam-abc-reproduce`~~ (dropped: switched to LeRobot)
- [ ] (optional) `viz_episode.py --mode physics`: confirm recorded actions re-simulate the demo
- [ ] (later) ACG per-layer variant: perturb only middle layers like the paper (needs a per-layer mask in `gemma.py`)

## Day 2: GPU box (2026-10-30)
Box layout (same as expo-ft): code in `/home/sra/arhan/dijitter`; env, caches, data, outputs in `/usr/local/models/sra-arhan-dijitter` (`data/`, `outputs/` in the repo are symlinks there). `source /usr/local/models/sra-arhan-dijitter/env.sh` first. `pi05_base` comes from the shared `/usr/local/models/hf-hub`.
- [x] Clone `dijitter` + `third_party/abc` on the box, make the SSD folders + `env.sh`
- [x] `pyproject.toml`: `cpu` / `gpu` torch extras, `lerobot[pi,peft]==0.6.1`, `openpi-client` from git
- [ ] Box env: `UV_PROJECT_ENVIRONMENT=$A/venv uv sync --extra gpu`; check torch sees the 4090 and `pi05_base` loads
- [ ] Sim data on the box (`prepare.py --sim-data sim_inhand_transfer_the_item_to_other_side`)
- [ ] Convert 100 episodes → `data/lerobot/abc_inhand_transfer`, then `check_lerobot.py` (re-check under LeRobot 0.6.1)
- [ ] Port ACG to LeRobot π0.5 (`denoise_step` patch from our repo, no LeRobot fork); test: scale 1.0 == baseline, 2.0 differs
- [ ] LoRA fine-tune `pi05_base` on the converted data (`lerobot-train --policy.type=pi05 --peft...`); needs ~22 GB GPU free (box is shared)
- [ ] Policy server wrapping LeRobot π0.5 (same websocket protocol, so `eval_sim.py` is unchanged), with `--acg-scale`
- [ ] `eval_sim.py --tag baseline`, then with ACG

## Day 3+: experiments
- [ ] Baseline: success rate + latency over N seeds, ACG off
- [ ] Baseline jitter: JerkRMS / ATV
- [ ] ACG on: sweep scale (e.g. 1.5, 2, 3) and perturbed-layer choice
- [ ] Compare smoothness vs success (want: smoother, same success)
- [ ] Open-loop smoothness on the 52 held-out `val_sim` episodes (ACG on/off)
- [ ] If ACG looks wrong: unit-test the mask (incoherent pass ≠ normal pass; scale=1 == baseline exactly)
- [ ] Stress tests: lighting, camera pose, object position (randomization in `abc_sim`) — ACG on/off
- [ ] If baseline success is low: convert more episodes and resume training
