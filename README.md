# dijitter

Making VLA robot actions smooth. First experiment: π0.5 + ACG on YAM (sim).

Fine-tune π0.5 (LoRA) on bimanual YAM data from the ABC simulator, then add
**Action Coherence Guidance (ACG)** to its denoising loop and measure whether motion gets
smoother (JerkRMS, ATV) without losing success. Simulation only; real-robot work comes later
in a separate repo.

Progress and plan: [TODO.md](TODO.md).

## Layout

```
dijitter/
├── README.md, TODO.md
├── pyproject.toml, uv.lock        laptop env (CPU torch): sim, viser, LeRobot conversion
├── scripts/
│   ├── abc_to_lerobot.py          ABC sim episodes -> LeRobot v3.0 (keys match openpi pi05_yam)
│   ├── check_lerobot.py           verify a converted dataset against its source episodes
│   ├── metrics.py                 JerkRMS / ATV (trajectories) + intra-chunk and chunk-switch smoothness
│   ├── eval_sim.py                closed-loop sim eval against a policy server; logs success, latency, chunks
│   └── mock_policy_server.py      GPU-free stand-in server (hold / demo replay / noisy) for testing eval_sim
├── third_party/                   git submodules (pinned)
│   ├── abc/                       amazon-far/abc: MuJoCo YAM sim, sim data, viser viewers
│   ├── yam-abc-reproduce/         i2rt YAM pipeline + vendored openpi (π0.5, pi05_yam_lora)
│   └── ACG/                       DAVIAN-Robotics/ACG: reference ACG implementation (GR00T)
├── data/                          (gitignored) LeRobot datasets, data/lerobot/<repo_id>
└── outputs/                       (gitignored) checks, eval logs, videos
```

Where things live:

| What | Path |
|---|---|
| Sim + episode viewer | `third_party/abc/viz_episode.py`, `third_party/abc/abc_sim/` |
| Sim data / assets (downloaded, gitignored there) | `third_party/abc/cache/`, `third_party/abc/abc_sim/models/assets*` |
| π0.5 model (ACG goes in `sample_actions`) | `third_party/yam-abc-reproduce/third_party/policy/openpi/src/openpi/models/pi0.py` |
| π0.5 YAM configs | `.../openpi/src/openpi/training/config.py` (`pi05_yam_lora_inhand`, ours) |
| ACG tests | `.../openpi/src/openpi/models/acg_test.py` |
| Policy server (`--acg-scale`) | `third_party/yam-abc-reproduce/yam_abc_reproduce/deploy/servers/openpi_server.py` |
| ACG reference | `third_party/ACG/libs/robomimic/robomimic/algo/guidance/acg.py` |

## Setup

```bash
git clone --recurse-submodules https://github.com/4rhan/dijitter.git
cd dijitter
# yam-abc-reproduce's own i2rt submodule uses SSH; use HTTPS instead:
git -C third_party/yam-abc-reproduce -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init
UV_HTTP_TIMEOUT=300 uv sync          # laptop env (CPU)
source .venv/bin/activate
```

## Laptop (CPU): sim data and viewing

```bash
cd third_party/abc
python prepare.py --sim-data-list                                         # list tasks + sizes
python prepare.py --sim-data sim_inhand_transfer_the_item_to_other_side   # ~1.2 GB data + ~7.7 GB object assets
python viz_episode.py --root cache/train_sim --port 8080                  # http://localhost:8080
cd ../..
```

## Convert to LeRobot (laptop or GPU box)

```bash
python scripts/abc_to_lerobot.py --repo-id abc_inhand_transfer --max-episodes 100
python scripts/check_lerobot.py  --repo-id abc_inhand_transfer            # expect 100/100 OK
```

The output goes to `data/lerobot/abc_inhand_transfer`. Actions stay absolute; the openpi YAM config
converts arm joints to deltas itself.

## Metrics and the eval loop (laptop, no GPU)

```bash
python scripts/metrics.py --max-episodes 200                    # jitter of the recorded demos

# terminal 1: a stand-in server that replays one held-out demo
python scripts/mock_policy_server.py --mode replay --episode-dir third_party/abc/cache/val_sim/<episode_dir>
# terminal 2: run the sim against it (same client as for π0.5)
python scripts/eval_sim.py --replay-episode third_party/abc/cache/val_sim/<episode_dir> --num-episodes 1 --tag mock_replay --save-video
```

`--mode noisy` adds noise to every action: the metrics should clearly rise relative to `replay`.

## ACG

`scripts/acg_pi05.py` follows the paper's Algorithm 1 on LeRobot π0.5. Each denoising step runs a second pass in which
the chosen layers stop action tokens attending to each other (they still see image, text and state, as in GR00T's
separate cross-attention blocks), then uses `v = (1 + λ) * v - λ * v_incoherent`. On the server, `--acg-scale` is λ: 0
reproduces the baseline, and the authors' released setting is λ = 2 (the paper's text says 3). Eval runs named `*_acg<s>`
used λ = s - 1; `*_lambda<λ>` used the literal Eq. 11 mask, which also blinds those layers; `*_lambda<λ>_keepprefix` is
the current version. By default the last half of the action expert's layers are perturbed.

## GPU box: LoRA fine-tune π0.5

Run from `third_party/yam-abc-reproduce`, with that repo's own env (`uv sync --extra gui --extra convert --group openpi`).
See TODO.md (Day 2) for the steps; commands are filled in there as they are verified.
