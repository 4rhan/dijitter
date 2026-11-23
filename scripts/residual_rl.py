"""Residual RL on a frozen pi0.5 checkpoint in the batched ABC sim (MJWarp, GPU physics + rendering).

    final action = pi0.5 action (frozen, chunk-blended)  +  scale * tanh(residual)

The residual is a small MLP trained with PPO. It sees privileged sim state (arm state, object pose,
the base action for this step, the step's position inside the chunk), which only the sim has: it is a
sim teacher, to be distilled into pi0.5 later (DAgger). pi0.5 sees only what it saw in training
(3 cameras + arm state + prompt) and is queried once per `execute_steps` for every world in one batch.

Reward per step: progress_weight * (task progress gain) + success_bonus on first success
                 - action_penalty * |residual / scale|^2
where progress is the in-hand evaluator's own reward (how far the object travelled to the other side,
0..0.99, 1 at success).

Batching: every world in an MJWarp batch must compile to the same model, and each in-hand world has its
own object mesh and scale, so a batch holds `copies` copies of ONE world. `envs` batches (different
worlds) run per iteration, so an iteration covers envs x copies rollouts of `envs` distinct worlds.

  python scripts/residual_rl.py train --out outputs/rl/res_easy15k            # PPO, resumes from out/
  python scripts/residual_rl.py train --dry-run                               # 2 tiny iterations
  python scripts/residual_rl.py eval --residual outputs/rl/res_easy15k/last.pt  # test worlds, residual on
  python scripts/residual_rl.py eval                                          # test worlds, pi0.5 alone
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abc_to_lerobot import CAMERA_KEYS, REPO  # noqa: E402

ABC = REPO / "third_party" / "abc"
sys.path.insert(0, str(ABC))
from eval_sim import TRAIN_PROMPT, blended_action, demo_start_state  # noqa: E402

EASY_CATEGORIES = ("can", "rolling_pin", "pizza_cutter", "whisk")
GRIPPERS: tuple[int, ...] = (6, 13)  # replaced by the sim's own indices in main()


@dataclasses.dataclass
class Args:
    mode: tyro.conf.Positional[Literal["train", "eval"]]
    checkpoint: Path = REPO / "outputs/train/pi05_easy_lora/checkpoints/015000/pretrained_model"
    repo_id: str = "abc_easy"
    root: Path = REPO / "data" / "lerobot"
    categories: tuple[str, ...] = EASY_CATEGORIES
    sides: tuple[str, ...] = ("left",)
    out: Path = REPO / "outputs" / "rl" / "res_easy15k"
    residual: Path | None = None
    """eval: residual checkpoint (.pt); None = pi0.5 alone."""

    # worlds
    train_seed_start: int = 10_000
    """Training worlds come from here on, disjoint from the eval worlds (seeds 0..~240)."""
    eval_seed_start: int = 0
    eval_worlds: int = 50
    envs: int = 8
    """Distinct worlds per iteration (one MJWarp batch each)."""
    copies: int = 16
    """Rollouts per world per iteration."""
    max_steps: int = 900
    execute_steps: int = 25
    blend_steps: int = 10
    infer_batch: int = 32
    """pi0.5 micro-batch size."""

    # residual
    arm_scale: float = 0.05
    gripper_scale: float = 0.1
    hidden: int = 256
    init_log_std: float = -0.5

    # reward
    progress_weight: float = 10.0
    success_bonus: float = 5.0
    action_penalty: float = 0.01

    # PPO
    iterations: int = 400
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 8192
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    save_every: int = 5
    seed: int = 0
    dry_run: bool = False


# --------------------------------------------------------------------------- residual policy

class RunningNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def update(self, x: torch.Tensor) -> None:
        m, v, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = m - self.mean, self.count + n
        self.mean += delta * n / total
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / total) / total
        self.count = total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10, 10)


class Residual(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, init_log_std: float):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        nn.init.zeros_(self.pi[-1].weight)  # starts as "no correction": pi0.5 alone
        nn.init.zeros_(self.pi[-1].bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs: torch.Tensor):
        mu = self.pi(self.norm(obs))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(self.norm(obs)).squeeze(-1)


# --------------------------------------------------------------------------- sim

def make_env(copies: int):
    import abc_sim

    os.environ["DEPLOY_INIT_Q"] = json.dumps(demo_start_state().tolist())
    spec = abc_sim.get_task_spec("inhand_transfer")
    return abc_sim.make_batched_env(spec.name, num_worlds=copies, prompt=TRAIN_PROMPT,
                                    camera_height=168, camera_width=224, camera_gpu_id=0)


def reset_copies(env, seed: int):
    """Reset every world of `env` to world `seed`: one CPU reset, replicated (BatchedWarpYAMEnv.reset
    would rebuild the same scene once per copy)."""
    from abc_sim.batched_env import PerWorldEvaluators

    base = env.base_env
    base.forget_arm_state()
    base.reset(seed=seed, randomize=True)
    n = env.num_worlds
    env.seeds = [seed] * n
    env.randomization = [base._last_randomization] * n
    env.prompts = [TRAIN_PROMPT] * n
    snap = env._snapshot()
    env._bind_model()
    stacked = {k: np.repeat(v[None], n, axis=0) for k, v in snap.items() if k != "signature"}
    rt = env._runtime
    rt.reset_from_mujoco()
    rt.sync_model_from_mujoco(stacked)
    rt.load_state_batch(qpos=stacked["qpos"], qvel=stacked["qvel"], ctrl=stacked["ctrl"],
                        act=stacked["act"] if env.model.na else None,
                        mocap_pos=stacked["mocap_pos"] if env.model.nmocap else None,
                        mocap_quat=stacked["mocap_quat"] if env.model.nmocap else None,
                        time=stacked["time"])
    rt.forward()
    env.qpos = stacked["qpos"]
    if isinstance(env._evaluator, PerWorldEvaluators):
        env._evaluator.reset(env.model, env.randomization)
        env.per_world_metric_keys = env._evaluator.per_world_keys(env.qpos)
    elif env._evaluator is not None:
        env._evaluator.reset(nworld=n)
    jnt = env.model.joint("task_object_joint").id
    return int(env.model.jnt_qposadr[jnt])


def select_seeds(env, start: int, count: int, categories, sides) -> list[int]:
    base, seeds, seed = env.base_env, [], start
    while len(seeds) < count:
        try:
            base.reset(seed=seed, randomize=True)
            if base._inhand_category in categories and base._inhand_side in sides:
                seeds.append(seed)
        except RuntimeError as e:  # RandomizationSamplingError: an unplaceable world, skip it
            print(f"seed {seed} skipped: {e}", flush=True)
        seed += 1
    return seeds


# --------------------------------------------------------------------------- pi0.5

class Pi05:
    def __init__(self, args: Args):
        from serve_pi05 import load

        ns = SimpleNamespace(checkpoint=args.checkpoint, repo_id=args.repo_id, root=args.root,
                             num_steps=None, device="cuda")
        self.policy, self.pre, self.post, self.cfg = load(ns)
        self.mb = args.infer_batch

    @torch.inference_mode()
    def chunks(self, images: dict[str, torch.Tensor], state: np.ndarray) -> np.ndarray:
        """images: camera -> (B, 3, H, W) uint8 CUDA; state (B, 14) -> (B, 50, 14) numpy."""
        out = []
        for i in range(0, state.shape[0], self.mb):
            sl = slice(i, i + self.mb)
            batch = {key: images[cam][sl].float() / 255.0 for cam, key in CAMERA_KEYS.items()}
            batch["observation.state"] = torch.as_tensor(state[sl], dtype=torch.float32, device="cuda")
            batch["task"] = [TRAIN_PROMPT] * batch["observation.state"].shape[0]
            out.append(self.post(self.policy.predict_action_chunk(self.pre(batch))).float().cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------- rollout

OBS_DIM = 14 + 7 + 14 + 1 + 1  # state, object pose, base action, chunk phase, time
ACT_DIM = 14


def rollout(args: Args, pi05: Pi05, envs, seeds: list[int], residual: Residual | None, stochastic: bool):
    """Run every env (one world each, `copies` copies) for one episode. Returns per-step arrays [T, B]."""
    B = len(envs) * args.copies
    scale = torch.full((ACT_DIM,), args.arm_scale, device="cuda")
    scale[list(GRIPPERS)] = args.gripper_scale
    adrs = [reset_copies(env, s) for env, s in zip(envs, seeds)]
    obs_l, raw_l, logp_l, rew_l, alive_l, val_l = [], [], [], [], [], []
    prev_prog = np.zeros(B, np.float32)
    done = np.zeros(B, bool)
    success = np.zeros(B, bool)
    steps_to_success = np.full(B, -1)
    chunk = prev = None
    used = 0
    tm = dict(render=0.0, infer=0.0, step=0.0, other=0.0)
    t_last = time.perf_counter()
    for t in range(args.max_steps):
        state = np.concatenate([env.state() for env in envs])
        k = t % args.execute_steps
        if k == 0:
            t1 = time.perf_counter(); tm["other"] += t1 - t_last
            cams = [env.render_cameras() for env in envs]
            images = {c: torch.cat([cam[c] for cam in cams]) for c in CAMERA_KEYS}
            torch.cuda.synchronize(); t2 = time.perf_counter(); tm["render"] += t2 - t1
            prev, used = chunk, args.execute_steps
            chunk = pi05.chunks(images, state)
            t_last = time.perf_counter(); tm["infer"] += t_last - t2
        base = np.stack([blended_action(chunk[b], k, None if prev is None else prev[b], used, args.blend_steps)
                         for b in range(B)])
        qpos = np.concatenate([env.qpos for env in envs])
        obj = np.stack([qpos[e * args.copies:(e + 1) * args.copies, a:a + 7] for e, a in enumerate(adrs)]).reshape(B, 7)
        o = np.concatenate([state, obj, base, np.full((B, 1), k / args.execute_steps),
                            np.full((B, 1), t / args.max_steps)], axis=1).astype(np.float32)
        o_t = torch.as_tensor(o, device="cuda")
        if residual is None:
            raw = torch.zeros(B, ACT_DIM, device="cuda")
            logp = val = torch.zeros(B, device="cuda")
        else:
            with torch.no_grad():
                d = residual.dist(o_t)
                raw = d.sample() if stochastic else d.mean
                logp, val = d.log_prob(raw).sum(-1), residual.value(o_t)
        delta = (torch.tanh(raw) * scale).cpu().numpy()
        act = base + delta
        act[:, list(GRIPPERS)] = np.clip(act[:, list(GRIPPERS)], 0.0, 1.0)
        prog = np.zeros(B, np.float32)
        succ_now = np.zeros(B, bool)
        t1 = time.perf_counter(); tm["other"] += t1 - t_last
        for e, env in enumerate(envs):
            sl = slice(e * args.copies, (e + 1) * args.copies)
            res = env.step(act[sl])
            prog[sl] = res.reward
            succ_now[sl] = np.asarray(res.metrics.get("ever_success", res.success), bool)
        t_last = time.perf_counter(); tm["step"] += t_last - t1
        new_success = succ_now & ~success
        r = (args.progress_weight * (prog - prev_prog) + args.success_bonus * new_success
             - args.action_penalty * (np.tanh(raw.cpu().numpy()) ** 2).sum(-1))
        alive = ~done
        obs_l.append(o); raw_l.append(raw.cpu().numpy()); logp_l.append(logp.cpu().numpy())
        val_l.append(val.cpu().numpy()); rew_l.append(np.where(alive, r, 0.0).astype(np.float32))
        alive_l.append(alive)
        steps_to_success[new_success & alive] = t + 1
        success |= succ_now
        prev_prog = prog
        done |= success
        if done.all():
            break
    with torch.no_grad():
        last_v = (residual.value(o_t).cpu().numpy() if residual is not None else np.zeros(B))
    return dict(obs=np.stack(obs_l), raw=np.stack(raw_l), logp=np.stack(logp_l), rew=np.stack(rew_l),
                alive=np.stack(alive_l), val=np.stack(val_l), last_v=np.where(success, 0.0, last_v),
                success=success, steps=steps_to_success, final_progress=prev_prog,
                steps_run=t + 1, timing={k: round(v, 1) for k, v in tm.items()})


def gae(ro, gamma, lam):
    rew, val, alive = ro["rew"], ro["val"], ro["alive"].astype(np.float32)
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    last = np.zeros(rew.shape[1], np.float32)
    for t in reversed(range(T)):
        nxt_v = ro["last_v"] if t == T - 1 else val[t + 1] * alive[t + 1]
        nxt_alive = np.ones_like(last) if t == T - 1 else alive[t + 1]
        # a world that just succeeded has alive[t+1] = 0: no bootstrap past the success
        delta = rew[t] + gamma * nxt_v - val[t]
        last = delta + gamma * lam * nxt_alive * last
        adv[t] = last * alive[t]
    return adv, adv + val


def ppo_update(args: Args, res: Residual, opt, ro):
    adv, ret = gae(ro, args.gamma, args.lam)
    m = ro["alive"].reshape(-1)
    obs = torch.as_tensor(ro["obs"].reshape(-1, OBS_DIM)[m], device="cuda")
    raw = torch.as_tensor(ro["raw"].reshape(-1, ACT_DIM)[m], device="cuda")
    old_logp = torch.as_tensor(ro["logp"].reshape(-1)[m], device="cuda")
    adv = torch.as_tensor(adv.reshape(-1)[m], device="cuda")
    ret = torch.as_tensor(ret.reshape(-1)[m], device="cuda")
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    res.norm.update(obs)
    with torch.no_grad():  # the behaviour policy under the updated input normalisation
        old_logp = torch.cat([res.dist(obs[i:i + 65536]).log_prob(raw[i:i + 65536]).sum(-1)
                              for i in range(0, obs.shape[0], 65536)])
    n, stats = obs.shape[0], []
    for _ in range(args.epochs):
        perm = torch.randperm(n, device="cuda")
        for i in range(0, n, args.minibatch):
            idx = perm[i:i + args.minibatch]
            d = res.dist(obs[idx])
            logp = d.log_prob(raw[idx]).sum(-1)
            ratio = (logp - old_logp[idx]).exp()
            pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv[idx]).mean()
            vl = ((res.value(obs[idx]) - ret[idx]) ** 2).mean()
            ent = d.entropy().sum(-1).mean()
            loss = pg + args.value_coef * vl - args.entropy_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(res.parameters(), args.max_grad_norm)
            opt.step()
            stats.append([pg.item(), vl.item(), ent.item(), ((ratio - 1) - (ratio.log())).mean().item()])
    return dict(zip(["pg_loss", "v_loss", "entropy", "approx_kl"], np.mean(stats, 0).tolist()), transitions=n)


# --------------------------------------------------------------------------- main

def summary(ro, copies) -> dict:
    s = ro["success"]
    per_world = s.reshape(-1, copies).mean(1)
    return dict(success=float(s.mean()), per_world=per_world.round(3).tolist(),
                progress=float(ro["final_progress"].mean()),
                steps_to_success=float(ro["steps"][s].mean()) if s.any() else None)


def main(args: Args) -> None:
    if args.dry_run:
        args = dataclasses.replace(args, envs=2, copies=4, max_steps=60, iterations=2, eval_worlds=2,
                                   minibatch=256, out=args.out.with_name(args.out.name + "_dry"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    envs = [make_env(args.copies) for _ in range(args.envs)]
    global GRIPPERS
    GRIPPERS = tuple(int(i) for i in envs[0].base_env._gripper_indices)
    print(f"gripper action indices: {GRIPPERS}", flush=True)
    pi05 = Pi05(args)
    print(f"envs + pi0.5 ready in {time.time() - t0:.0f}s", flush=True)

    if args.mode == "eval":
        res = None
        if args.residual is not None:
            res = Residual(OBS_DIM, ACT_DIM, args.hidden, args.init_log_std).cuda()
            res.load_state_dict(torch.load(args.residual, map_location="cuda")["residual"])
        seeds = select_seeds(envs[0], args.eval_seed_start, args.eval_worlds, args.categories, args.sides)
        rows = []
        for i in range(0, len(seeds), args.envs):
            group = seeds[i:i + args.envs]
            ro = rollout(args, pi05, envs[:len(group)], group, res, stochastic=False)
            s = summary(ro, args.copies)
            rows += [dict(seed=sd, success=p) for sd, p in zip(group, s["per_world"])]
            print(f"eval {i + len(group)}/{len(seeds)} worlds: running success "
                  f"{np.mean([r['success'] for r in rows]):.3f} | batch {s['success']:.3f}, {ro['steps_run']} steps, "
                  f"time {ro['timing']}", flush=True)
        tag = "residual" if res is not None else "pi05_alone"
        result = dict(tag=tag, residual=str(args.residual), worlds=len(rows), copies=args.copies,
                      success=float(np.mean([r["success"] for r in rows])), per_world=rows)
        (args.out / f"eval_{tag}.json").write_text(json.dumps(result, indent=1))
        print(f"{tag}: success {result['success']:.3f} over {len(rows)} worlds x {args.copies} copies")
        return

    seed_file = args.out / "train_seeds.json"
    if seed_file.exists():
        train_seeds = json.loads(seed_file.read_text())
    else:
        train_seeds = select_seeds(envs[0], args.train_seed_start, 400 if not args.dry_run else 8,
                                   args.categories, args.sides)
        seed_file.write_text(json.dumps(train_seeds))
    res = Residual(OBS_DIM, ACT_DIM, args.hidden, args.init_log_std).cuda()
    opt = torch.optim.Adam(res.parameters(), lr=args.lr)
    start = 0
    if (args.out / "last.pt").exists():
        ck = torch.load(args.out / "last.pt", map_location="cuda")
        res.load_state_dict(ck["residual"]); opt.load_state_dict(ck["opt"]); start = ck["iteration"] + 1
        print(f"resuming at iteration {start}", flush=True)
    log = (args.out / "log.jsonl").open("a")
    for it in range(start, args.iterations):
        t_it = time.time()
        seeds = [int(s) for s in rng.choice(train_seeds, size=args.envs, replace=False)]
        ro = rollout(args, pi05, envs, seeds, res, stochastic=True)
        t_roll = time.time() - t_it
        stats = ppo_update(args, res, opt, ro)
        row = dict(iteration=it, **summary(ro, args.copies), seeds=seeds,
                   ep_return=float(ro["rew"].sum(0).mean()),
                   mean_abs_delta=float(np.abs(np.tanh(ro["raw"][ro["alive"]])).mean()),
                   std=float(res.log_std.exp().mean()), rollout_s=round(t_roll, 1),
                   total_s=round(time.time() - t_it, 1), steps_run=ro["steps_run"], timing=ro["timing"], **stats)
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"it {it}: success {row['success']:.3f} progress {row['progress']:.3f} return {row['ep_return']:.2f} "
              f"|delta| {row['mean_abs_delta']:.3f} kl {row['approx_kl']:.4f} ({row['rollout_s']}s roll, "
              f"{row['total_s']}s total, {row['steps_run']} steps, {row['timing']})", flush=True)
        if it % args.save_every == 0 or it == args.iterations - 1:
            ck = dict(residual=res.state_dict(), opt=opt.state_dict(), iteration=it, args=dataclasses.asdict(args))
            torch.save(ck, args.out / "last.pt")
            if it % (args.save_every * 10) == 0:
                torch.save(ck, args.out / f"iter_{it:04d}.pt")


if __name__ == "__main__":
    main(tyro.cli(Args))
