"""Action Coherence Guidance (ACG) for LeRobot pi0.5, applied to a loaded policy without forking LeRobot.

Follows the paper (arXiv 2510.22201), Algorithm 1. At every flow-matching denoising step, run the model twice:
  v     normal pass
  v_ic  incoherent pass: in a subset of the action expert's layers the action tokens cannot attend to
        each other, only to themselves (the identity attention map of Eq. 11)
and steer away from the incoherent prediction (Eq. 8):
  v_acg = (1 + scale) * v - scale * v_ic
scale is the paper's lambda: 0 reproduces the normal policy exactly. The paper's text gives 3 as the
default, but the authors' code (third_party/ACG) computes v + (s - 1) * (v - v_ic) with s = 3, which is
lambda = 2 and is the code behind their results. Runs named *_acg<s> used lambda = s - 1.

pi0.5 integrates from noise at t=1 to actions at t=0 with v = noise - actions, the reverse of the
paper's t=0 to 1 with v = actions - noise. Eq. 8 is linear in v, so the guided sample is unchanged.

Denoising steps: guidance pushes noise out faster than Euler at 10 steps can follow, so the final chunk
keeps a residual of the initial noise (correlation with it -0.81 at lambda 1, last half, on the 15k
checkpoint), and that is the extra jerk. On the same frames, intra-chunk jerk against plain pi0.5 at the
same step count, lambda 1, layers 15-17: 2.88x at 10 steps, 2.10x at 16, 1.14x at 32, 0.62x at 64. Use
serve_pi05's --num_steps or run_eval's NUM_STEPS. Each step costs two model passes.

Images, text and state: GR00T perturbs pure self-attention blocks (encoder_hidden_states=None) and takes
vision and language through separate cross-attention blocks that ACG never touches. pi0.5 merges prefix
and action attention into one operation, so the equivalent is to make only the actions<->actions block
the identity and keep the prefix columns (keep_prefix=True, the default). keep_prefix=False applies
Eq. 11 literally and also cuts the perturbed layers off from what the robot sees: v_ic stops resembling
a policy (485x jerkier on our 15k checkpoint, against about 3.4x in the paper) and closed-loop success
falls from 54% to 6% at lambda 3. Runs named *_lambda<x> without _keepprefix used that literal mask.

Which layers: the paper perturbs the 4th-6th self-attention layers of GR00T N1 (blocks 7, 9, 11 in
third_party/ACG), the last half, and gives no layers for pi0. By default we perturb the last half of
pi0.5's 18 action-expert layers; `layers` picks others.

Mechanics: LeRobot's PI05Pytorch.sample_actions calls self.denoise_step(...) once per step, and the
action expert's layers each receive the same additive 4D mask (batch, 1, actions, prefix + actions)
as `attention_mask`. We override denoise_step on the instance, and wrap the chosen layers' self_attn
so that during the incoherent pass their attention map is the identity.

Usage:
  from acg_pi05 import set_acg_scale
  set_acg_scale(policy, 2.0)                 # last half of the expert layers
  set_acg_scale(policy, 2.0, layers=range(18))  # every layer
  set_acg_scale(policy, None)                # off
"""

from __future__ import annotations

import types
from collections.abc import Iterable

import torch

from lerobot.policies.common.vla_utils import OPENPI_ATTENTION_MASK_VALUE


def find_pi05_model(policy) -> torch.nn.Module:
    """The PI05Pytorch module inside a PI05Policy (also when wrapped by PEFT)."""
    for module in policy.modules():
        if type(module).__name__ == "PI05Pytorch":
            return module
    raise ValueError("no PI05Pytorch module found; is this a LeRobot pi05 policy?")


def expert_layers(model) -> torch.nn.ModuleList:
    return model.paligemma_with_expert.gemma_expert.model.layers


def default_layers(model) -> list[int]:
    n = len(expert_layers(model))
    return list(range(n // 2, n))


def parse_layers(spec: str, model) -> list[int]:
    """"half" (last half of the action expert, the default), "all", or an inclusive range like "15-17"."""
    n = len(expert_layers(model))
    if spec == "all":
        return list(range(n))
    if spec == "half":
        return default_layers(model)
    lo, hi = (int(x) for x in spec.split("-"))
    if not 0 <= lo <= hi < n:
        raise ValueError(f"layer range {spec!r} outside 0-{n - 1}")
    return list(range(lo, hi + 1))


def identity_attention(mask: torch.Tensor, keep_prefix: bool = True) -> torch.Tensor:
    """Additive mask (B, 1, S, prefix + S) with its last S x S (actions <-> actions) block made diagonal.
    keep_prefix=False also masks the prefix, so each action token attends only to itself."""
    s = mask.shape[-2]
    eye = torch.eye(s, dtype=torch.bool, device=mask.device)
    out = mask.clone() if keep_prefix else torch.full_like(mask, OPENPI_ATTENTION_MASK_VALUE)
    blocked = torch.full_like(mask[..., -s:], OPENPI_ATTENTION_MASK_VALUE)
    out[..., -s:] = torch.where(eye, mask[..., -s:], blocked)
    return out


def _wrap_self_attn(attn: torch.nn.Module, idx: int, state: dict) -> None:
    plain = attn.forward

    def forward(*args, **kwargs):
        if state["incoherent"] and idx in state["layers"] and kwargs.get("attention_mask") is not None:
            kwargs["attention_mask"] = identity_attention(kwargs["attention_mask"], state["keep_prefix"])
        return plain(*args, **kwargs)

    attn.forward = forward


def _acg_denoise_step(self, **kwargs):
    v = self._plain_denoise_step(**kwargs)
    if self.acg_scale is None or self.acg_scale == 0.0:
        return v
    self._acg_state["incoherent"] = True
    try:
        v_ic = self._plain_denoise_step(**kwargs)
    finally:
        self._acg_state["incoherent"] = False
    return (1.0 + self.acg_scale) * v - self.acg_scale * v_ic


def set_acg_scale(
    policy, scale: float | None, layers: Iterable[int] | None = None, keep_prefix: bool = True
) -> None:
    """Turn ACG on with `scale` (perturbing `layers` of the action expert, default: the last half),
    or off with None. keep_prefix is explained in the module docstring. Safe to call repeatedly."""
    model = find_pi05_model(policy)
    if not hasattr(model, "_plain_denoise_step"):
        model._acg_state = {"incoherent": False, "layers": set(), "keep_prefix": True}
        for i, layer in enumerate(expert_layers(model)):
            _wrap_self_attn(layer.self_attn, i, model._acg_state)
        model._plain_denoise_step = model.denoise_step
        model.denoise_step = types.MethodType(_acg_denoise_step, model)
    model._acg_state["layers"] = set(default_layers(model) if layers is None else layers)
    model._acg_state["keep_prefix"] = keep_prefix
    model.acg_scale = scale
