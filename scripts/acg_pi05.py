"""Action Coherence Guidance (ACG) for LeRobot pi0.5, applied to a loaded policy without forking LeRobot.

Follows the paper (arXiv 2510.22201), Algorithm 1. At every flow-matching denoising step, run the model twice:
  v     normal pass
  v_ic  incoherent pass: in a subset of the action expert's layers the attention map is the identity,
        so each action token attends only to itself and the attention output is its own V (Eq. 11)
and steer away from the incoherent prediction (Eq. 8):
  v_acg = (1 + scale) * v - scale * v_ic
scale is the paper's lambda: 0 reproduces the normal policy exactly, the paper's default is 3.

The authors' code (third_party/ACG) computes v + (scale - 1) * (v - v_ic) instead, so its scale s is
lambda = s - 1. This file follows the paper, so runs made before this change have lambda = old scale - 1.

pi0.5 integrates from noise at t=1 to actions at t=0 with v = noise - actions, the reverse of the
paper's t=0 to 1 with v = actions - noise. Eq. 8 is linear in v, so the guided sample is unchanged.

pi0.5 merges prefix and action attention into one operation, where GR00T has separate cross-attention
blocks. An identity map therefore also cuts images, text and state out of the perturbed layers; the
unperturbed layers still see them.

Which layers: the paper perturbs the 4th-6th self-attention layers of GR00T N1 (blocks 7, 9, 11 in
third_party/ACG), the last half, and gives no layers for pi0. By default we perturb the last half of
pi0.5's 18 action-expert layers; `layers` picks others.

Mechanics: LeRobot's PI05Pytorch.sample_actions calls self.denoise_step(...) once per step, and the
action expert's layers each receive the same additive 4D mask (batch, 1, actions, prefix + actions)
as `attention_mask`. We override denoise_step on the instance, and wrap the chosen layers' self_attn
so that during the incoherent pass their attention map is the identity.

Usage:
  from acg_pi05 import set_acg_scale
  set_acg_scale(policy, 3.0)                 # last half of the expert layers
  set_acg_scale(policy, 3.0, layers=range(18))  # every layer
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


def identity_attention(mask: torch.Tensor) -> torch.Tensor:
    """Additive mask (B, 1, S, prefix + S) that leaves only the diagonal of the actions block: an identity
    attention map, so each action token attends only to itself (Eq. 11)."""
    s = mask.shape[-2]
    eye = torch.eye(s, dtype=torch.bool, device=mask.device)
    out = torch.full_like(mask, OPENPI_ATTENTION_MASK_VALUE)
    out[..., -s:] = torch.where(eye, mask[..., -s:], out[..., -s:])
    return out


def _wrap_self_attn(attn: torch.nn.Module, idx: int, state: dict) -> None:
    plain = attn.forward

    def forward(*args, **kwargs):
        if state["incoherent"] and idx in state["layers"] and kwargs.get("attention_mask") is not None:
            kwargs["attention_mask"] = identity_attention(kwargs["attention_mask"])
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


def set_acg_scale(policy, scale: float | None, layers: Iterable[int] | None = None) -> None:
    """Turn ACG on with `scale` (perturbing `layers` of the action expert, default: the last half),
    or off with None. Safe to call repeatedly."""
    model = find_pi05_model(policy)
    if not hasattr(model, "_plain_denoise_step"):
        model._acg_state = {"incoherent": False, "layers": set()}
        for i, layer in enumerate(expert_layers(model)):
            _wrap_self_attn(layer.self_attn, i, model._acg_state)
        model._plain_denoise_step = model.denoise_step
        model.denoise_step = types.MethodType(_acg_denoise_step, model)
    model._acg_state["layers"] = set(default_layers(model) if layers is None else layers)
    model.acg_scale = scale
