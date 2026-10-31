"""Action Coherence Guidance (ACG) for LeRobot pi0.5, applied to a loaded policy without forking LeRobot.

ACG (arXiv 2510.22201): at every flow-matching denoising step, run the model twice:
  v      normal pass
  v_inc  "incoherent" pass: in a subset of the action expert's layers, each action token attends to
         the prefix (images, text, state) and to itself only, so those layers cannot coordinate the
         action tokens with each other
and steer away from the incoherent prediction:
  v_acg = v + (scale - 1) * (v - v_inc)
scale = 1.0 reproduces the normal policy exactly; scale > 1 pushes toward temporally coherent chunks.

Which layers: the reference implementation (third_party/ACG, GR00T N1) replaces self-attention with
identity attention in 3 of its 6 self-attention layers, the last half (blocks 7, 9, 11). Perturbing
every layer is far too strong: v_inc stops resembling a policy at all (on our checkpoint its chunks
were ~100x jerkier than plain pi0.5), and steering away from it adds noise instead of coherence. So
by default we perturb the last half of pi0.5's 18 action-expert layers; `layers` picks others.

Mechanics: LeRobot's PI05Pytorch.sample_actions calls self.denoise_step(...) once per step, and the
action expert's layers each receive the same additive 4D mask (batch, 1, actions, prefix + actions)
as `attention_mask`. We override denoise_step on the instance, and wrap the chosen layers' self_attn
so that during the incoherent pass the actions<->actions block of their mask is the identity.

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


def identity_actions(mask: torch.Tensor) -> torch.Tensor:
    """Additive mask (B, 1, S, prefix + S) with its last S x S (actions <-> actions) block made diagonal."""
    s = mask.shape[-2]
    eye = torch.eye(s, dtype=torch.bool, device=mask.device)
    out = mask.clone()
    block = out[..., -s:]
    out[..., -s:] = torch.where(eye, block, torch.full_like(block, OPENPI_ATTENTION_MASK_VALUE))
    return out


def _wrap_self_attn(attn: torch.nn.Module, idx: int, state: dict) -> None:
    plain = attn.forward

    def forward(*args, **kwargs):
        if state["incoherent"] and idx in state["layers"] and kwargs.get("attention_mask") is not None:
            kwargs["attention_mask"] = identity_actions(kwargs["attention_mask"])
        return plain(*args, **kwargs)

    attn.forward = forward


def _acg_denoise_step(self, **kwargs):
    v = self._plain_denoise_step(**kwargs)
    if self.acg_scale is None or self.acg_scale == 1.0:
        return v
    self._acg_state["incoherent"] = True
    try:
        v_inc = self._plain_denoise_step(**kwargs)
    finally:
        self._acg_state["incoherent"] = False
    return v + (self.acg_scale - 1.0) * (v - v_inc)


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
