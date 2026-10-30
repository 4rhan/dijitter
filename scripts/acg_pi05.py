"""Action Coherence Guidance (ACG) for LeRobot pi0.5, applied to a loaded policy without forking LeRobot.

ACG (arXiv 2510.22201): at every flow-matching denoising step, run the model twice:
  v      normal pass
  v_inc  "incoherent" pass: each action token attends to the prefix (images, text, state) and to
         itself only, so the action tokens cannot coordinate with each other
and steer away from the incoherent prediction:
  v_acg = v + (scale - 1) * (v - v_inc)
scale = 1.0 reproduces the normal policy exactly; scale > 1 pushes toward temporally coherent chunks.

LeRobot's PI05Pytorch.sample_actions calls self.denoise_step(...) once per step, and denoise_step
builds the action<->action attention block with a single make_att_2d_masks call. So we override
denoise_step on the instance, and for the incoherent pass wrap make_att_2d_masks so that block
becomes the identity. This perturbs all transformer layers (the paper perturbs a subset).

Usage:
  from acg_pi05 import set_acg_scale
  set_acg_scale(policy, 2.0)   # None turns it off again
"""

from __future__ import annotations

import contextlib
import types

import torch

import lerobot.policies.pi05.modeling_pi05 as modeling_pi05


@contextlib.contextmanager
def incoherent_actions():
    """Within this block, masks built by pi05's denoise_step let each action token see only itself
    among the action tokens (its view of the prefix is untouched: that part is built separately)."""
    original = modeling_pi05.make_att_2d_masks

    def self_only(pad_masks, att_masks):
        masks = original(pad_masks, att_masks)
        eye = torch.eye(masks.shape[-1], dtype=torch.bool, device=masks.device)
        return masks & eye

    modeling_pi05.make_att_2d_masks = self_only
    try:
        yield
    finally:
        modeling_pi05.make_att_2d_masks = original


def find_pi05_model(policy) -> torch.nn.Module:
    """The PI05Pytorch module inside a PI05Policy (also when wrapped by PEFT)."""
    for module in policy.modules():
        if type(module).__name__ == "PI05Pytorch":
            return module
    raise ValueError("no PI05Pytorch module found; is this a LeRobot pi05 policy?")


def _acg_denoise_step(self, **kwargs):
    v = self._plain_denoise_step(**kwargs)
    if self.acg_scale is None:
        return v
    with incoherent_actions():
        v_inc = self._plain_denoise_step(**kwargs)
    return v + (self.acg_scale - 1.0) * (v - v_inc)


def set_acg_scale(policy, scale: float | None) -> None:
    """Turn ACG on with `scale`, or off with None. Safe to call repeatedly."""
    model = find_pi05_model(policy)
    if not hasattr(model, "_plain_denoise_step"):
        model._plain_denoise_step = model.denoise_step
        model.denoise_step = types.MethodType(_acg_denoise_step, model)
    model.acg_scale = scale
