"""Check that a RELATIVE=1 checkpoint really trained on relative arm actions with absolute grippers.

Reads the processor configs the checkpoint saved. Exits non-zero if the conversion is off or the
grippers would be converted too, which would make the run worthless without any error during training.

  python scripts/cloud/check_relative.py outputs/train/pi05_easy_h100/checkpoints/last/pretrained_model
"""

import json
import sys
from pathlib import Path


def step_config(path: Path, name: str) -> dict:
    steps = json.loads(path.read_text())["steps"]
    return next(s.get("config", {}) for s in steps if s.get("registry_name") == name)


ckpt = Path(sys.argv[1])
relative = step_config(ckpt / "policy_preprocessor.json", "relative_actions_processor")
absolute = step_config(ckpt / "policy_postprocessor.json", "absolute_actions_processor")
names = relative.get("action_names") or []
excluded = relative.get("exclude_joints") or []
kept_absolute = [n for n in names if any(token in n.lower() for token in excluded)]
print(f"relative enabled: {relative.get('enabled')}, absolute enabled: {absolute.get('enabled')}")
print(f"kept absolute: {kept_absolute}")
if not (relative.get("enabled") and absolute.get("enabled")):
    sys.exit("FAILED: the relative/absolute conversion is off")
if kept_absolute != ["left_gripper", "right_gripper"]:
    sys.exit("FAILED: expected exactly the two grippers to stay absolute")
print("relative actions OK")
