from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

DUAL_ROOT = Path(__file__).resolve().parent
VOICE_ROOT = DUAL_ROOT.parent
DOCUMENTS_ROOT = VOICE_ROOT.parent
POGUISE_ROOT = DOCUMENTS_ROOT / "poguise"
OVERRIDES_PATH = DUAL_ROOT / "config.overrides.json"

for path in (VOICE_ROOT, POGUISE_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _field(
    path: str,
    label: str,
    section: str,
    description: str,
    field_type: str,
    **kwargs: Any,
) -> dict:
    item = {
        "path": path,
        "label": label,
        "section": section,
        "description": description,
        "type": field_type,
    }
    item.update(kwargs)
    return item


def available_poguise_checkpoints() -> List[str]:
    checkpoints = sorted(path.name for path in POGUISE_ROOT.glob("*.ckpt"))
    preferred = []
    for name in (
        "poguise_runtime_sanitized.ckpt",
        "poguise_c2hntf6v_epoch=51-val_loss=0.507.ckpt",
    ):
        if name in checkpoints:
            preferred.append(name)
            checkpoints.remove(name)
    return preferred + checkpoints


def default_poguise_config() -> Dict[str, Any]:
    checkpoints = available_poguise_checkpoints()
    default_checkpoint = (
        checkpoints[0]
        if checkpoints
        else "poguise_runtime_sanitized.ckpt"
    )
    return {
        "checkpoint": default_checkpoint,
        "camera": 0,
        "fp16": True,
        "infer_every": 6,
        "confidence_threshold": 0.12,
        "debounce_frames": 3,
    }


def default_scheduler_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "voice_target_ms": 275.0,
        "voice_hard_limit_ms": 330.0,
        "max_infer_every": 10,
        "pause_after_voice_sec": 0.35,
    }


def _poguise_fields() -> List[dict]:
    checkpoints = available_poguise_checkpoints() or ["poguise_runtime_sanitized.ckpt"]
    return [
        _field(
            "checkpoint",
            "Checkpoint",
            "Model",
            "Runtime checkpoint used for live PO-GUISE inference. The sanitized runtime checkpoint is preferred when available.",
            "select",
            options=checkpoints,
        ),
        _field(
            "camera",
            "Camera Index",
            "Capture",
            "OpenCV camera index used for the live video feed.",
            "integer",
            min=0,
            step=1,
        ),
        _field(
            "fp16",
            "FP16 Inference",
            "Model",
            "Enable half precision on CUDA. This should stay on for Jetson unless you are debugging precision issues.",
            "boolean",
        ),
        _field(
            "infer_every",
            "Base Infer Every",
            "Scheduling",
            "Run one PO-GUISE inference every N captured frames before scheduler backoff is applied.",
            "integer",
            min=1,
            max=30,
            step=1,
        ),
        _field(
            "confidence_threshold",
            "Confidence Threshold",
            "Recognition",
            "Minimum smoothed top-1 probability required before the action debounce logic can commit a new action.",
            "number",
            min=0.01,
            max=0.95,
            step=0.01,
        ),
        _field(
            "debounce_frames",
            "Debounce Inferences",
            "Recognition",
            "How many consecutive winning inferences are required before the current action changes.",
            "integer",
            min=1,
            max=12,
            step=1,
        ),
    ]


SCHEDULER_FIELDS = [
    _field(
        "enabled",
        "Enable Voice Priority",
        "Policy",
        "When enabled, PO-GUISE backs off automatically when recent Voice latency goes above target.",
        "boolean",
    ),
    _field(
        "voice_target_ms",
        "Voice Target",
        "Policy",
        "Soft latency target for Voice. Once recent Voice inference crosses this, PO-GUISE starts to back off.",
        "number",
        min=100.0,
        max=600.0,
        step=5.0,
    ),
    _field(
        "voice_hard_limit_ms",
        "Voice Hard Limit",
        "Policy",
        "Hard latency guardrail for Voice. Crossing this can temporarily pause PO-GUISE inference.",
        "number",
        min=120.0,
        max=900.0,
        step=5.0,
    ),
    _field(
        "max_infer_every",
        "Max Infer Every",
        "Policy",
        "Upper limit the scheduler can push PO-GUISE to when protecting Voice.",
        "integer",
        min=2,
        max=60,
        step=1,
    ),
    _field(
        "pause_after_voice_sec",
        "Pause After Spike",
        "Policy",
        "How long PO-GUISE should stand down after a hard Voice latency spike.",
        "number",
        min=0.0,
        max=3.0,
        step=0.05,
    ),
]


SECTION_DEFAULTS = {
    "poguise": default_poguise_config,
    "scheduler": default_scheduler_config,
}


def section_fields(section: str) -> List[dict]:
    if section == "poguise":
        return _poguise_fields()
    if section == "scheduler":
        return deepcopy(SCHEDULER_FIELDS)
    raise KeyError(section)


def load_overrides() -> Dict[str, Dict[str, Any]]:
    if not OVERRIDES_PATH.exists():
        return {}
    with open(OVERRIDES_PATH, "r", encoding="utf-8") as handle:
        data = json.load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("dual_dashboard config.overrides.json must be a JSON object")
    return data


def save_overrides(data: Dict[str, Dict[str, Any]]) -> None:
    if data:
        with open(OVERRIDES_PATH, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
    elif OVERRIDES_PATH.exists():
        OVERRIDES_PATH.unlink()


def load_section_config(section: str) -> Dict[str, Any]:
    defaults = SECTION_DEFAULTS[section]()
    overrides = load_overrides().get(section, {})
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if key in merged:
            merged[key] = value
    if section == "poguise":
        checkpoints = available_poguise_checkpoints()
        if checkpoints and merged["checkpoint"] not in checkpoints:
            merged["checkpoint"] = checkpoints[0]
    return merged


def _coerce_editor_value(raw_value: Any, meta: dict) -> Any:
    field_type = meta["type"]
    value = raw_value
    if field_type == "boolean":
        value = bool(raw_value)
    elif field_type == "integer":
        try:
            value = int(raw_value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid integer for {meta['label']}") from exc
    elif field_type == "number":
        try:
            value = float(raw_value)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid number for {meta['label']}") from exc
    elif field_type == "select":
        if raw_value not in meta.get("options", []):
            raise ValueError(f"Invalid option for {meta['label']}")
    else:
        raise ValueError(f"Unsupported field type: {field_type}")

    if isinstance(value, (int, float)):
        if "min" in meta and value < meta["min"]:
            raise ValueError(f"{meta['label']} must be >= {meta['min']}")
        if "max" in meta and value > meta["max"]:
            raise ValueError(f"{meta['label']} must be <= {meta['max']}")
    return value


def editor_payload(section: str) -> dict:
    overrides = load_overrides()
    return {
        "config": load_section_config(section),
        "fields": section_fields(section),
        "override_file": OVERRIDES_PATH.name,
        "has_overrides": section in overrides and bool(overrides[section]),
    }


def apply_editor_update(section: str, values: Dict[str, Any]) -> dict:
    defaults = SECTION_DEFAULTS[section]()
    fields = section_fields(section)
    field_map = {field["path"]: field for field in fields}
    overrides = load_overrides()
    section_override = dict(overrides.get(section, {}))

    for path, raw_value in values.items():
        meta = field_map.get(path)
        if meta is None:
            raise ValueError(f"Unsupported setting: {path}")
        value = _coerce_editor_value(raw_value, meta)
        if value == defaults[path]:
            section_override.pop(path, None)
        else:
            section_override[path] = value

    if section_override:
        overrides[section] = section_override
    else:
        overrides.pop(section, None)
    save_overrides(overrides)
    return editor_payload(section)


def reset_editor_section(section: str) -> dict:
    overrides = load_overrides()
    overrides.pop(section, None)
    save_overrides(overrides)
    return editor_payload(section)
