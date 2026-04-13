#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from enrollment_store import rebuild_enrollment_cache
from pipeline import RealtimePipeline


BASE_DIR = Path(__file__).resolve().parent


def deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {k: copy.deepcopy(v) for k, v in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def load_config(extra_override: dict | None = None) -> dict:
    with open(BASE_DIR / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    override_path = BASE_DIR / "config.overrides.yaml"
    if override_path.exists():
        with open(override_path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if extra_override:
        cfg = deep_merge(cfg, extra_override)
    cfg.setdefault("debug", {})
    cfg["debug"]["step_perf_log"] = False
    cfg["debug"]["capture_live_session"] = False
    return cfg


def parse_override_file(path: str | None) -> dict:
    if not path:
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("override file must contain a mapping")
    return data


def apply_set(overrides: dict, assignment: str) -> None:
    if "=" not in assignment:
        raise ValueError(f"Expected key=value, got: {assignment}")
    key, raw_value = assignment.split("=", 1)
    value = yaml.safe_load(raw_value)
    cursor = overrides
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def replay_capture(session_dir: Path, cfg: dict) -> dict[str, Any]:
    capture = session_dir / "mic_input.wav"
    if not capture.exists():
        raise FileNotFoundError(f"Missing {capture}")
    audio, _ = sf.read(str(capture), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    enrolled = rebuild_enrollment_cache(Path(cfg["enrollment"]["dir"]), cfg)
    pipe = RealtimePipeline(cfg, enrolled)
    try:
        step_samples = pipe.step_samples
        chunk_samples = pipe.chunk_samples
        buf = np.zeros(chunk_samples, dtype=np.float32)
        rows = []
        for i in range(0, len(audio), step_samples):
            hop = audio[i : i + step_samples]
            if len(hop) < step_samples:
                hop = np.pad(hop, (0, step_samples - len(hop)))
            buf = np.roll(buf, -step_samples)
            buf[-step_samples:] = hop
            rows.append(pipe.step(buf))
        return summarize(rows)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def summarize(rows) -> dict[str, Any]:
    infer = [float(r.infer_ms) for r in rows]
    mich_steps = 0
    unknown_steps = 0
    both_steps = 0
    no_label_steps = 0
    mich_rms_sum = 0.0
    unknown_rms_sum = 0.0
    onset_sequences: list[list[tuple[int, float]]] = []
    current_sequence: list[tuple[int, float]] = []

    for r in rows:
        speakers = list(r.speakers)
        labels = [s.label for s in speakers]
        has_mich = "mich" in labels
        has_unknown = any(label.startswith("Unknown-") for label in labels)
        if has_mich:
            mich_steps += 1
        if has_unknown:
            unknown_steps += 1
        if has_mich and has_unknown:
            both_steps += 1
        if not labels:
            no_label_steps += 1

        mich_packet = None
        for s in speakers:
            rms = float(np.sqrt(np.mean(np.asarray(s.audio, dtype=np.float64) ** 2))) if len(s.audio) else 0.0
            if s.label == "mich":
                mich_packet = (r.step_idx, rms)
                mich_rms_sum += rms
            elif s.label.startswith("Unknown-"):
                unknown_rms_sum += rms

        if mich_packet is not None:
            current_sequence.append(mich_packet)
        elif current_sequence:
            onset_sequences.append(current_sequence)
            current_sequence = []

    if current_sequence:
        onset_sequences.append(current_sequence)

    onset_ratios = []
    onset_bad_count = 0
    for seq in onset_sequences:
        peak = max(rms for _, rms in seq[: min(3, len(seq))])
        if peak < 0.01:
            continue
        ratio = seq[0][1] / peak if peak > 0 else 0.0
        onset_ratios.append(ratio)
        if ratio < 0.35:
            onset_bad_count += 1

    return {
        "steps": len(rows),
        "mich_steps": mich_steps,
        "unknown_steps": unknown_steps,
        "both_steps": both_steps,
        "no_label_steps": no_label_steps,
        "mich_rms_sum": round(mich_rms_sum, 4),
        "unknown_rms_sum": round(unknown_rms_sum, 4),
        "avg_infer_ms": round(sum(infer) / len(infer), 1),
        "p95_infer_ms": round(sorted(infer)[int(0.95 * (len(infer) - 1))], 1),
        "onset_count": len(onset_ratios),
        "onset_mean_ratio": round(float(np.mean(onset_ratios)), 3) if onset_ratios else None,
        "onset_median_ratio": round(float(np.median(onset_ratios)), 3) if onset_ratios else None,
        "onset_bad_count": int(onset_bad_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one saved live session and score continuity/onsets.")
    parser.add_argument("session_dir", help="Path to logs/live_sessions/<session_dir>")
    parser.add_argument("--override-file", help="Extra YAML override file applied on top of config.overrides.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra override, e.g. audio.output_latency_sec=1.5",
    )
    args = parser.parse_args()

    extra_override = parse_override_file(args.override_file)
    for assignment in args.set:
        apply_set(extra_override, assignment)

    cfg = load_config(extra_override)
    start = time.time()
    metrics = replay_capture(Path(args.session_dir), cfg)
    metrics["elapsed_sec"] = round(time.time() - start, 1)
    metrics["config"] = {
        "embedding.model": cfg["embedding"]["model"],
        "audio.output_latency_sec": cfg["audio"]["output_latency_sec"],
        "audio.enrolled_onset_activity_threshold": cfg["audio"].get(
            "enrolled_onset_activity_threshold",
            cfg["audio"].get("enrolled_onset_gate_threshold"),
        ),
        "audio.enrolled_min_activity": cfg["audio"].get("enrolled_min_activity"),
        "audio.enrolled_onset_min_activity": cfg["audio"].get("enrolled_onset_min_activity"),
        "clustering.tau_active": cfg["clustering"]["tau_active"],
        "clustering.enrolled_continuity_max_gap": cfg["clustering"]["enrolled_continuity_max_gap"],
        "clustering.onset_aux_dominance_ratio": cfg["clustering"]["onset_aux_dominance_ratio"],
        "clustering.onset_aux_dist_margin": cfg["clustering"]["onset_aux_dist_margin"],
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
