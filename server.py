#!/usr/bin/env python3
"""
Voice ID — Real-Time Diarization & Separation Server

FastAPI backend that:
  - Loads the PixIT + WeSpeaker + EnrolledClustering pipeline
  - Captures mic audio via sounddevice
  - Runs a rolling-buffer diarization loop in a background thread
  - Broadcasts per-speaker audio + metadata over WebSocket
  - Handles speaker enrollment (record → extract WeSpeaker embedding → save)

Playback continuity (crossfade, continuity rescue, tau_active tuning) is implemented
inside ``RealtimePipeline.step()`` — this server only feeds fixed-size chunks
and forwards ``StepResult`` frames; see ``pipeline.py`` module docstring.

Transport notes
───────────────
The websocket sends:

1. one JSON diarization message per emitted step
2. zero or more binary audio packets immediately after it

Each binary packet includes ``step_idx`` in the header. That field is not just
for debugging; the dashboard audio scheduler uses it to keep playback aligned
despite websocket/browser jitter. Earlier versions ignored it and scheduled
audio only by arrival time, which sounded like periodic disconnect/reconnect
even when backend packets were continuous.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import logging
import re
import struct
import threading
import time
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from embedding_runtime import EMBEDDING_MODEL_OPTIONS
from enrollment_store import (
    embedding_cache_path,
    rebuild_enrollment_cache,
    rebuild_speaker_embedding_cache,
    reference_paths,
)
from pipeline import RealtimePipeline

# ─── Globals ──────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"
OVERRIDE_CONFIG_PATH = BASE_DIR / "config.overrides.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("server")


def load_base_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_override_config() -> dict:
    if not OVERRIDE_CONFIG_PATH.exists():
        return {}
    with open(OVERRIDE_CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.overrides.yaml must contain a mapping")
    return data


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = {k: deepcopy(v) for k, v in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = _deep_merge(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        return merged
    return deepcopy(override)


def load_config() -> dict:
    base = load_base_config()
    overrides = load_override_config()
    return _deep_merge(base, overrides)


def _sanitize_slug(value: str, default: str = "clip") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or default


def _timestamp_suffix_removed(stem: str) -> str:
    return re.sub(r"_(\d{8}_\d{6})$", "", stem)


def _clip_label_from_name(path: Path, prefix: str = "") -> str:
    stem = path.stem
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
        if stem.startswith("_"):
            stem = stem[1:]
    stem = _timestamp_suffix_removed(stem)
    return stem or "base"


def _clip_duration_sec(path: Path) -> float:
    try:
        info = sf.info(str(path))
        if not info.samplerate:
            return 0.0
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def _serialize_clip(path: Path, prefix: str = "") -> dict:
    stat = path.stat()
    return {
        "name": path.name,
        "label": _clip_label_from_name(path, prefix),
        "size": stat.st_size,
        "duration_sec": round(_clip_duration_sec(path), 3),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def _speaker_reference_path(
    speaker_dir: Path,
    reference_label: Optional[str],
) -> Path:
    label = (reference_label or "").strip()
    if label:
        return speaker_dir / f"reference_{_sanitize_slug(label)}.wav"
    existing = reference_paths(speaker_dir)
    if not existing:
        return speaker_dir / "reference.wav"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return speaker_dir / f"reference_{timestamp}.wav"


def _config_get(cfg: dict, path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _config_set(cfg: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = cfg
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


def _config_delete(cfg: dict, path: str) -> None:
    parts = path.split(".")
    stack: List[tuple[dict, str]] = []
    cur: Any = cfg
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        stack.append((cur, part))
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
    for parent, key in reversed(stack):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)
        else:
            break


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


CONFIG_EDITOR_FIELDS: List[dict] = [
    _field(
        "audio.device_index",
        "Mic Device Index",
        "Audio",
        "sounddevice input index used for live capture. Change this when the app is listening to the wrong microphone.",
        "integer",
        min=0,
        step=1,
    ),
    _field(
        "audio.output_latency_sec",
        "Output Latency",
        "Audio",
        "How far the monitor output lags behind live capture. Higher values usually reduce edge-of-window artifacts, but add more delay.",
        "number",
        min=0.5,
        max=5.0,
        step=0.1,
    ),
    _field(
        "audio.enrolled_live_mix",
        "Enrolled Live Mix",
        "Audio",
        "When enabled, enrolled speakers play delayed gated mixed mic audio instead of the separated PixIT source. More stable, but it carries room/background with the target voice.",
        "boolean",
    ),
    _field(
        "audio.enrolled_mix_gate_threshold",
        "Mix Gate Threshold",
        "Audio",
        "Activity threshold used to open the enrolled live-mix gate. Lower values keep the lane open more easily; higher values cut more bleed but can clip soft speech.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "audio.enrolled_mix_gate_collar_sec",
        "Mix Gate Collar",
        "Audio",
        "Extra time added before and after detected speech when gating enrolled live mix. Useful for preserving consonants at the edges of phrases.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "audio.enrolled_onset_gate_threshold",
        "Onset Gate Threshold",
        "Audio",
        "More permissive gate threshold used only when an enrolled speaker is recognized one step later than the delayed packet being emitted. Lower values help recover clipped first syllables.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "audio.enrolled_onset_gate_collar_sec",
        "Onset Gate Collar",
        "Audio",
        "Extra collar used only for late-recognized enrolled onsets. Increase this to recover more of the first word at the cost of more room bleed right before speech starts.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "audio.enrolled_onset_min_activity",
        "Onset Min Activity",
        "Audio",
        "Reduced activity floor used only for late-recognized enrolled onsets. Lower values help the first part of a word come through when the speaker map stabilizes one step late.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "audio.enrolled_onset_fade_sec",
        "Onset Fade",
        "Audio",
        "Short fade-in applied only to newly emitted enrolled separated packets. This softens abrupt starts when using model output instead of live mix.",
        "number",
        min=0.0,
        max=0.25,
        step=0.005,
    ),
    _field(
        "audio.separated_leakage_removal",
        "Separated Leakage Removal",
        "Audio",
        "Approximate pyannote off-turn cleanup for separated audio. This only zeros the lane when the speaker is inactive; it does not remove another speaker during active speech.",
        "boolean",
    ),
    _field(
        "audio.separated_gate_threshold",
        "Separated Gate Threshold",
        "Audio",
        "Activity threshold for separated-output cleanup. Keep this low; higher values can punch holes into syllables on separated playback.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "audio.separated_leakage_collar_sec",
        "Separated Leakage Collar",
        "Audio",
        "Context kept around active separated speech before zeroing off-turn audio. Larger collars preserve more edges but also keep more residual bleed.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "pixit.duration",
        "Chunk Duration",
        "PixIT",
        "Length of each mono chunk sent to the separation model. The released AMI checkpoint is trained for 5 second chunks.",
        "number",
        min=1.0,
        max=10.0,
        step=0.5,
    ),
    _field(
        "pixit.step",
        "Chunk Step",
        "PixIT",
        "Hop between inference chunks. Smaller steps mean more overlap and smoother tracking, but more compute.",
        "number",
        min=0.1,
        max=5.0,
        step=0.1,
    ),
    _field(
        "pixit.max_speakers",
        "Local Sources",
        "PixIT",
        "Maximum number of speaker sources the PixIT model outputs for one 5 second chunk. Extra speakers or noise must be absorbed into those local slots.",
        "integer",
        min=1,
        max=8,
        step=1,
    ),
    _field(
        "pixit.aggregation_latency_sec",
        "Activity Aggregation Latency",
        "PixIT",
        "Hamming fusion window for speaker activity tracks. This smooths diarization/UI activity, not the separated waveform itself.",
        "number",
        min=0.5,
        max=5.0,
        step=0.1,
    ),
    _field(
        "embedding.model",
        "Embedding Model",
        "Embedding",
        "Speaker embedding backend used for enrollment rebuilds and live ID. WeSpeaker is the correct overlap-aware live path. TitaNet remains experimental here because its native API is full-utterance embedding, not diart-style weighted pooling.",
        "select",
        options=list(EMBEDDING_MODEL_OPTIONS),
        option_labels={
            "pyannote/wespeaker-voxceleb-resnet34-LM": "WeSpeaker (recommended)",
            "nvidia/speakerverification_en_titanet_large": "TitaNet (experimental)",
        },
    ),
    _field(
        "embedding.source_seg_threshold",
        "Embedding Seg Threshold",
        "Embedding",
        "Minimum local segmentation weight used when building the recent waveform/mask for overlap-aware identity embeddings.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "embedding.source_min_voiced_sec",
        "Embedding Min Voiced",
        "Embedding",
        "Minimum voiced duration before a local source is allowed to produce an identity embedding. Raising this ignores very short weak fragments.",
        "number",
        min=0.0,
        max=2.0,
        step=0.01,
    ),
    _field(
        "embedding.source_recent_sec",
        "Embedding Recent Window",
        "Embedding",
        "How much recent speech to keep for live identity extraction. Smaller values emphasize the latest speech; larger values smooth over more context.",
        "number",
        min=0.5,
        max=5.0,
        step=0.1,
    ),
    _field(
        "embedding.enrollment_profile_top_k",
        "Enrollment Top-K",
        "Embedding",
        "How many strong enrollment windows to average into the frozen speaker profile. Higher values smooth enrollment, but can blur distinct positions/rooms.",
        "integer",
        min=1,
        max=10,
        step=1,
    ),
    _field(
        "embedding.enrollment_scan_step_sec",
        "Enrollment Scan Step",
        "Embedding",
        "Hop used when scanning the enrollment reference for strong windows. Smaller steps search more densely, but cost more startup time.",
        "number",
        min=0.1,
        max=5.0,
        step=0.1,
    ),
    _field(
        "embedding.enrollment_min_voiced_sec",
        "Enrollment Min Voiced",
        "Embedding",
        "Minimum voiced speech required before an enrollment chunk can contribute to the stored profile.",
        "number",
        min=0.0,
        max=5.0,
        step=0.05,
    ),
    _field(
        "embedding.gamma",
        "Overlap Gamma",
        "Embedding",
        "Advanced overlap-aware embedding weighting knob. Change only if you are comparing replays; higher values make weighting more selective.",
        "number",
        min=0.1,
        max=10.0,
        step=0.1,
    ),
    _field(
        "embedding.beta",
        "Overlap Beta",
        "Embedding",
        "Advanced overlap-aware embedding sharpening knob. Higher values make the mask weighting steeper.",
        "number",
        min=0.1,
        max=20.0,
        step=0.1,
    ),
    _field(
        "clustering.tau_active",
        "Active Threshold",
        "Clustering",
        "Minimum local activity for a source to count as active. Lower values reduce dropouts; higher values ignore more weak/noisy locals.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "clustering.rho_update",
        "Center Update Rate",
        "Clustering",
        "How quickly non-enrolled speaker centers move toward new embeddings. Larger values adapt faster; smaller values are stabler.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "clustering.delta_new",
        "New Speaker Threshold",
        "Clustering",
        "Distance threshold for deciding when a local source is too far from existing centers and should become or reuse an unknown speaker.",
        "number",
        min=0.0,
        max=2.0,
        step=0.01,
    ),
    _field(
        "clustering.delta_enrolled",
        "Enrolled Match Threshold",
        "Clustering",
        "Strict distance threshold for assigning a local source to an enrolled anchor. Lower values are stricter and reduce false matches.",
        "number",
        min=0.0,
        max=2.0,
        step=0.01,
    ),
    _field(
        "clustering.max_speakers",
        "Global Speaker Limit",
        "Clustering",
        "Maximum number of global speakers the online clustering layer can keep alive.",
        "integer",
        min=1,
        max=16,
        step=1,
    ),
    _field(
        "clustering.leakage_delta",
        "Leakage Suppression",
        "Clustering",
        "Distance used to suppress duplicate PixIT locals that look like leakage of an already matched enrolled speaker instead of a new unknown lane.",
        "number",
        min=0.0,
        max=2.0,
        step=0.01,
    ),
    _field(
        "clustering.enrolled_continuity_margin",
        "Continuity Margin",
        "Clustering",
        "Advanced. Extra relaxed margin for enrolled continuity rescue. With max gap set to 0 this is effectively inactive.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "clustering.enrolled_continuity_max_gap",
        "Continuity Max Gap",
        "Clustering",
        "How many live steps an enrolled speaker can be missing before rescue is no longer allowed. Zero disables continuity rescue.",
        "integer",
        min=0,
        max=10,
        step=1,
    ),
    _field(
        "clustering.onset_aux_max_voiced_sec",
        "Onset Aux Max Voiced",
        "Clustering",
        "Advanced onset rescue guard. Only very short dominant locals can trigger the onset auxiliary remap at the start of an utterance.",
        "number",
        min=0.0,
        max=5.0,
        step=0.1,
    ),
    _field(
        "clustering.onset_aux_dominance_ratio",
        "Onset Aux Dominance",
        "Clustering",
        "How much louder the dominant local tail must be before onset rescue prefers it over the directly matched enrolled local.",
        "number",
        min=1.0,
        max=5.0,
        step=0.05,
    ),
    _field(
        "clustering.onset_aux_dist_margin",
        "Onset Aux Distance Margin",
        "Clustering",
        "How much farther the dominant local may be from the enrolled anchor and still win the onset rescue remap.",
        "number",
        min=0.0,
        max=1.0,
        step=0.01,
    ),
    _field(
        "denoiser.enabled",
        "Mic DeepFilter",
        "Denoiser",
        "Run DeepFilterNet on each captured mic block before PixIT and embeddings. This can reduce noise, but it also changes separator input and identity features.",
        "boolean",
    ),
    _field(
        "denoiser.enhance_enrolled_output",
        "Output DeepFilter",
        "Denoiser",
        "Experimental. Run DeepFilterNet only on emitted enrolled lanes after separation and gating. It does not affect identity, but on PixIT-separated speech it can chew up syllable interiors or sound pumpy/choppy.",
        "boolean",
    ),
    _field(
        "denoiser.model",
        "Denoiser Model",
        "Denoiser",
        "Which DeepFilterNet checkpoint to load when either denoiser path is enabled.",
        "select",
        options=["DeepFilterNet", "DeepFilterNet2", "DeepFilterNet3"],
    ),
]
CONFIG_EDITOR_FIELD_MAP = {field["path"]: field for field in CONFIG_EDITOR_FIELDS}


def _coerce_editor_value(raw_value: Any, meta: dict) -> Any:
    field_type = meta["type"]
    if field_type == "boolean":
        if isinstance(raw_value, bool):
            value = raw_value
        elif isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                value = True
            elif lowered in {"false", "0", "no", "off"}:
                value = False
            else:
                raise ValueError(f"Invalid boolean for {meta['path']}")
        else:
            raise ValueError(f"Invalid boolean for {meta['path']}")
    elif field_type == "integer":
        if isinstance(raw_value, bool):
            raise ValueError(f"Invalid integer for {meta['path']}")
        value = int(raw_value)
    elif field_type == "number":
        if isinstance(raw_value, bool):
            raise ValueError(f"Invalid number for {meta['path']}")
        value = float(raw_value)
    elif field_type in {"text", "select"}:
        value = str(raw_value)
    else:
        raise ValueError(f"Unsupported field type {field_type}")

    if "options" in meta and value not in meta["options"]:
        raise ValueError(f"Invalid option for {meta['path']}")
    if isinstance(value, (int, float)):
        if "min" in meta and value < meta["min"]:
            raise ValueError(f"{meta['label']} must be >= {meta['min']}")
        if "max" in meta and value > meta["max"]:
            raise ValueError(f"{meta['label']} must be <= {meta['max']}")
    return value


def _config_editor_payload() -> dict:
    return {
        "config": load_config(),
        "fields": CONFIG_EDITOR_FIELDS,
        "override_file": OVERRIDE_CONFIG_PATH.name,
        "has_overrides": OVERRIDE_CONFIG_PATH.exists(),
    }


app = FastAPI(title="Voice ID — Live Diarization")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loop: Optional[asyncio.AbstractEventLoop] = None


# ─── Monitor state ────────────────────────────────────────────────

class MonitorState:
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.step_count = 0
        self.last_infer_ms = 0.0
        self.last_capture_dir: Optional[str] = None

    def reset(self):
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.step_count = 0
        self.last_infer_ms = 0.0
        self.last_capture_dir = None


def _slug_label(label: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label))
    safe = safe.strip("_")
    return safe or "speaker"


class LiveSessionCapture:
    """Capture raw mic + emitted per-speaker audio + per-step metadata."""

    def __init__(self, root_dir: Path, sample_rate: int, step_samples: int):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"live_{ts}"
        self.root_dir = root_dir / self.session_id
        self.sample_rate = int(sample_rate)
        self.step_samples = int(step_samples)
        self._silence = np.zeros(self.step_samples, dtype=np.int16)
        self.mic_blocks: List[np.ndarray] = []
        self.speaker_blocks: Dict[int, List[np.ndarray]] = {}
        self.speaker_labels: Dict[int, List[str]] = {}
        self.step_meta: List[dict] = []

    def record_mic_block(self, block: np.ndarray) -> None:
        audio_i16 = np.clip(
            np.asarray(block, dtype=np.float32) * 32767.0,
            -32768,
            32767,
        ).astype(np.int16)
        self.mic_blocks.append(audio_i16.copy())

    def record_step(self, result) -> None:
        step_idx = int(result.step_idx)
        present_audio: Dict[int, np.ndarray] = {}
        present_labels: Dict[int, str] = {}
        speakers_meta = []

        for sp in result.speakers:
            speaker_id = int(sp.global_idx)
            audio_i16 = np.clip(sp.audio * 32767, -32768, 32767).astype(np.int16)
            present_audio[speaker_id] = audio_i16
            present_labels[speaker_id] = str(sp.label)
            audio_f64 = np.asarray(sp.audio, dtype=np.float64)
            audio_peak = (
                float(np.max(np.abs(audio_f64)))
                if audio_f64.size
                else 0.0
            )
            audio_rms = (
                float(np.sqrt(np.mean(audio_f64 ** 2)))
                if audio_f64.size
                else 0.0
            )
            speakers_meta.append({
                "id": speaker_id,
                "label": str(sp.label),
                "enrolled": bool(sp.is_enrolled),
                "activity": round(float(sp.activity), 4),
                "identity_similarity": (
                    round(float(sp.identity_similarity), 4)
                    if sp.identity_similarity is not None
                    else None
                ),
                "audio_active": bool(audio_peak > (32.0 / 32767.0)),
                "audio_rms": round(audio_rms, 6),
                "audio_peak": round(audio_peak, 6),
            })

        for speaker_id, label in present_labels.items():
            if speaker_id not in self.speaker_blocks:
                self.speaker_blocks[speaker_id] = [
                    self._silence.copy() for _ in range(step_idx)
                ]
                self.speaker_labels[speaker_id] = []
            history = self.speaker_labels[speaker_id]
            if not history or history[-1] != label:
                history.append(label)

        for speaker_id, blocks in self.speaker_blocks.items():
            while len(blocks) < step_idx:
                blocks.append(self._silence.copy())
            blocks.append(present_audio.get(speaker_id, self._silence.copy()))

        self.step_meta.append({
            "step": step_idx,
            "infer_ms": round(float(result.infer_ms), 3),
            "speakers": speakers_meta,
        })

    def finalize(self) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if self.mic_blocks:
            mic = np.concatenate(self.mic_blocks)
        else:
            mic = np.zeros(0, dtype=np.int16)
        sf.write(
            str(self.root_dir / "mic_input.wav"),
            mic,
            self.sample_rate,
            subtype="PCM_16",
        )

        manifest = {
            "session_id": self.session_id,
            "sample_rate": self.sample_rate,
            "step_samples": self.step_samples,
            "num_steps": len(self.step_meta),
            "num_speakers": len(self.speaker_blocks),
            "speakers": {},
        }

        for speaker_id, blocks in sorted(self.speaker_blocks.items()):
            audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int16)
            labels = self.speaker_labels.get(speaker_id, [])
            label = labels[-1] if labels else f"speaker_{speaker_id}"
            fname = f"speaker_{speaker_id}_{_slug_label(label)}.wav"
            sf.write(
                str(self.root_dir / fname),
                audio,
                self.sample_rate,
                subtype="PCM_16",
            )
            manifest["speakers"][str(speaker_id)] = {
                "file": fname,
                "labels": labels,
            }

        with open(self.root_dir / "steps.ndjson", "w") as f:
            for row in self.step_meta:
                f.write(json.dumps(row) + "\n")

        with open(self.root_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return self.root_dir


monitor = MonitorState()
ws_clients: Set[WebSocket] = set()
AUDIO_PACKET_MAGIC = b"VAUD"


# ─── Enrollment helpers ───────────────────────────────────────────

def get_enrollment_dir(cfg: dict) -> Path:
    return BASE_DIR / cfg["enrollment"]["dir"]


def load_enrolled_embeddings(cfg: dict) -> Dict[str, np.ndarray]:
    """Rebuild and load enrolled speaker embeddings from reference audio."""
    return rebuild_enrollment_cache(get_enrollment_dir(cfg), cfg)


def list_speakers(cfg: dict) -> List[dict]:
    """Return info about enrolled speakers."""
    enroll_dir = get_enrollment_dir(cfg)
    speakers = []
    if enroll_dir.exists():
        for speaker_dir in sorted(enroll_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            refs = reference_paths(speaker_dir)
            if not refs:
                continue
            references = [_serialize_clip(path, prefix="reference") for path in refs]
            speakers.append({
                "name": speaker_dir.name,
                "has_embedding": embedding_cache_path(speaker_dir).exists(),
                "has_audio": True,
                "reference_count": len(refs),
                "total_reference_sec": round(
                    sum(item["duration_sec"] for item in references),
                    3,
                ),
                "references": references,
            })
    return speakers


# ─── WebSocket broadcast ─────────────────────────────────────────

async def _ws_broadcast(message: str):
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


def ws_broadcast(data: dict):
    """Thread-safe WebSocket broadcast."""
    msg = json.dumps(data)
    if loop is not None:
        asyncio.run_coroutine_threadsafe(_ws_broadcast(msg), loop)


async def _ws_broadcast_diarization(message: str, audio_packets: list[bytes]):
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send_text(message)
            for packet in audio_packets:
                await ws.send_bytes(packet)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)


def ws_broadcast_diarization(data: dict, audio_packets: list[bytes]):
    """Broadcast diarization metadata followed by binary audio packets."""
    if loop is not None:
        asyncio.run_coroutine_threadsafe(
            _ws_broadcast_diarization(json.dumps(data), audio_packets),
            loop,
        )


def ws_event(event_type: str, **kwargs):
    """Send a system event to all WebSocket clients."""
    ws_broadcast({"type": event_type, **kwargs})


def _build_live_chunk(audio_buffer: np.ndarray, chunk_samples: int) -> np.ndarray:
    """Build a chunk-sized window ending at the newest captured samples."""
    if len(audio_buffer) >= chunk_samples:
        return audio_buffer[-chunk_samples:].copy()
    chunk = np.zeros(chunk_samples, dtype=np.float32)
    if len(audio_buffer) > 0:
        chunk[-len(audio_buffer):] = audio_buffer
    return chunk


def _trim_live_buffer(
    audio_buffer: np.ndarray,
    chunk_samples: int,
    step_samples: int,
) -> np.ndarray:
    """Keep only the overlap context needed for the next step."""
    keep = max(0, chunk_samples - step_samples)
    if len(audio_buffer) <= keep:
        return audio_buffer
    return audio_buffer[-keep:].copy()


def _pack_audio_packet(
    step_idx: int,
    speaker_id: int,
    sample_rate: int,
    audio_i16: np.ndarray,
) -> bytes:
    """Pack a speaker audio chunk into a binary websocket frame.

    Header layout:
    ``magic(4) | step_idx(4) | speaker_id(4) | sample_rate(4) | num_samples(4)``

    ``step_idx`` is part of the scheduling contract with ``dashboard.html``.
    Keep it stable if the client ever changes.
    """
    header = struct.pack(
        "<4sIIII",
        AUDIO_PACKET_MAGIC,
        int(step_idx),
        int(speaker_id),
        int(sample_rate),
        int(len(audio_i16)),
    )
    return header + audio_i16.tobytes()


# ─── Diarization monitor ─────────────────────────────────────────

def _run_diarization_monitor():
    """Background thread: mic → rolling buffer → pipeline → WebSocket."""
    cfg = load_config()
    dbg_cfg = cfg.get("debug", {})
    sample_rate = cfg["audio"]["sample_rate"]
    device_index = cfg["audio"]["device_index"]
    duration = cfg["pixit"]["duration"]
    step = cfg["pixit"]["step"]

    chunk_samples = int(duration * sample_rate)    # 80000
    step_samples = int(step * sample_rate)         # 8000
    capture: Optional[LiveSessionCapture] = None
    if dbg_cfg.get("capture_live_session", True):
        capture_root = BASE_DIR / dbg_cfg.get(
            "live_capture_dir", "logs/live_sessions"
        )
        capture = LiveSessionCapture(capture_root, sample_rate, step_samples)
        log.info("Live session capture enabled: %s", capture.root_dir)

    # Load enrolled embeddings
    enrolled = load_enrolled_embeddings(cfg)
    if enrolled:
        log.info("Loaded %d enrolled speaker(s): %s", len(enrolled), ", ".join(enrolled.keys()))
    else:
        log.info("No enrolled speakers found.")

    ws_event("status", message="Loading models...")

    # Create pipeline
    try:
        pipeline = RealtimePipeline(cfg, enrolled)
    except Exception as e:
        log.error("Failed to create pipeline: %s", e)
        ws_event("error", message=f"Pipeline init failed: {e}")
        monitor.running = False
        return

    ws_event("status", message="Models loaded. Starting mic capture...")

    # ── Rolling buffer ──
    buf_lock = threading.Lock()
    audio_buffer = np.zeros(0, dtype=np.float32)

    # Queue for mic blocks to be denoised in the main loop (not in callback)
    import queue
    mic_queue: queue.Queue = queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            log.warning("Mic status: %s", status)
        mic_queue.put(indata[:, 0].copy())

    # Open mic stream
    try:
        mic_stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=step_samples,
            device=device_index,
            callback=mic_callback,
        )
        mic_stream.start()
    except Exception as e:
        log.error("Mic open failed: %s", e)
        ws_event("error", message=f"Mic failed: {e}")
        monitor.running = False
        return

    ws_event("status", message="Live — listening...")
    log.info("Diarization monitor started (duration=%.1fs, step=%.1fs)", duration, step)

    def broadcast_step_result(result):
        monitor.step_count = result.step_idx
        monitor.last_infer_ms = result.infer_ms
        if capture is not None:
            capture.record_step(result)

        speakers_data = []
        audio_packets = []
        for sp in result.speakers:
            audio_i16 = np.clip(sp.audio * 32767, -32768, 32767).astype(np.int16)
            speaker_id = int(sp.global_idx)
            audio_peak = (
                float(np.max(np.abs(np.asarray(sp.audio, dtype=np.float32))))
                if np.asarray(sp.audio).size
                else 0.0
            )
            audio_rms = (
                float(np.sqrt(np.mean(np.asarray(sp.audio, dtype=np.float64) ** 2)))
                if np.asarray(sp.audio).size
                else 0.0
            )
            audio_active = bool(audio_peak > (32.0 / 32767.0))
            diar_active = bool(sp.activity > cfg["clustering"]["tau_active"])
            _sd = {
                "id": speaker_id,
                "label": str(sp.label),
                "active": bool(audio_active or diar_active),
                "audio_active": audio_active,
                "diar_active": diar_active,
                "audio_rms": round(audio_rms, 6),
                "audio_peak": round(audio_peak, 6),
                "activity": round(float(sp.activity), 3),
                "enrolled": bool(sp.is_enrolled),
            }
            if sp.identity_similarity is not None:
                _sd["identity_similarity"] = round(float(sp.identity_similarity), 4)
            speakers_data.append(_sd)
            audio_packets.append(
                _pack_audio_packet(result.step_idx, speaker_id, sample_rate, audio_i16)
            )

        ws_broadcast_diarization({
            "type": "diarization",
            "speakers": speakers_data,
            "step": result.step_idx,
            "infer_ms": round(result.infer_ms, 1),
            "sample_rate": sample_rate,
            "samples": pipeline.step_samples,
        }, audio_packets)

    try:
        while not monitor.stop_event.is_set():
            # Process one inference step per captured mic block so live playback
            # starts immediately instead of waiting for a full 5 s buffer.
            drained = False
            while not mic_queue.empty():
                try:
                    block = mic_queue.get_nowait()
                    block = pipeline.denoise_block(block)  # ~8ms for 500ms
                    if capture is not None:
                        capture.record_mic_block(block)
                    with buf_lock:
                        audio_buffer = np.concatenate([audio_buffer, block])
                        chunk = _build_live_chunk(audio_buffer, chunk_samples)
                        audio_buffer = _trim_live_buffer(
                            audio_buffer, chunk_samples, step_samples
                        )
                    result = pipeline.step(chunk)
                    broadcast_step_result(result)
                    drained = True
                except queue.Empty:
                    break

            if not drained:
                time.sleep(0.05)

    except Exception as e:
        log.error("Monitor error: %s", e, exc_info=True)
        ws_event("error", message=f"Monitor error: {e}")
    finally:
        mic_stream.stop()
        mic_stream.close()
        if capture is not None:
            saved_dir = capture.finalize()
            monitor.last_capture_dir = str(saved_dir)
            log.info("Live session capture saved: %s", saved_dir)
            ws_event("status", message=f"Capture saved: {saved_dir.name}")
        monitor.running = False
        ws_event("status", message="Monitor stopped.")
        log.info("Diarization monitor stopped.")


# ─── Enrollment recording ────────────────────────────────────────

_enroll_state: Dict[str, dict] = {}


def _do_enrollment(
    name: str,
    duration_sec: int,
    reference_label: Optional[str],
    cfg: dict,
):
    """Record mic audio and rebuild the speaker cache from reference audio."""
    enroll_id = str(uuid.uuid4())[:8]
    _enroll_state[name] = {"status": "recording", "id": enroll_id}

    sample_rate = cfg["audio"]["sample_rate"]
    device_index = cfg["audio"]["device_index"]
    speaker_dir = get_enrollment_dir(cfg) / name
    speaker_dir.mkdir(parents=True, exist_ok=True)
    wav_path = _speaker_reference_path(speaker_dir, reference_label)
    label_suffix = f" [{reference_label.strip()}]" if (reference_label or "").strip() else ""

    ws_event("enrollment", name=name, status="recording",
             message=f"Recording {duration_sec}s for '{name}'{label_suffix}...")

    try:
        # Record
        log.info("Enrolling '%s': recording %ds...", name, duration_sec)
        audio = sd.rec(
            int(duration_sec * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
        audio = audio.flatten()

        # Save reference wav
        sf.write(str(wav_path), audio, sample_rate)
        log.info("Saved reference audio: %s", wav_path)

        ws_event("enrollment", name=name, status="extracting",
                 message=f"Rebuilding enrollment cache for {name}...")

        embedding = rebuild_speaker_embedding_cache(speaker_dir, cfg)
        if embedding is None:
            raise RuntimeError("No usable reference audio found after enrollment")

        _enroll_state[name] = {"status": "done", "id": enroll_id}
        ws_event("enrollment", name=name, status="done",
                 message=f"Saved {wav_path.name} for '{name}'.")

    except Exception as e:
        log.error("Enrollment failed for '%s': %s", name, e, exc_info=True)
        _enroll_state[name] = {"status": "error", "id": enroll_id, "error": str(e)}
        ws_event("enrollment", name=name, status="error",
                 message=f"Enrollment failed: {e}")


# ─── API endpoints ────────────────────────────────────────────────

class StartRequest(BaseModel):
    pass


class ConfigUpdateRequest(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict)


class EnrollRequest(BaseModel):
    name: str
    duration: int = 30
    reference_label: str | None = None


class DiagRecordRequest(BaseModel):
    label: str
    duration: int = 5


class PromoteDiagRequest(BaseModel):
    speaker_name: str
    clip_name: str
    reference_label: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "dashboard.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Voice ID</h1><p>dashboard.html not found</p>")


@app.post("/api/start")
async def api_start():
    if monitor.running:
        raise HTTPException(400, "Monitor already running")
    monitor.reset()
    monitor.running = True
    monitor.thread = threading.Thread(target=_run_diarization_monitor, daemon=True)
    monitor.thread.start()
    return {"status": "started"}


@app.post("/api/stop")
async def api_stop():
    if not monitor.running:
        raise HTTPException(400, "Monitor not running")
    monitor.stop_event.set()
    return {"status": "stopping"}


@app.get("/api/status")
async def api_status():
    return {
        "running": monitor.running,
        "step": monitor.step_count,
        "infer_ms": round(monitor.last_infer_ms, 1),
        "last_capture_dir": monitor.last_capture_dir,
    }


@app.get("/api/speakers")
async def api_speakers():
    cfg = load_config()
    return list_speakers(cfg)


@app.delete("/api/speakers/{name}")
async def api_delete_speaker(name: str):
    cfg = load_config()
    speaker_dir = get_enrollment_dir(cfg) / name
    if not speaker_dir.exists():
        raise HTTPException(404, f"Speaker '{name}' not found")
    shutil.rmtree(speaker_dir)
    return {"status": "deleted", "name": name}


@app.delete("/api/speakers/{name}/references/{filename}")
async def api_delete_reference(name: str, filename: str):
    if monitor.running:
        raise HTTPException(400, "Stop the monitor before editing references")
    cfg = load_config()
    speaker_dir = get_enrollment_dir(cfg) / name
    ref_path = speaker_dir / filename
    if not speaker_dir.exists() or not ref_path.exists():
        raise HTTPException(404, f"Reference '{filename}' for '{name}' not found")
    if ref_path.suffix.lower() != ".wav" or not ref_path.name.startswith("reference"):
        raise HTTPException(404, f"Reference '{filename}' for '{name}' not found")
    ref_path.unlink()
    rebuilt = rebuild_speaker_embedding_cache(speaker_dir, cfg)
    if rebuilt is None:
        cache_path = embedding_cache_path(speaker_dir)
        if cache_path.exists():
            cache_path.unlink()
        if speaker_dir.exists() and not any(speaker_dir.iterdir()):
            speaker_dir.rmdir()
    return {
        "status": "deleted",
        "speaker_name": name,
        "reference_name": filename,
    }


@app.post("/api/diag/record")
async def api_diag_record(req: DiagRecordRequest):
    """Record a labeled diagnostic clip from the mic and save to disk."""
    cfg = load_config()
    sr = cfg["audio"]["sample_rate"]
    dev = cfg["audio"]["device_index"]
    diag_dir = BASE_DIR / "diagnostic_clips"
    diag_dir.mkdir(exist_ok=True)

    label = req.label.strip().replace(" ", "_")
    duration = max(2, min(15, req.duration))

    def _record():
        try:
            ws_event("diag", status="recording",
                     message=f"Recording '{label}' for {duration}s...")
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1,
                           dtype="float32", device=dev)
            sd.wait()
            audio = audio[:, 0]

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{label}_{ts}.wav"
            fpath = diag_dir / fname
            sf.write(str(fpath), audio, sr)

            ws_event("diag", status="done",
                     message=f"Saved '{fname}' ({len(audio)/sr:.1f}s, peak={np.max(np.abs(audio)):.3f})")
            log.info("Diagnostic clip saved: %s", fpath)
        except Exception as e:
            ws_event("diag", status="error", message=f"Recording failed: {e}")
            log.error("Diag record failed: %s", e)

    thread = threading.Thread(target=_record, daemon=True)
    thread.start()
    return {"status": "recording", "label": label, "duration": duration}


@app.get("/api/diag/clips")
async def api_diag_clips():
    """List all diagnostic clips."""
    diag_dir = BASE_DIR / "diagnostic_clips"
    if not diag_dir.exists():
        return []
    return [_serialize_clip(path) for path in sorted(diag_dir.glob("*.wav"))]


@app.delete("/api/diag/clips/{name}")
async def api_delete_diag_clip(name: str):
    if monitor.running:
        raise HTTPException(400, "Stop the monitor before editing diagnostic clips")
    diag_dir = BASE_DIR / "diagnostic_clips"
    path = diag_dir / name
    if not path.exists():
        raise HTTPException(404, f"Diagnostic clip '{name}' not found")
    path.unlink()
    return {"status": "deleted", "name": name}


@app.post("/api/diag/promote")
async def api_diag_promote(req: PromoteDiagRequest):
    if monitor.running:
        raise HTTPException(400, "Stop the monitor before editing references")
    speaker_name = req.speaker_name.strip()
    if not speaker_name:
        raise HTTPException(400, "Speaker name is required")

    diag_dir = BASE_DIR / "diagnostic_clips"
    source_path = diag_dir / req.clip_name
    if not source_path.exists():
        raise HTTPException(404, f"Diagnostic clip '{req.clip_name}' not found")

    cfg = load_config()
    speaker_dir = get_enrollment_dir(cfg) / speaker_name
    speaker_dir.mkdir(parents=True, exist_ok=True)
    effective_label = (req.reference_label or "").strip() or _clip_label_from_name(source_path)
    target_path = _speaker_reference_path(speaker_dir, effective_label)
    shutil.copy2(source_path, target_path)
    rebuilt = rebuild_speaker_embedding_cache(speaker_dir, cfg)
    if rebuilt is None:
        raise HTTPException(500, "Failed to rebuild speaker cache from references")
    return {
        "status": "promoted",
        "speaker_name": speaker_name,
        "reference_name": target_path.name,
        "source_clip": req.clip_name,
    }


@app.post("/api/enroll")
async def api_enroll(req: EnrollRequest):
    if monitor.running:
        raise HTTPException(400, "Stop the monitor before enrolling")
    cfg = load_config()
    thread = threading.Thread(
        target=_do_enrollment,
        args=(req.name, req.duration, req.reference_label, cfg),
        daemon=True,
    )
    thread.start()
    return {
        "status": "enrolling",
        "name": req.name,
        "duration": req.duration,
        "reference_label": req.reference_label,
    }


@app.get("/api/enroll/status/{name}")
async def api_enroll_status(name: str):
    if name in _enroll_state:
        return _enroll_state[name]
    return {"status": "unknown"}


@app.get("/api/config")
async def api_config():
    return load_config()


@app.get("/api/config/editor")
async def api_config_editor():
    return _config_editor_payload()


@app.put("/api/config")
async def api_update_config(req: ConfigUpdateRequest):
    if monitor.running:
        raise HTTPException(400, "Stop the monitor before changing settings")

    base = load_base_config()
    overrides = load_override_config()

    try:
        for path, raw_value in req.values.items():
            meta = CONFIG_EDITOR_FIELD_MAP.get(path)
            if meta is None:
                raise ValueError(f"Unsupported setting: {path}")
            value = _coerce_editor_value(raw_value, meta)
            base_value = _config_get(base, path)
            if value == base_value:
                _config_delete(overrides, path)
            else:
                _config_set(overrides, path, value)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))

    if overrides:
        with open(OVERRIDE_CONFIG_PATH, "w") as f:
            yaml.safe_dump(overrides, f, sort_keys=False)
    elif OVERRIDE_CONFIG_PATH.exists():
        OVERRIDE_CONFIG_PATH.unlink()

    return _config_editor_payload()


@app.delete("/api/config")
async def api_reset_config():
    if monitor.running:
        raise HTTPException(400, "Stop the monitor before resetting settings")
    if OVERRIDE_CONFIG_PATH.exists():
        OVERRIDE_CONFIG_PATH.unlink()
    return _config_editor_payload()


# ─── WebSocket ────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    ws_clients.add(ws)
    log.info("WebSocket client connected (%d total)", len(ws_clients))
    try:
        while True:
            # Wait for any message (client sends "ping" as keepalive)
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug("WebSocket error: %s", e)
    finally:
        ws_clients.discard(ws)
        log.info("WebSocket client disconnected (%d total)", len(ws_clients))


# ─── Startup ──────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    global loop
    loop = asyncio.get_event_loop()
    cfg = load_config()
    log.info("Voice ID server starting on %s:%d", cfg["server"]["host"], cfg["server"]["port"])


if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "server:app",
        host=cfg["server"]["host"],
        port=cfg["server"]["port"],
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
