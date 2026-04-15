from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import queue
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from fastapi import WebSocket, WebSocketDisconnect

from enrollment_store import (
    embedding_cache_path,
    rebuild_speaker_embedding_cache,
    reference_paths,
)
import server as voice_source
from pipeline import RealtimePipeline
from .brain.event_bus import Event, EventBus


try:
    LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    LIBC = None


def trim_process_heap() -> None:
    if LIBC is None:
        return
    try:
        LIBC.malloc_trim(0)
    except Exception:
        pass


class VoiceService:
    def __init__(self, gpu_lock: threading.Lock, policy):
        self.gpu_lock = gpu_lock
        self.policy = policy
        self.monitor = voice_source.MonitorState()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.event_bus: Optional[EventBus] = None
        self.ws_clients: Set[WebSocket] = set()
        self._enroll_state: Dict[str, dict] = {}
        self._pipeline = None
        self._mic_stream = None
        self._cleanup_lock = threading.Lock()
        self._brain_active_speakers: Set[str] = set()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def set_event_bus(self, bus: EventBus) -> None:
        self.event_bus = bus

    def _publish_brain_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish_sync(Event(type=event_type, data=data), loop=self.loop)

    def _sync_brain_speakers(self, speakers: List[dict]) -> None:
        active_now: Set[str] = set()
        for speaker in speakers:
            if not speaker.get("active"):
                continue
            who = str(speaker.get("label") or "unknown").strip() or "unknown"
            active_now.add(who)
            self._publish_brain_event(
                "speaker_active",
                {
                    "who": who,
                    "enrolled": bool(speaker.get("enrolled")),
                    "confidence": float(
                        speaker.get("identity_similarity")
                        if speaker.get("identity_similarity") is not None
                        else speaker.get("activity", 0.0)
                    ),
                },
            )

        for who in sorted(self._brain_active_speakers - active_now):
            self._publish_brain_event("speaker_silent", {"who": who})
        self._brain_active_speakers = active_now

    async def connect_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        self.ws_clients.add(ws)

    def disconnect_ws(self, ws: WebSocket) -> None:
        self.ws_clients.discard(ws)

    async def _broadcast_text(self, payload: str) -> None:
        dead = set()
        for ws in self.ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self.ws_clients.difference_update(dead)

    async def _broadcast_diarization(self, payload: str, audio_packets: list[bytes]) -> None:
        dead = set()
        for ws in self.ws_clients:
            try:
                await ws.send_text(payload)
                for packet in audio_packets:
                    await ws.send_bytes(packet)
            except Exception:
                dead.add(ws)
        self.ws_clients.difference_update(dead)

    def _send_json(self, data: dict) -> None:
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_text(json.dumps(data)),
            self.loop,
        )

    def _send_diarization(self, data: dict, audio_packets: list[bytes]) -> None:
        if self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(
            self._broadcast_diarization(json.dumps(data), audio_packets),
            self.loop,
        )

    def ws_event(self, event_type: str, **kwargs: Any) -> None:
        self._send_json({"type": event_type, **kwargs})

    def config_payload(self) -> dict:
        return {
            "config": voice_source.load_config(),
            "fields": voice_source.CONFIG_EDITOR_FIELDS,
            "override_file": voice_source.OVERRIDE_CONFIG_PATH.name,
            "has_overrides": voice_source.OVERRIDE_CONFIG_PATH.exists(),
        }

    def update_config(self, values: Dict[str, Any]) -> dict:
        if self.monitor.running:
            raise ValueError("Stop Voice before changing settings")

        base = voice_source.load_base_config()
        overrides = voice_source.load_override_config()
        for path, raw_value in values.items():
            meta = voice_source.CONFIG_EDITOR_FIELD_MAP.get(path)
            if meta is None:
                raise ValueError(f"Unsupported setting: {path}")
            value = voice_source._coerce_editor_value(raw_value, meta)
            base_value = voice_source._config_get(base, path)
            if value == base_value:
                voice_source._config_delete(overrides, path)
            else:
                voice_source._config_set(overrides, path, value)

        if overrides:
            with open(voice_source.OVERRIDE_CONFIG_PATH, "w", encoding="utf-8") as handle:
                voice_source.yaml.safe_dump(overrides, handle, sort_keys=False)
        elif voice_source.OVERRIDE_CONFIG_PATH.exists():
            voice_source.OVERRIDE_CONFIG_PATH.unlink()
        return self.config_payload()

    def reset_config(self) -> dict:
        if self.monitor.running:
            raise ValueError("Stop Voice before resetting settings")
        if voice_source.OVERRIDE_CONFIG_PATH.exists():
            voice_source.OVERRIDE_CONFIG_PATH.unlink()
        return self.config_payload()

    def status(self) -> dict:
        return {
            "running": self.monitor.running,
            "step": self.monitor.step_count if self.monitor.running else 0,
            "infer_ms": round(self.monitor.last_infer_ms, 1) if self.monitor.running else None,
            "last_capture_dir": self.monitor.last_capture_dir,
        }

    def _sanitize_slug(self, value: str, default: str = "clip") -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
        return slug or default

    def _timestamp_suffix_removed(self, stem: str) -> str:
        return re.sub(r"_(\d{8}_\d{6})$", "", stem)

    def _clip_label_from_name(self, path: Path, prefix: str = "") -> str:
        stem = path.stem
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix):]
            if stem.startswith("_"):
                stem = stem[1:]
        stem = self._timestamp_suffix_removed(stem)
        return stem or "base"

    def _clip_duration_sec(self, path: Path) -> float:
        try:
            info = sf.info(str(path))
            if not info.samplerate:
                return 0.0
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return 0.0

    def _serialize_clip(self, path: Path, prefix: str = "") -> dict:
        stat = path.stat()
        return {
            "name": path.name,
            "label": self._clip_label_from_name(path, prefix),
            "size": stat.st_size,
            "duration_sec": round(self._clip_duration_sec(path), 3),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }

    def _speaker_reference_path(
        self,
        speaker_dir: Path,
        reference_label: Optional[str],
    ) -> Path:
        label = (reference_label or "").strip()
        if label:
            return speaker_dir / f"reference_{self._sanitize_slug(label)}.wav"
        existing = sorted(speaker_dir.glob("reference*.wav"))
        if not existing:
            return speaker_dir / "reference.wav"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return speaker_dir / f"reference_{timestamp}.wav"

    def list_speakers(self) -> list[dict]:
        cfg = voice_source.load_config()
        enroll_dir = voice_source.get_enrollment_dir(cfg)
        speakers: List[dict] = []
        if not enroll_dir.exists():
            return speakers

        for speaker_dir in sorted(enroll_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            ref_files = reference_paths(speaker_dir)
            if not ref_files:
                continue
            references = [
                self._serialize_clip(path, prefix="reference")
                for path in ref_files
            ]
            has_emb = embedding_cache_path(speaker_dir).exists()
            speakers.append({
                "name": speaker_dir.name,
                "has_embedding": has_emb,
                "has_audio": bool(references),
                "reference_count": len(references),
                "total_reference_sec": round(
                    sum(item["duration_sec"] for item in references),
                    3,
                ),
                "references": references,
            })
        return speakers

    def delete_speaker(self, name: str) -> dict:
        if self.monitor.running:
            raise ValueError("Stop Voice before deleting speakers")
        cfg = voice_source.load_config()
        speaker_dir = voice_source.get_enrollment_dir(cfg) / name
        if not speaker_dir.exists():
            raise FileNotFoundError(name)
        shutil.rmtree(speaker_dir)
        return {"status": "deleted", "name": name}

    def delete_reference(self, speaker_name: str, filename: str) -> dict:
        if self.monitor.running:
            raise ValueError("Stop Voice before editing references")
        cfg = voice_source.load_config()
        speaker_dir = voice_source.get_enrollment_dir(cfg) / speaker_name
        ref_path = speaker_dir / filename
        if not speaker_dir.exists() or not ref_path.exists():
            raise FileNotFoundError(f"Reference '{filename}' for '{speaker_name}' not found")
        if ref_path.suffix.lower() != ".wav" or not ref_path.name.startswith("reference"):
            raise FileNotFoundError(f"Reference '{filename}' for '{speaker_name}' not found")
        ref_path.unlink()
        rebuilt = rebuild_speaker_embedding_cache(speaker_dir, cfg)
        if rebuilt is None:
            cache_path = embedding_cache_path(speaker_dir)
            if cache_path.exists():
                cache_path.unlink()
            if not any(speaker_dir.iterdir()):
                speaker_dir.rmdir()
        return {
            "status": "deleted",
            "speaker_name": speaker_name,
            "reference_name": filename,
        }

    def diag_clips(self) -> list[dict]:
        diag_dir = voice_source.BASE_DIR / "diagnostic_clips"
        if not diag_dir.exists():
            return []
        return [
            self._serialize_clip(path)
            for path in sorted(diag_dir.glob("*.wav"))
        ]

    def delete_diag_clip(self, name: str) -> dict:
        if self.monitor.running:
            raise ValueError("Stop Voice before editing diagnostic clips")
        diag_dir = voice_source.BASE_DIR / "diagnostic_clips"
        path = diag_dir / name
        if not path.exists():
            raise FileNotFoundError(name)
        path.unlink()
        return {"status": "deleted", "name": name}

    def promote_diag_clip(
        self,
        speaker_name: str,
        clip_name: str,
        reference_label: Optional[str] = None,
    ) -> dict:
        if self.monitor.running:
            raise ValueError("Stop Voice before editing references")
        speaker_name = speaker_name.strip()
        if not speaker_name:
            raise ValueError("Speaker name is required")

        diag_dir = voice_source.BASE_DIR / "diagnostic_clips"
        source_path = diag_dir / clip_name
        if not source_path.exists():
            raise FileNotFoundError(f"Diagnostic clip '{clip_name}' not found")

        cfg = voice_source.load_config()
        speaker_dir = voice_source.get_enrollment_dir(cfg) / speaker_name
        speaker_dir.mkdir(parents=True, exist_ok=True)

        effective_label = (reference_label or "").strip() or self._clip_label_from_name(source_path)
        target_path = self._speaker_reference_path(speaker_dir, effective_label)
        shutil.copy2(source_path, target_path)

        rebuilt = rebuild_speaker_embedding_cache(speaker_dir, cfg)
        if rebuilt is None:
            raise RuntimeError("Failed to rebuild speaker cache from references")
        return {
            "status": "promoted",
            "speaker_name": speaker_name,
            "reference_name": target_path.name,
            "source_clip": clip_name,
        }

    def record_diag(self, label: str, duration: int) -> dict:
        cfg = voice_source.load_config()
        sample_rate = cfg["audio"]["sample_rate"]
        device_index = cfg["audio"]["device_index"]
        diag_dir = voice_source.BASE_DIR / "diagnostic_clips"
        diag_dir.mkdir(exist_ok=True)

        safe_label = label.strip().replace(" ", "_")
        duration = max(2, min(15, duration))

        def _record() -> None:
            try:
                self.ws_event(
                    "diag",
                    status="recording",
                    message=f"Recording '{safe_label}' for {duration}s...",
                )
                audio = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                    device=device_index,
                )
                sd.wait()
                audio = audio[:, 0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{safe_label}_{timestamp}.wav"
                path = diag_dir / filename
                sf.write(str(path), audio, sample_rate)
                self.ws_event(
                    "diag",
                    status="done",
                    message=(
                        f"Saved '{filename}' ({len(audio) / sample_rate:.1f}s, "
                        f"peak={np.max(np.abs(audio)):.3f})"
                    ),
                )
            except Exception as exc:
                self.ws_event("diag", status="error", message=f"Recording failed: {exc}")

        threading.Thread(target=_record, daemon=True).start()
        return {"status": "recording", "label": safe_label, "duration": duration}

    def enroll(
        self,
        name: str,
        duration: int,
        reference_label: Optional[str] = None,
    ) -> dict:
        if self.monitor.running:
            raise ValueError("Stop Voice before enrolling")
        cfg = voice_source.load_config()
        thread = threading.Thread(
            target=self._do_enrollment,
            args=(name, duration, reference_label, cfg),
            daemon=True,
        )
        thread.start()
        return {
            "status": "enrolling",
            "name": name,
            "duration": duration,
            "reference_label": reference_label,
        }

    def enroll_status(self, name: str) -> dict:
        return self._enroll_state.get(name, {"status": "unknown"})

    def _do_enrollment(
        self,
        name: str,
        duration_sec: int,
        reference_label: Optional[str],
        cfg: dict,
    ) -> None:
        enroll_id = str(voice_source.uuid.uuid4())[:8]
        self._enroll_state[name] = {"status": "recording", "id": enroll_id}

        sample_rate = cfg["audio"]["sample_rate"]
        device_index = cfg["audio"]["device_index"]
        speaker_dir = voice_source.get_enrollment_dir(cfg) / name
        speaker_dir.mkdir(parents=True, exist_ok=True)
        wav_path = self._speaker_reference_path(speaker_dir, reference_label)
        label_suffix = f" [{reference_label.strip()}]" if (reference_label or "").strip() else ""

        self.ws_event(
            "enrollment",
            name=name,
            status="recording",
            message=f"Recording {duration_sec}s for '{name}'{label_suffix}...",
        )

        try:
            audio = sd.rec(
                int(duration_sec * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device_index,
            )
            sd.wait()
            audio = audio.flatten()

            sf.write(str(wav_path), audio, sample_rate)

            self.ws_event(
                "enrollment",
                name=name,
                status="extracting",
                message=f"Rebuilding enrollment cache for {name}...",
            )

            rebuilt = rebuild_speaker_embedding_cache(speaker_dir, cfg)
            if rebuilt is None:
                raise RuntimeError("Failed to rebuild speaker cache from references")

            self._enroll_state[name] = {"status": "done", "id": enroll_id}
            self.ws_event(
                "enrollment",
                name=name,
                status="done",
                message=f"Saved {wav_path.name} for '{name}'.",
            )
        except Exception as exc:
            self._enroll_state[name] = {
                "status": "error",
                "id": enroll_id,
                "error": str(exc),
            }
            self.ws_event(
                "enrollment",
                name=name,
                status="error",
                message=f"Enrollment failed: {exc}",
            )

    def start(self) -> dict:
        if self.monitor.running:
            raise ValueError("Voice is already running")
        self._brain_active_speakers.clear()
        self.monitor.reset()
        self.monitor.running = True
        self._publish_brain_event("system_status", {"voice_running": True})
        self.monitor.thread = threading.Thread(target=self._run_monitor, daemon=True)
        self.monitor.thread.start()
        return {"status": "started"}

    def stop(self, wait: bool = True, timeout: float = 15.0) -> dict:
        if not self.monitor.running:
            raise ValueError("Voice is not running")
        self.ws_event("status", message="Stopping Voice...")
        self.monitor.stop_event.set()
        mic_stream = self._mic_stream
        if mic_stream is not None:
            try:
                mic_stream.abort()
            except Exception:
                pass
            try:
                mic_stream.stop()
            except Exception:
                pass
        if wait and self.monitor.thread and self.monitor.thread.is_alive():
            self.monitor.thread.join(timeout=timeout)
        self._finalize_runtime_refs()
        return {"status": "stopped" if not self.monitor.running else "stopping"}

    def shutdown(self, timeout: float = 15.0) -> None:
        if self.monitor.running:
            self.monitor.stop_event.set()
        mic_stream = self._mic_stream
        if mic_stream is not None:
            try:
                mic_stream.abort()
            except Exception:
                pass
            try:
                mic_stream.stop()
            except Exception:
                pass
        if self.monitor.thread and self.monitor.thread.is_alive():
            self.monitor.thread.join(timeout=timeout)
        self._finalize_runtime_refs()

    def _finalize_runtime_refs(self) -> None:
        with self._cleanup_lock:
            self._pipeline = None
            self._mic_stream = None
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
            trim_process_heap()

    def _run_monitor(self) -> None:
        cfg = voice_source.load_config()
        dbg_cfg = cfg.setdefault("debug", {})
        # Keep the dual dashboard quiet and lightweight by default.
        dbg_cfg["step_perf_log"] = False
        dbg_cfg["step_perf_every_step"] = False
        dbg_cfg["capture_live_session"] = False
        sample_rate = cfg["audio"]["sample_rate"]
        device_index = cfg["audio"]["device_index"]
        duration = cfg["pixit"]["duration"]
        step = cfg["pixit"]["step"]

        chunk_samples = int(duration * sample_rate)
        step_samples = int(step * sample_rate)
        capture: Optional[voice_source.LiveSessionCapture] = None
        if dbg_cfg.get("capture_live_session", True):
            capture_root = voice_source.BASE_DIR / dbg_cfg.get(
                "live_capture_dir",
                "logs/live_sessions",
            )
            capture = voice_source.LiveSessionCapture(
                capture_root,
                sample_rate,
                step_samples,
            )

        enrolled = voice_source.load_enrolled_embeddings(cfg)
        self.ws_event("status", message="Loading Voice models...")

        try:
            with self.gpu_lock:
                pipeline = RealtimePipeline(cfg, enrolled)
                self._pipeline = pipeline
        except Exception as exc:
            self.ws_event("error", message=f"Voice init failed: {exc}")
            self.monitor.running = False
            self._publish_brain_event("system_status", {"voice_running": False})
            self._finalize_runtime_refs()
            return

        self.ws_event("status", message="Voice ready. Starting mic capture...")

        audio_buffer = np.zeros(0, dtype=np.float32)
        mic_queue: queue.Queue[np.ndarray] = queue.Queue()

        def mic_callback(indata, frames, time_info, status) -> None:
            if status:
                voice_source.log.warning("Mic status: %s", status)
            mic_queue.put(indata[:, 0].copy())

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
            self._mic_stream = mic_stream
        except Exception as exc:
            self.ws_event("error", message=f"Voice mic failed: {exc}")
            self.monitor.running = False
            self._publish_brain_event("system_status", {"voice_running": False})
            self._finalize_runtime_refs()
            return

        self.ws_event("status", message="Voice live.")

        def broadcast_step_result(result) -> None:
            self.monitor.step_count = result.step_idx
            self.monitor.last_infer_ms = result.infer_ms
            self.policy.record_voice_step(result.infer_ms)

            if capture is not None:
                capture.record_step(result)

            speakers_data = []
            audio_packets = []
            for sp in result.speakers:
                audio_i16 = np.clip(sp.audio * 32767, -32768, 32767).astype(np.int16)
                speaker_id = int(sp.global_idx)
                audio_data = np.asarray(sp.audio, dtype=np.float32)
                audio_peak = float(np.max(np.abs(audio_data))) if audio_data.size else 0.0
                audio_rms = (
                    float(np.sqrt(np.mean(np.asarray(sp.audio, dtype=np.float64) ** 2)))
                    if np.asarray(sp.audio).size
                    else 0.0
                )
                audio_active = bool(audio_peak > (32.0 / 32767.0))
                diar_active = bool(sp.activity > cfg["clustering"]["tau_active"])

                item = {
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
                    item["identity_similarity"] = round(float(sp.identity_similarity), 4)
                speakers_data.append(item)
                audio_packets.append(
                    voice_source._pack_audio_packet(
                        result.step_idx,
                        speaker_id,
                        sample_rate,
                        audio_i16,
                    )
                )

            self._send_diarization(
                {
                    "type": "diarization",
                    "speakers": speakers_data,
                    "step": result.step_idx,
                    "infer_ms": round(result.infer_ms, 1),
                    "sample_rate": sample_rate,
                    "samples": pipeline.step_samples,
                },
                audio_packets,
            )
            self._sync_brain_speakers(speakers_data)

        try:
            while not self.monitor.stop_event.is_set():
                drained = False
                while not mic_queue.empty():
                    try:
                        block = mic_queue.get_nowait()
                    except queue.Empty:
                        break

                    block = pipeline.denoise_block(block)
                    if capture is not None:
                        capture.record_mic_block(block)

                    audio_buffer = np.concatenate([audio_buffer, block])
                    chunk = voice_source._build_live_chunk(audio_buffer, chunk_samples)
                    audio_buffer = voice_source._trim_live_buffer(
                        audio_buffer,
                        chunk_samples,
                        step_samples,
                    )
                    with self.gpu_lock:
                        result = pipeline.step(chunk)
                    broadcast_step_result(result)
                    drained = True

                if not drained:
                    time.sleep(0.05)
        except Exception as exc:
            self.ws_event("error", message=f"Voice monitor error: {exc}")
        finally:
            try:
                mic_stream.stop()
            except Exception:
                pass
            try:
                mic_stream.close()
            except Exception:
                pass
            if capture is not None:
                saved_dir = capture.finalize()
                self.monitor.last_capture_dir = str(saved_dir)
                self.ws_event("status", message=f"Voice capture saved: {saved_dir.name}")
            self.monitor.running = False
            for who in sorted(self._brain_active_speakers):
                self._publish_brain_event("speaker_silent", {"who": who})
            self._brain_active_speakers.clear()
            self._publish_brain_event("system_status", {"voice_running": False})
            self._pipeline = None
            self._mic_stream = None
            self._finalize_runtime_refs()
            self.ws_event("status", message="Voice stopped.")

    async def websocket_loop(self, ws: WebSocket) -> None:
        await self.connect_ws(ws)
        try:
            while True:
                message = await ws.receive_text()
                if message == "ping":
                    await ws.send_text('{"type":"pong"}')
        except WebSocketDisconnect:
            pass
        finally:
            self.disconnect_ws(ws)
