#!/usr/bin/env python3
"""
Voice ID — Real-Time Diarization & Separation Server

FastAPI backend that:
  - Loads the PixIT + WeSpeaker + EnrolledClustering pipeline
  - Captures mic audio via sounddevice
  - Runs a rolling-buffer diarization loop in a background thread
  - Broadcasts per-speaker audio + metadata over WebSocket
  - Handles speaker enrollment (record → extract WeSpeaker embedding → save)

Playback continuity (crossfade, holdover, tau_active tuning) is implemented
inside ``RealtimePipeline.step()`` — this server only feeds fixed-size chunks
and forwards ``StepResult`` frames; see ``pipeline.py`` module docstring.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from pipeline import RealtimePipeline

# ─── Globals ──────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("server")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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

    def reset(self):
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
        self.step_count = 0
        self.last_infer_ms = 0.0


monitor = MonitorState()
ws_clients: Set[WebSocket] = set()
AUDIO_PACKET_MAGIC = b"VAUD"


# ─── Enrollment helpers ───────────────────────────────────────────

def get_enrollment_dir(cfg: dict) -> Path:
    return BASE_DIR / cfg["enrollment"]["dir"]


def load_enrolled_embeddings(cfg: dict) -> Dict[str, np.ndarray]:
    """Load all enrolled speaker embeddings from disk."""
    enroll_dir = get_enrollment_dir(cfg)
    enrolled = {}
    if enroll_dir.exists():
        for speaker_dir in sorted(enroll_dir.iterdir()):
            emb_path = speaker_dir / "embedding.npy"
            if speaker_dir.is_dir() and emb_path.exists():
                enrolled[speaker_dir.name] = np.load(emb_path)
    return enrolled


def list_speakers(cfg: dict) -> List[dict]:
    """Return info about enrolled speakers."""
    enroll_dir = get_enrollment_dir(cfg)
    speakers = []
    if enroll_dir.exists():
        for speaker_dir in sorted(enroll_dir.iterdir()):
            if speaker_dir.is_dir():
                has_emb = (speaker_dir / "embedding.npy").exists()
                has_wav = (speaker_dir / "reference.wav").exists()
                speakers.append({
                    "name": speaker_dir.name,
                    "has_embedding": has_emb,
                    "has_audio": has_wav,
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
    """Pack a speaker audio chunk into a binary websocket frame."""
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
    sample_rate = cfg["audio"]["sample_rate"]
    device_index = cfg["audio"]["device_index"]
    duration = cfg["pixit"]["duration"]
    step = cfg["pixit"]["step"]

    chunk_samples = int(duration * sample_rate)    # 80000
    step_samples = int(step * sample_rate)         # 8000

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

        speakers_data = []
        audio_packets = []
        for sp in result.speakers:
            audio_i16 = np.clip(sp.audio * 32767, -32768, 32767).astype(np.int16)
            speaker_id = int(sp.global_idx)
            _sd = {
                "id": speaker_id,
                "label": str(sp.label),
                "active": bool(sp.activity > cfg["clustering"]["tau_active"]),
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
            _n_queued = mic_queue.qsize()
            while not mic_queue.empty():
                try:
                    block = mic_queue.get_nowait()
                    block = pipeline.denoise_block(block)  # ~8ms for 500ms
                    with buf_lock:
                        audio_buffer = np.concatenate([audio_buffer, block])
                        _buf_len = len(audio_buffer)
                        chunk = _build_live_chunk(audio_buffer, chunk_samples)
                        audio_buffer = _trim_live_buffer(
                            audio_buffer, chunk_samples, step_samples
                        )
                    _t_step_start = time.perf_counter()
                    result = pipeline.step(chunk)
                    _t_step_ms = (time.perf_counter() - _t_step_start) * 1000
                    broadcast_step_result(result)
                    drained = True
                    # #region agent log — timing + buffer health
                    if monitor.step_count % 4 == 0:
                        import json as _j
                        open('/home/michel/Documents/Voice/.cursor/debug-21cffc.log','a').write(_j.dumps({"sessionId":"21cffc","hypothesisId":"TIMING","location":"server.py:loop","message":"STEP_TIMING","data":{"step":monitor.step_count,"queued_blocks":_n_queued,"buf_before_trim":_buf_len,"step_ms":round(_t_step_ms,1),"step_budget_ms":step*1000},"timestamp":int(time.time()*1000)})+'\n')
                    # #endregion
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
        monitor.running = False
        ws_event("status", message="Monitor stopped.")
        log.info("Diarization monitor stopped.")


# ─── Enrollment recording ────────────────────────────────────────

_enroll_state: Dict[str, dict] = {}


def _do_enrollment(name: str, duration_sec: int, cfg: dict):
    """Record mic audio and extract WeSpeaker embedding for enrollment."""
    enroll_id = str(uuid.uuid4())[:8]
    _enroll_state[name] = {"status": "recording", "id": enroll_id}

    sample_rate = cfg["audio"]["sample_rate"]
    device_index = cfg["audio"]["device_index"]
    speaker_dir = get_enrollment_dir(cfg) / name
    speaker_dir.mkdir(parents=True, exist_ok=True)

    ws_event("enrollment", name=name, status="recording",
             message=f"Recording {duration_sec}s...")

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
        wav_path = speaker_dir / "reference.wav"
        sf.write(str(wav_path), audio, sample_rate)
        log.info("Saved reference audio: %s", wav_path)

        ws_event("enrollment", name=name, status="extracting",
                 message="Extracting embedding...")

        # Extract embedding using the same pyannote model as the live pipeline
        from pyannote.audio import Model as PyanModel
        device = torch.device(cfg["device"])
        emb_model = PyanModel.from_pretrained(cfg["embedding"]["model"]).to(device)
        emb_model.eval()

        # Peak-normalize + extract (same logic as pipeline.extract_embedding)
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio_norm = audio / peak
        else:
            audio_norm = audio
        waveform = torch.from_numpy(audio_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = emb_model(waveform)  # (1, emb_dim)

        embedding = embedding.squeeze(0).cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Save
        emb_path = speaker_dir / "embedding.npy"
        np.save(emb_path, embedding)
        log.info("Saved embedding: %s (dim=%d)", emb_path, len(embedding))

        _enroll_state[name] = {"status": "done", "id": enroll_id}
        ws_event("enrollment", name=name, status="done",
                 message=f"Enrolled '{name}' successfully.")

    except Exception as e:
        log.error("Enrollment failed for '%s': %s", name, e, exc_info=True)
        _enroll_state[name] = {"status": "error", "id": enroll_id, "error": str(e)}
        ws_event("enrollment", name=name, status="error",
                 message=f"Enrollment failed: {e}")


# ─── API endpoints ────────────────────────────────────────────────

class StartRequest(BaseModel):
    pass


class EnrollRequest(BaseModel):
    name: str
    duration: int = 10


class DiagRecordRequest(BaseModel):
    label: str
    duration: int = 5


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
    import shutil
    shutil.rmtree(speaker_dir)
    return {"status": "deleted", "name": name}


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
    clips = []
    for f in sorted(diag_dir.glob("*.wav")):
        clips.append({"name": f.name, "size": f.stat().st_size})
    return clips


@app.post("/api/enroll")
async def api_enroll(req: EnrollRequest):
    if monitor.running:
        raise HTTPException(400, "Stop the monitor before enrolling")
    cfg = load_config()
    thread = threading.Thread(
        target=_do_enrollment,
        args=(req.name, req.duration, cfg),
        daemon=True,
    )
    thread.start()
    return {"status": "enrolling", "name": req.name, "duration": req.duration}


@app.get("/api/enroll/status/{name}")
async def api_enroll_status(name: str):
    if name in _enroll_state:
        return _enroll_state[name]
    return {"status": "unknown"}


@app.get("/api/config")
async def api_config():
    return load_config()


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
