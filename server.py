#!/usr/bin/env python3
"""
FastAPI backend for Voice ID Dashboard.
Target Speaker Extraction (TSE) using X-TF-GridNet.
Records a reference clip, then continuously extracts that speaker's voice
from live microphone audio using a rolling buffer + overlap-add approach.
"""

import asyncio
import base64
import json
import os
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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

# ─── App ─────────────────────────────────────────────────────────────
app = FastAPI(title="Voice ID API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ─── Global State ────────────────────────────────────────────────────
class MonitorState:
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.events: list = []
        self.max_events = 500
        self.active_speakers: dict = {}
        self.ws_clients: list = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def add_event(self, event: dict):
        """Store event in log and broadcast to WS clients."""
        with self.lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
        asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)

    def send_audio(self, event: dict):
        """Broadcast audio to WS clients WITHOUT storing (prevents replay)."""
        asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)

    async def _broadcast(self, event: dict):
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)


monitor = MonitorState()
loop: Optional[asyncio.AbstractEventLoop] = None


# ─── Enrollment State ────────────────────────────────────────────────
class EnrollmentJob:
    def __init__(self):
        self.status = "idle"       # idle | recording | processing | done | error
        self.message = ""
        self.speaker_name = ""
        self.result = {}
        self.lock = threading.Lock()

    def set(self, status, message="", result=None):
        with self.lock:
            self.status = status
            self.message = message
            if result:
                self.result = result

    def get(self):
        with self.lock:
            return {
                "status": self.status,
                "message": self.message,
                "speaker_name": self.speaker_name,
                "result": self.result,
            }


enroll_job = EnrollmentJob()


# ─── Models ──────────────────────────────────────────────────────────
class SpeakerInfo(BaseModel):
    name: str
    display_name: str
    has_reference: bool
    num_samples: int


class EnrollRequest(BaseModel):
    duration: int = 5
    display_name: Optional[str] = None


class MonitorStatus(BaseModel):
    running: bool
    event_count: int


class ConfigUpdate(BaseModel):
    chunk_seconds: Optional[float] = None
    hop_seconds: Optional[float] = None


# ─── Speaker Management API ──────────────────────────────────────────
@app.get("/api/speakers")
def list_speakers():
    config = load_config()
    enrollment_dir = BASE_DIR / config["enrollment"]["dir"]
    speakers = []
    if enrollment_dir.exists():
        for d in sorted(enrollment_dir.iterdir()):
            if d.is_dir():
                has_ref = (d / "reference.wav").exists()
                samples = list(d.glob("sample_*.wav"))
                speakers.append(SpeakerInfo(
                    name=d.name,
                    display_name=d.name.replace("_", " ").title(),
                    has_reference=has_ref,
                    num_samples=len(samples),
                ))
    return {"speakers": [s.dict() for s in speakers]}


@app.delete("/api/speakers/{name}")
def delete_speaker(name: str):
    config = load_config()
    enrollment_dir = BASE_DIR / config["enrollment"]["dir"]
    speaker_dir = enrollment_dir / name
    if not speaker_dir.exists():
        raise HTTPException(404, "Speaker not found")
    import shutil
    shutil.rmtree(speaker_dir)
    return {"status": "deleted", "name": name}


@app.get("/api/speakers/{name}/audio/{sample_id}")
def get_speaker_audio(name: str, sample_id: int):
    config = load_config()
    enrollment_dir = BASE_DIR / config["enrollment"]["dir"]
    wav_path = enrollment_dir / name / f"sample_{sample_id}.wav"
    if not wav_path.exists():
        raise HTTPException(404, "Audio sample not found")
    return FileResponse(str(wav_path), media_type="audio/wav")


@app.post("/api/speakers/{name}/enroll")
def enroll_speaker(name: str, req: EnrollRequest):
    """Start async enrollment: record a reference audio clip for TSE."""
    if enroll_job.status in ("recording", "processing"):
        raise HTTPException(409, detail="Enrollment already in progress")

    enroll_job.speaker_name = name
    enroll_job.result = {}
    enroll_job.set("recording", f"Recording {req.duration}s...")

    thread = threading.Thread(
        target=_do_enrollment,
        args=(name, req.duration),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "message": f"Recording {req.duration}s..."}


@app.get("/api/enroll/status")
def get_enroll_status():
    return enroll_job.get()


def _do_enrollment(name: str, duration: int):
    """Background enrollment: record reference clip, trim silence, save as reference.wav.
    
    X-TF-GridNet uses the raw reference audio directly (no embedding extraction).
    The model's internal AuxEncoder handles speaker representation.
    """
    config = load_config()
    mic_sr = config["audio"]["mic_sample_rate"]
    model_sr = config["audio"]["model_sample_rate"]
    device_index = config["audio"]["device_index"]
    enrollment_dir = BASE_DIR / config["enrollment"]["dir"]

    speaker_dir = enrollment_dir / name.lower().replace(" ", "_")
    speaker_dir.mkdir(parents=True, exist_ok=True)

    existing = list(speaker_dir.glob("sample_*.wav"))
    sample_num = len(existing) + 1

    # Record at mic sample rate
    try:
        print(f"[ENROLL] Recording {duration}s from device {device_index}...", flush=True)
        enroll_job.set("recording", f"Recording {duration}s — speak now!")
        audio = sd.rec(
            int(duration * mic_sr),
            samplerate=mic_sr,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
        audio = audio.squeeze()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        print(f"[ENROLL] Recorded {len(audio)} samples, RMS={rms:.4f}", flush=True)
    except Exception as e:
        print(f"[ENROLL] Recording failed: {e}", flush=True)
        enroll_job.set("error", f"Recording failed: {e}")
        return

    # Trim silence
    enroll_job.set("processing", "Processing reference clip...")

    def _trim_silence(audio_1d, sr, frame_ms=25, threshold=0.01):
        frame_len = int(sr * frame_ms / 1000)
        n_frames = len(audio_1d) // frame_len
        if n_frames == 0:
            return audio_1d
        frame_rms = np.array([
            np.sqrt(np.mean(audio_1d[i*frame_len:(i+1)*frame_len] ** 2))
            for i in range(n_frames)
        ])
        above = np.where(frame_rms > threshold)[0]
        if len(above) == 0:
            return audio_1d
        first = max(0, above[0] - 1)
        last = min(n_frames - 1, above[-1] + 1)
        return audio_1d[first * frame_len : (last + 1) * frame_len]

    trimmed = _trim_silence(audio, mic_sr)
    trim_dur = len(trimmed) / mic_sr
    print(f"[ENROLL] Trimmed silence: {len(audio)/mic_sr:.1f}s -> {trim_dur:.1f}s speech", flush=True)

    # Save full recording as sample
    wav_path = speaker_dir / f"sample_{sample_num}.wav"
    sf.write(str(wav_path), audio, mic_sr)

    # Resample trimmed audio to model sample rate (8kHz) and save as reference.wav
    try:
        import librosa
        trimmed_8k = librosa.resample(trimmed, orig_sr=mic_sr, target_sr=model_sr)
        ref_path = speaker_dir / "reference.wav"
        sf.write(str(ref_path), trimmed_8k, model_sr)
        ref_dur = len(trimmed_8k) / model_sr
        print(f"[ENROLL] Saved reference.wav ({ref_dur:.1f}s at {model_sr}Hz)", flush=True)

        enroll_job.set("done", "Enrollment complete!", result={
            "name": name,
            "sample_num": sample_num,
            "duration": duration,
            "rms": round(rms, 4),
            "ref_duration": round(ref_dur, 1),
        })
    except Exception as e:
        import traceback
        print(f"[ENROLL] Processing FAILED:\n{traceback.format_exc()}", flush=True)
        enroll_job.set("error", f"Processing failed: {e}")


# ─── Config API ──────────────────────────────────────────────────────
@app.get("/api/config")
def get_config():
    return load_config()


@app.patch("/api/config")
def update_config(update: ConfigUpdate):
    config = load_config()
    if update.chunk_seconds is not None:
        config["tse"]["chunk_seconds"] = update.chunk_seconds
    if update.hop_seconds is not None:
        config["tse"]["hop_seconds"] = update.hop_seconds
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config


# ─── Audio Devices API ───────────────────────────────────────────────
@app.get("/api/devices")
def list_audio_devices():
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append({
                "index": i,
                "name": d["name"],
                "channels": d["max_input_channels"],
                "sample_rate": d["default_samplerate"],
            })
    return {"devices": devices}


# ─── Monitor Control API ─────────────────────────────────────────────
@app.get("/api/monitor/status")
def get_monitor_status():
    return MonitorStatus(
        running=monitor.running,
        event_count=len(monitor.events),
    )


@app.get("/api/monitor/events")
def get_monitor_events(limit: int = 100):
    return {"events": monitor.events[-limit:]}


@app.post("/api/monitor/start")
def start_monitor():
    if monitor.running:
        return {"status": "already_running"}

    monitor.stop_event.clear()
    monitor.events.clear()
    monitor.active_speakers.clear()
    monitor.thread = threading.Thread(target=_run_monitor, daemon=True)
    monitor.thread.start()
    monitor.running = True
    return {"status": "started"}


@app.post("/api/monitor/stop")
def stop_monitor():
    if not monitor.running:
        return {"status": "not_running"}
    monitor.stop_event.set()
    monitor.running = False
    return {"status": "stopping"}


def _load_tse_model(config):
    """Load X-TF-GridNet model and return (model, device)."""
    tse = config["tse"]
    compute_device = torch.device(config["device"])

    sys.path.insert(0, str(BASE_DIR / tse["model_dir"] / "nnet"))
    from pTFGridNet import pTFGridNet

    net = pTFGridNet(
        n_fft=tse["n_fft"],
        n_layers=tse["n_layers"],
        lstm_hidden_units=tse["lstm_hidden_units"],
        attn_n_head=tse["attn_n_head"],
        attn_approx_qk_dim=tse["attn_approx_qk_dim"],
        emb_dim=tse["emb_dim"],
        emb_ks=tse["emb_ks"],
        emb_hs=tse["emb_hs"],
        num_spks=tse["num_spks"],
        activation="prelu",
        eps=1e-5,
    )

    cpt_path = str(BASE_DIR / tse["model_dir"] / tse["checkpoint"])
    cpt = torch.load(cpt_path, map_location="cpu")
    net.load_state_dict(cpt["model_state_dict"])
    net.to(compute_device).eval()
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[TSE] Loaded X-TF-GridNet ({n_params:.2f}M params, epoch {cpt.get('epoch', '?')})", flush=True)
    return net, compute_device


def _prepare_stft(waveform_tensor, config, device):
    """Convert raw waveform to STFT representation for X-TF-GridNet.
    Input: waveform_tensor [B, samples] on device
    Output: stft_input [B, 2, T, F] on device, n_frames
    """
    tse = config["tse"]
    sr = config["audio"]["model_sample_rate"]
    win_size = int(sr * tse["win_size"])   # 256
    win_shift = int(sr * tse["win_shift"]) # 64
    n_fft = tse["n_fft"]
    beta = tse["beta"]

    hann = torch.hann_window(win_size).to(device)
    stft = torch.stft(
        waveform_tensor, n_fft=n_fft, hop_length=win_shift,
        win_length=win_size, return_complex=True, window=hann,
    )
    stft = torch.view_as_real(stft)  # [B, F, T, 2]
    mag = torch.norm(stft, dim=-1) ** beta
    phase = torch.atan2(stft[..., -1], stft[..., 0])
    # [B, 2, T, F]
    out = torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1)
    out = out.permute(0, 3, 2, 1)
    return out


def _istft_from_output(est_stft, config, device, length):
    """Convert X-TF-GridNet output [B, 2, T, F] back to waveform."""
    tse = config["tse"]
    sr = config["audio"]["model_sample_rate"]
    win_size = int(sr * tse["win_size"])
    win_shift = int(sr * tse["win_shift"])
    n_fft = tse["n_fft"]
    beta = tse["beta"]

    # [B, 2, T, F] -> [B, F, T, 2]
    est = est_stft.permute(0, 3, 2, 1)
    mag = torch.norm(est, dim=-1) ** (1.0 / beta)
    phase = torch.atan2(est[..., -1], est[..., 0])
    est_complex = torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=-1)
    est_complex = torch.view_as_complex(est_complex)

    hann = torch.hann_window(win_size).to(device)
    wav = torch.istft(
        est_complex, n_fft=n_fft, hop_length=win_shift,
        win_length=win_size, window=hann, length=length,
    )
    return wav  # [B, samples]


def _run_monitor():
    """Target Speaker Extraction pipeline.

    Rolling buffer captures mic audio -> process chunk_seconds windows every
    hop_seconds -> X-TF-GridNet extracts target speaker -> linear crossfade
    stitching -> stream extracted audio to dashboard via WebSocket.
    """
    try:
        config = load_config()
        mic_sr = config["audio"]["mic_sample_rate"]
        model_sr = config["audio"]["model_sample_rate"]
        device_index = config["audio"]["device_index"]
        tse_cfg = config["tse"]
        enrollment_dir = BASE_DIR / config["enrollment"]["dir"]

        chunk_sec = tse_cfg["chunk_seconds"]
        hop_sec = tse_cfg["hop_seconds"]
        chunk_samples = int(chunk_sec * model_sr)
        hop_samples = int(hop_sec * model_sr)
        overlap_samples = chunk_samples - hop_samples

        monitor.add_event({"type": "system", "time": _now(), "message": "Loading X-TF-GridNet..."})

        # Load model
        net, device = _load_tse_model(config)

        # Find enrolled speaker and load reference
        ref_audio = None
        speaker_name = None
        if enrollment_dir.exists():
            for d in sorted(enrollment_dir.iterdir()):
                ref_path = d / "reference.wav"
                if d.is_dir() and ref_path.exists():
                    ref_data, ref_sr = sf.read(str(ref_path))
                    ref_audio = ref_data.astype(np.float32).squeeze()
                    if ref_sr != model_sr:
                        import librosa
                        ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=model_sr)
                    speaker_name = d.name.replace("_", " ").title()
                    print(f"[TSE] Loaded reference for '{speaker_name}' ({len(ref_audio)/model_sr:.1f}s)", flush=True)
                    break

        if ref_audio is None:
            monitor.add_event({"type": "error", "time": _now(),
                               "message": "No enrolled speaker found. Enroll a speaker first."})
            return

        # Prepare reference STFT (computed once, reused every chunk)
        # Use max-abs normalization (matches model training pipeline)
        ref_scale = np.max(np.abs(ref_audio))
        if ref_scale < 1e-8:
            ref_scale = 1.0
        ref_norm = ref_audio / ref_scale
        ref_tensor = torch.tensor(ref_norm, dtype=torch.float32).unsqueeze(0).to(device)
        ref_stft = _prepare_stft(ref_tensor, config, device)
        ref_n_frames = ref_stft.shape[2]
        ref_len_t = torch.tensor([ref_n_frames], dtype=torch.int, device=device)

        monitor.add_event({
            "type": "system", "time": _now(),
            "message": f"Model loaded. Target: {speaker_name}. Listening...",
        })

        # Rolling buffer at model sample rate (8kHz)
        buffer_max = int(tse_cfg["buffer_seconds"] * model_sr)
        audio_buffer = np.zeros(0, dtype=np.float32)

        # Crossfade state: store the tail of the previous chunk for blending
        prev_tail = None  # will hold last overlap_samples of previous output

        chunk_count = 0

        # Mic callback: accumulate audio into buffer
        mic_buffer_lock = threading.Lock()
        ratio = int(mic_sr / model_sr)  # 16000/8000 = 2

        def mic_callback(indata, frames, time_info, status):
            nonlocal audio_buffer
            if status:
                print(f"[MIC] {status}", flush=True)
            mono = indata[:, 0].astype(np.float32)
            # Anti-aliased 2:1 decimation: average adjacent sample pairs
            if ratio > 1:
                # Truncate to even length, then average pairs
                n = (len(mono) // ratio) * ratio
                downsampled = mono[:n].reshape(-1, ratio).mean(axis=1)
            else:
                downsampled = mono
            with mic_buffer_lock:
                audio_buffer = np.concatenate([audio_buffer, downsampled])
                if len(audio_buffer) > buffer_max:
                    audio_buffer = audio_buffer[-buffer_max:]

        # Start mic stream — use small blocksize for low latency
        block_ms = 50  # 50ms blocks
        block_samples_mic = int(block_ms / 1000 * mic_sr)
        stream = sd.InputStream(
            samplerate=mic_sr,
            device=device_index,
            channels=1,
            dtype="float32",
            blocksize=block_samples_mic,
            callback=mic_callback,
        )
        stream.start()
        print(f"[TSE] Mic stream started (block={block_samples_mic}, chunk={chunk_sec}s, hop={hop_sec}s, overlap={overlap_samples/model_sr:.2f}s)", flush=True)

        # Build linear crossfade ramps once
        if overlap_samples > 0:
            fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
            fade_out = 1.0 - fade_in
        else:
            fade_in = fade_out = None

        # Track buffer position: we want to advance by hop_samples each iteration
        last_buf_end = 0  # how many total samples we've consumed from the buffer

        # Main processing loop — poll frequently, process when ready
        while not monitor.stop_event.is_set():
            time.sleep(0.05)  # 50ms poll interval — low latency

            with mic_buffer_lock:
                buf_len = len(audio_buffer)

            if buf_len < chunk_samples:
                continue

            # Only process if hop_samples of new audio have arrived since last chunk
            if chunk_count > 0 and buf_len < last_buf_end + hop_samples:
                continue

            # Grab the latest chunk_samples from buffer
            with mic_buffer_lock:
                chunk = audio_buffer[-chunk_samples:].copy()
            last_buf_end = buf_len

            # Max-abs normalization (matches model training pipeline)
            chunk_scale = np.max(np.abs(chunk))
            if chunk_scale < 1e-6:
                # Silence — skip
                continue
            chunk_norm = chunk / chunk_scale

            # Run TSE inference
            t0 = time.perf_counter()
            mix_tensor = torch.tensor(chunk_norm, dtype=torch.float32).unsqueeze(0).to(device)
            mix_stft = _prepare_stft(mix_tensor, config, device)

            with torch.no_grad():
                est_stft, _ = net(mix_stft, ref_stft, ref_len_t)

            est_wav = _istft_from_output(est_stft, config, device, length=chunk_samples)
            est_np = est_wav.squeeze().cpu().numpy() * chunk_scale  # denormalize
            infer_ms = (time.perf_counter() - t0) * 1000

            chunk_count += 1

            # ── Linear crossfade stitching ──
            # Split output into: overlap region (start) + new region (rest)
            if prev_tail is not None and overlap_samples > 0:
                # Crossfade: blend prev_tail with start of current output
                blended = prev_tail * fade_out + est_np[:overlap_samples] * fade_in
                # New audio = crossfaded overlap + non-overlapping new portion
                new_audio = np.concatenate([blended, est_np[overlap_samples:]])
            else:
                # First chunk — send everything
                new_audio = est_np

            # Store tail for next crossfade
            if overlap_samples > 0:
                prev_tail = est_np[-overlap_samples:].copy()
            else:
                prev_tail = None

            # Clip to prevent distortion
            peak = np.max(np.abs(new_audio))
            if peak > 1.0:
                new_audio = new_audio / peak

            # Encode as 16-bit PCM -> base64 and send via WebSocket (NOT stored)
            pcm_16 = (new_audio * 32767).astype(np.int16)
            audio_b64 = base64.b64encode(pcm_16.tobytes()).decode("ascii")

            monitor.send_audio({
                "type": "audio",
                "time": _now(),
                "data": audio_b64,
                "sample_rate": model_sr,
                "samples": len(pcm_16),
            })

            # Log every 5 chunks (not audio data, just status)
            if chunk_count % 5 == 1:
                monitor.add_event({
                    "type": "system", "time": _now(),
                    "message": f"Chunk {chunk_count}: {infer_ms:.0f}ms inference, {len(new_audio)/model_sr:.2f}s audio",
                })

        stream.stop()
        stream.close()

    except Exception as e:
        import traceback
        print(f"[MONITOR] Crashed:\n{traceback.format_exc()}", flush=True)
        monitor.add_event({"type": "error", "time": _now(), "message": f"Monitor crashed: {e}"})
    finally:
        monitor.running = False
        monitor.add_event({"type": "system", "time": _now(), "message": "Monitor stopped."})


def _now():
    return datetime.now().strftime("%H:%M:%S")


# ─── WebSocket ───────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    monitor.ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in monitor.ws_clients:
            monitor.ws_clients.remove(ws)


# ─── Dashboard ───────────────────────────────────────────────────────
@app.get("/")
def serve_dashboard():
    html_path = BASE_DIR / "dashboard.html"
    return HTMLResponse(html_path.read_text())


# ─── Startup ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global loop
    loop = asyncio.get_event_loop()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8042)
