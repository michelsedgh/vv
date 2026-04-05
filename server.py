#!/usr/bin/env python3
"""
MeanFlow-TSE live mic → WebSocket (for STT or headphones).
Enrollment: per-speaker reference.wav at 16 kHz mono (first enroll_seconds used at inference).

Streaming architecture (overlap-add):
  The model always processes full 3s windows (matching training segment).
  A sliding window advances by hop_seconds (default 1s) each step.
  Steady-state latency ≈ hop_seconds + GPU inference time.
"""
from __future__ import annotations

import asyncio
import base64
import threading
import time
import uuid
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

BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


app = FastAPI(title="MeanFlow-TSE Voice")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loop: Optional[asyncio.AbstractEventLoop] = None


class MonitorState:
    def __init__(self):
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.events: list = []
        self.max_events = 300
        self.ws_clients: list = []
        self.ws_lock = threading.Lock()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        # Set on /api/monitor/start: enrolled folder name (slug) or None → use config / newest file
        self.start_speaker: Optional[str] = None

    def add_event(self, event: dict):
        with self.lock:
            self.events.append(event)
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events :]
        if loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)

    def send_audio(self, event: dict):
        if loop is not None:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)

    async def _broadcast(self, event: dict):
        with self.ws_lock:
            clients = list(self.ws_clients)
        if event.get("type") == "audio":
            st = event.get("step")
            if st is not None and st % 20 == 0:
                print(
                    f"[ws] audio step={st} -> {len(clients)} WebSocket client(s) "
                    f"(if >1, same chunk plays in every tab/device)",
                    flush=True,
                )
        dead = []
        for client in clients:
            try:
                await client.send_json(event)
            except Exception:
                dead.append(client)
        if not dead:
            return
        with self.ws_lock:
            for d in dead:
                try:
                    self.ws_clients.remove(d)
                except ValueError:
                    pass


monitor = MonitorState()
# Prevents two concurrent POST /monitor/start from spawning two GPU/mic threads (would duplicate audio + load).
monitor_start_stop_lock = threading.Lock()


class EnrollJob:
    def __init__(self):
        self.status = "idle"
        self.message = ""
        self.speaker_name = ""
        self.result: dict = {}
        self.lock = threading.Lock()

    def set(self, status, message="", result=None):
        with self.lock:
            self.status = status
            self.message = message
            if result is not None:
                self.result = result

    def get(self):
        with self.lock:
            return {
                "status": self.status,
                "message": self.message,
                "speaker_name": self.speaker_name,
                "result": self.result,
            }


enroll_job = EnrollJob()


def _now():
    return datetime.now().strftime("%H:%M:%S")


# ─── API models ─────────────────────────────────────────────────────
class EnrollRequest(BaseModel):
    # Seconds of mic capture; default from enrollment.record_seconds (≥ meanflow.enroll_seconds)
    duration: Optional[int] = None


class SpeakerInfo(BaseModel):
    name: str
    display_name: str
    has_reference: bool


# ─── Speakers ───────────────────────────────────────────────────────
@app.get("/api/speakers")
def list_speakers():
    config = load_config()
    root = BASE_DIR / config["enrollment"]["dir"]
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if d.is_dir():
                out.append(
                    SpeakerInfo(
                        name=d.name,
                        display_name=d.name.replace("_", " ").title(),
                        has_reference=(d / "reference.wav").exists(),
                    )
                )
    return {"speakers": [s.dict() for s in out]}


@app.delete("/api/speakers/{name}")
def delete_speaker(name: str):
    config = load_config()
    p = BASE_DIR / config["enrollment"]["dir"] / name
    if not p.exists():
        raise HTTPException(404, "Not found")
    import shutil

    shutil.rmtree(p)
    return {"status": "deleted"}


@app.get("/api/config")
def get_config():
    return load_config()


@app.get("/api/devices")
def list_devices():
    devs = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devs.append(
                {
                    "index": i,
                    "name": d["name"],
                    "channels": d["max_input_channels"],
                    "sample_rate": d["default_samplerate"],
                }
            )
    return {"devices": devs}


# ─── Enrollment (reference.wav for MeanFlow) ─────────────────────────
@app.post("/api/speakers/{name}/enroll")
def enroll_start(name: str, req: EnrollRequest):
    if enroll_job.status in ("recording", "processing"):
        raise HTTPException(409, "Enrollment in progress")
    cfg = load_config()
    rec_default = int(cfg["enrollment"].get("record_seconds") or cfg["meanflow"]["enroll_seconds"])
    need = int(cfg["meanflow"]["enroll_seconds"])
    duration = req.duration if req.duration is not None else rec_default
    if duration < need:
        raise HTTPException(
            400,
            f"duration must be >= meanflow.enroll_seconds ({need}s); got {duration}s",
        )
    enroll_job.speaker_name = name
    enroll_job.result = {}
    enroll_job.set("recording", f"Recording {duration}s @ {cfg['audio']['sample_rate']} Hz...")
    threading.Thread(
        target=_do_enroll,
        args=(name, duration),
        daemon=True,
    ).start()
    return {"status": "started"}


@app.get("/api/enroll/status")
def enroll_status():
    return enroll_job.get()


def _do_enroll(name: str, duration: int):
    config = load_config()
    sr = int(config["audio"]["sample_rate"])
    dev = int(config["audio"]["device_index"])
    root = BASE_DIR / config["enrollment"]["dir"]
    safe = name.lower().replace(" ", "_")
    speaker_dir = root / safe
    speaker_dir.mkdir(parents=True, exist_ok=True)

    try:
        enroll_job.set("recording", "Speak now...")
        audio = sd.rec(
            int(duration * sr),
            samplerate=sr,
            channels=1,
            dtype="float32",
            device=dev,
        )
        sd.wait()
        mono = audio.squeeze()
        peak = float(np.max(np.abs(mono)))
        ref_path = speaker_dir / "reference.wav"
        sf.write(str(ref_path), mono, sr)
        enroll_job.set(
            "done",
            "Saved reference.wav",
            {"name": name, "peak": round(peak, 4), "path": str(ref_path)},
        )
    except Exception as e:
        enroll_job.set("error", str(e))


# ─── Monitor ──────────────────────────────────────────────────────────
@app.get("/api/monitor/status")
def monitor_status():
    return {"running": monitor.running, "events": len(monitor.events)}


@app.get("/api/monitor/events")
def monitor_events(limit: int = 80):
    return {"events": monitor.events[-limit:]}


@app.post("/api/monitor/start")
def monitor_start(speaker: Optional[str] = None):
    """
    speaker: optional enrolled folder name (same as /api/speakers). If omitted, uses
    enrollment.active_speaker from config, else the newest reference.wav under enrolled_speakers/.
    """
    with monitor_start_stop_lock:
        if monitor.running:
            return {"status": "already_running"}
        monitor.stop_event.clear()
        monitor.events.clear()
        monitor.start_speaker = (
            speaker.strip().lower().replace(" ", "_") if speaker and speaker.strip() else None
        )
        monitor.running = True
        try:
            t = threading.Thread(
                target=_run_meanflow_monitor,
                daemon=True,
                name="meanflow-monitor",
            )
            monitor.thread = t
            t.start()
        except Exception:
            monitor.running = False
            raise
    return {"status": "started"}


@app.post("/api/monitor/stop")
def monitor_stop():
    with monitor_start_stop_lock:
        if not monitor.running:
            return {"status": "not_running"}
        monitor.stop_event.set()
        monitor.running = False
    return {"status": "stopping"}


def _normalize_speaker_slug(name: Optional[str]) -> Optional[str]:
    if not name or not str(name).strip():
        return None
    return str(name).strip().lower().replace(" ", "_")


def _resolve_enrollment_reference(
    enroll_root: Path,
    sr: int,
    *,
    start_speaker: Optional[str],
    config_speaker: Optional[str],
) -> tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    Pick reference.wav per MeanFlow/LibriMix: mono float32 @ sr, first enroll_seconds used later.
    Priority: API start_speaker → config active_speaker → newest reference.wav by mtime.
    Returns (audio, display_name, dir_name) or (None, None, None).
    """
    from mftse_infer import to_mono_float32

    if not enroll_root.is_dir():
        return None, None, None
    rows: list[tuple[str, Path, float]] = []
    for d in enroll_root.iterdir():
        if not d.is_dir():
            continue
        ref = d / "reference.wav"
        if ref.is_file():
            rows.append((d.name, ref, ref.stat().st_mtime))
    if not rows:
        return None, None, None

    want = _normalize_speaker_slug(start_speaker) or _normalize_speaker_slug(config_speaker)
    if want:
        for dir_name, ref_path, _ in rows:
            if dir_name == want:
                break
        else:
            return None, None, None
    else:
        dir_name, ref_path, _ = max(rows, key=lambda r: r[2])

    wav, rsr = sf.read(str(ref_path), dtype="float32")
    ref_audio = to_mono_float32(wav)
    if rsr != sr:
        import librosa

        ref_audio = librosa.resample(
            ref_audio, orig_sr=rsr, target_sr=sr
        ).astype(np.float32)
    display = dir_name.replace("_", " ").title()
    return ref_audio, display, dir_name


def _run_meanflow_monitor():
    try:
        from mftse_infer import MeanFlowTSERunner, fix_length

        config = load_config()
        audio_cfg = config["audio"]
        mf = config["meanflow"]
        enroll_cfg = config["enrollment"]
        sr = int(audio_cfg["sample_rate"])
        dev_idx = int(audio_cfg["device_index"])
        max_chunks = max(2, int(mf.get("max_queued_chunks", 4)))
        min_infer_gap = float(mf.get("min_inference_interval_seconds", 0.0))
        enroll_sec = float(mf.get("enroll_seconds", 3))
        euler_steps = int(mf.get("euler_steps", 1))

        # ── Sliding-window streaming parameters ──────────────────────
        # Window size = dataset.segment from base config (always 3s for this model).
        base_cfg = BASE_DIR / mf["base_config"]
        ckpt = BASE_DIR / mf["checkpoint"]
        use_tp = bool(mf.get("use_t_predictor", True))
        tp_ckpt = (BASE_DIR / mf["t_predictor_checkpoint"]) if use_tp else None

        try:
            seg_ds = float(yaml.safe_load(base_cfg.read_text())["dataset"]["segment"])
        except Exception:
            seg_ds = 3.0
        window_sec = seg_ds
        window_samples = int(round(window_sec * sr))
        # e.g. window_sec=3.0, sr=16000 → window_samples=48000

        hop_sec = float(mf.get("hop_seconds", 1.0))
        # Clamp: hop can't be larger than window (no overlap) or tiny (<50 ms)
        hop_sec = max(0.05, min(hop_sec, window_sec))
        hop_samples = int(round(hop_sec * sr))
        # e.g. hop_sec=1.0 → hop_samples=16000

        crossfade_sec = float(mf.get("crossfade_seconds", 0.01))
        crossfade_samples = int(round(crossfade_sec * sr))
        # e.g. crossfade_sec=0.01 → crossfade_samples=160
        # Clamp: crossfade can't exceed half the hop (would eat into output)
        crossfade_samples = min(crossfade_samples, hop_samples // 2)

        enroll_samples = int(enroll_sec * sr)
        # e.g. enroll_sec=3 → enroll_samples=48000

        # Max buffered audio before we start dropping to stay near real-time
        # First step needs window_samples, subsequent steps need hop_samples each.
        cap_samples = window_samples + max_chunks * hop_samples
        # e.g. 48000 + 4*16000 = 112000 = 7s

        use_fp16 = bool(mf.get("use_fp16", False))
        use_compile = bool(mf.get("use_torch_compile", False))

        is_ola = hop_sec < window_sec  # True = overlap-add mode
        mode_label = (
            f"overlap-add (window={window_sec}s, hop={hop_sec}s, "
            f"overlap={window_sec - hop_sec:.1f}s, crossfade={crossfade_sec*1000:.0f}ms)"
            if is_ola
            else f"non-overlapping {window_sec}s chunks"
        )
        monitor.add_event(
            {
                "type": "system",
                "time": _now(),
                "message": f"Streaming mode: {mode_label}",
            }
        )

        monitor.add_event(
            {"type": "system", "time": _now(), "message": "Loading MeanFlow-TSE..."}
        )

        runner = MeanFlowTSERunner(
            base_config_path=base_cfg,
            checkpoint_path=ckpt,
            device=config.get("device"),
            use_t_predictor=use_tp,
            t_predictor_checkpoint=tp_ckpt if use_tp else None,
        )

        # ── Resolve enrollment reference ─────────────────────────────
        enroll_root = BASE_DIR / enroll_cfg["dir"]
        cfg_spk = enroll_cfg.get("active_speaker") or None
        ref_audio, speaker_name, _ = _resolve_enrollment_reference(
            enroll_root,
            sr,
            start_speaker=monitor.start_speaker,
            config_speaker=cfg_spk,
        )

        if ref_audio is None and enroll_root.is_dir():
            want = monitor.start_speaker or _normalize_speaker_slug(
                cfg_spk if cfg_spk else None
            )
            if want:
                monitor.add_event(
                    {
                        "type": "error",
                        "time": _now(),
                        "message": f"No reference for speaker '{want}' (folder under {enroll_cfg['dir']}/).",
                    }
                )
                return

        if ref_audio is None:
            monitor.add_event(
                {
                    "type": "error",
                    "time": _now(),
                    "message": "No reference.wav — enroll a speaker first.",
                }
            )
            return

        ref_fixed = fix_length(ref_audio, enroll_samples)

        # ── Apply optimizations ──────────────────────────────────────
        if use_fp16:
            runner.enable_amp(torch.float16)
        if use_compile:
            runner.compile_model()

        # Cache enrollment STFT (same reference every chunk — no reason to recompute)
        runner.cache_enrollment(ref_fixed)

        alpha_mode = (
            f"ECAPA t-predictor ({mf.get('t_predictor_checkpoint', '')})"
            if use_tp
            else "alpha=0.5 (HALF)"
        )
        opt_flags = []
        if use_fp16:
            opt_flags.append("FP16")
        if use_compile:
            opt_flags.append("torch.compile")
        opt_label = ", ".join(opt_flags) if opt_flags else "none"

        monitor.add_event(
            {
                "type": "system",
                "time": _now(),
                "message": (
                    f"MeanFlow-TSE on {speaker_name}; {sr} Hz; ref {enroll_sec}s; "
                    f"hop {hop_sec}s; Euler steps={euler_steps}; {alpha_mode}; "
                    f"optimizations: {opt_label}."
                ),
            }
        )

        # ── Warmup (triggers torch.compile JIT if enabled) ───────────
        if use_compile:
            monitor.add_event(
                {"type": "system", "time": _now(), "message": "Running warmup inference (torch.compile JIT)..."}
            )
        runner.warmup(enroll_samples, num_steps=euler_steps)
        monitor.add_event(
            {"type": "system", "time": _now(), "message": "Warmup done. Starting mic capture..."}
        )

        # ── Mic capture → audio_buffer ───────────────────────────────
        audio_buffer = np.zeros(0, dtype=np.float32)
        buf_lock = threading.Lock()
        block = max(1024, int(0.05 * sr))
        # e.g. sr=16000 → block=max(1024, 800)=1024

        def mic_cb(indata, frames, tinfo, status):
            nonlocal audio_buffer
            if status:
                print(f"[MIC] {status}", flush=True)
            mono = indata[:, 0].astype(np.float32)
            with buf_lock:
                audio_buffer = np.concatenate([audio_buffer, mono])
                # Drop oldest audio in hop-sized steps if backlog is too large.
                dropped = 0
                while len(audio_buffer) > cap_samples:
                    audio_buffer = audio_buffer[hop_samples:].copy()
                    dropped += 1
                if dropped:
                    print(f"[MeanFlow] dropped {dropped} hop(s) (catch-up)", flush=True)

        stream = sd.InputStream(
            samplerate=sr,
            device=dev_idx,
            channels=1,
            dtype="float32",
            blocksize=block,
            callback=mic_cb,
        )
        stream.start()

        # ── Main inference loop: sliding-window overlap-add ──────────
        #
        # DIMENSION TRACE (sr=16000, window=3s, hop=1s, crossfade=10ms):
        #   window_samples = 48000
        #   hop_samples    = 16000
        #   crossfade_samples = 160
        #
        # Step 0 (initial fill):
        #   consume window_samples=48000 from audio_buffer
        #   window[:] = audio[0 : 48000]                   (3.0 s)
        #   model(window) → output[48000]                   (3.0 s)
        #   send: output[0 : 48000-160] = 47840 samples    (2.99 s)
        #   hold: prev_tail = output[47840 : 48000]         (160 samples = 10 ms)
        #
        # Step 1+:
        #   consume hop_samples=16000 from audio_buffer     (1.0 s of new audio)
        #   window = window[16000:] ++ new_audio            (slide forward 1s)
        #     = audio[16000 : 64000]                        (still 48000 = 3.0 s)
        #   model(window) → output[48000]
        #   extract = output[-(16000+160):] = output[31840 : 48000]  (16160 samples)
        #     extract[:160]  = model's view of boundary region (overlaps with prev window)
        #     extract[160:]  = model's view of new hop region
        #   crossfade: blend = fade_out * prev_tail + fade_in * extract[:160]
        #   send: blend(160) + extract[160 : 16000] = 16000 samples  (1.0 s)
        #   hold: prev_tail = extract[16000 : 16160]        (160 samples = 10 ms)
        #
        # Latency (steady state) = hop_sec + inference_time ≈ 1.0 + 0.1-0.3 s

        window = None  # Will be filled on step 0
        prev_tail = None  # Held-back samples for crossfade
        last_infer_end = 0.0
        step = 0

        # Pre-compute crossfade ramps (constant, reused every step)
        if crossfade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, crossfade_samples, dtype=np.float32)
            fade_out = 1.0 - fade_in
        else:
            fade_in = fade_out = None

        while not monitor.stop_event.is_set():
            time.sleep(0.02)
            now = time.monotonic()
            if min_infer_gap > 0 and (now - last_infer_end) < min_infer_gap:
                continue

            # ── Determine how many new samples we need ───────────────
            if step == 0:
                needed = window_samples  # First step: fill entire 3s window
            else:
                needed = hop_samples  # Subsequent: just the 1s hop

            with buf_lock:
                if len(audio_buffer) < needed:
                    continue
                new_audio = audio_buffer[:needed].copy()
                audio_buffer = audio_buffer[needed:].copy()

            # ── Build / slide the window ─────────────────────────────
            if step == 0:
                window = new_audio  # shape: (window_samples,) = (48000,)
            else:
                # Slide: drop oldest hop, append new audio
                # window[hop_samples:] has (window_samples - hop_samples) samples = 32000
                # new_audio has hop_samples = 16000
                # total = 48000 ✓
                window = np.concatenate([window[hop_samples:], new_audio])

            assert len(window) == window_samples, (
                f"Window size mismatch: {len(window)} != {window_samples}"
            )

            # ── Run model ────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                output = runner.infer_chunk(window, ref_fixed, num_steps=euler_steps)
            except Exception as e:
                # On failure, put audio back so we don't lose it
                with buf_lock:
                    audio_buffer = np.concatenate([new_audio, audio_buffer])
                monitor.add_event(
                    {
                        "type": "error",
                        "time": _now(),
                        "message": f"Inference failed: {e}",
                    }
                )
                print(f"[MeanFlow] {e}", flush=True)
                last_infer_end = time.monotonic()
                continue
            dt_ms = (time.perf_counter() - t0) * 1000
            last_infer_end = time.monotonic()
            step += 1

            # ── Extract output with crossfade ────────────────────────
            # Ensure output is exactly window_samples
            output = np.asarray(output, dtype=np.float32).reshape(-1)
            if output.size != window_samples:
                if output.size > window_samples:
                    output = output[:window_samples]
                else:
                    tmp = np.zeros(window_samples, dtype=np.float32)
                    tmp[: output.size] = output
                    output = tmp

            if step == 1:
                # ── FIRST OUTPUT: send most of the full window ───────
                if crossfade_samples > 0:
                    # Hold back last crossfade_samples for blending with next chunk
                    out_play = output[: window_samples - crossfade_samples]
                    prev_tail = output[window_samples - crossfade_samples :].copy()
                    # out_play.size = 48000 - 160 = 47840
                    # prev_tail.size = 160
                else:
                    out_play = output
            else:
                # ── SUBSEQUENT: extract last (hop + crossfade) from output
                if crossfade_samples > 0:
                    extract_len = hop_samples + crossfade_samples
                    # e.g. 16000 + 160 = 16160
                    extract = output[-extract_len:]
                    # extract[:160]  → boundary region (same time as prev_tail)
                    # extract[160:]  → new hop region (16000 samples)

                    if prev_tail is not None:
                        # Crossfade: blend prev_tail with start of extract
                        blend = fade_out * prev_tail + fade_in * extract[:crossfade_samples]
                        # blend.size = 160
                    else:
                        blend = extract[:crossfade_samples]

                    # Output = blend + middle portion (exclude new tail)
                    # middle = extract[crossfade_samples : -crossfade_samples]
                    #        = extract[160 : 16000]     (size = 15840)
                    # total  = 160 + 15840 = 16000 = hop_samples ✓
                    middle = extract[crossfade_samples : extract_len - crossfade_samples]
                    out_play = np.concatenate([blend, middle])

                    # Hold back new tail for next crossfade
                    prev_tail = extract[-crossfade_samples:].copy()
                    # prev_tail.size = 160
                else:
                    out_play = output[-hop_samples:]

            # ── Validate + encode ────────────────────────────────────
            peak = float(np.max(np.abs(out_play))) if out_play.size else 0.0
            if peak < 1e-5:
                print(
                    f"[MeanFlow] step {step}: output peak={peak:.2e} (inaudible? check mic / enrollment / GPU)",
                    flush=True,
                )
            if peak > 1.0:
                out_play = (out_play / peak).astype(np.float32)
            pcm = (np.clip(out_play, -1.0, 1.0) * 32767).astype(np.int16)
            b64 = base64.b64encode(pcm.tobytes()).decode("ascii")

            # ── Send to WebSocket ────────────────────────────────────
            monitor.send_audio(
                {
                    "type": "audio",
                    "time": _now(),
                    "chunk_id": str(uuid.uuid4()),
                    "data": b64,
                    "sample_rate": sr,
                    "samples": len(pcm),
                    "step": step,
                    "infer_ms": round(dt_ms, 0),
                }
            )
            dur_label = f"{len(pcm)/sr:.2f}s"
            if step == 1:
                monitor.add_event(
                    {
                        "type": "system",
                        "time": _now(),
                        "message": f"Step {step} (initial): {dt_ms:.0f} ms → {dur_label} output",
                    }
                )
            else:
                monitor.add_event(
                    {
                        "type": "system",
                        "time": _now(),
                        "message": f"Step {step}: {dt_ms:.0f} ms → {dur_label} output (hop={hop_sec}s)",
                    }
                )

            if runner.device.type == "cuda" and step % 8 == 0:
                torch.cuda.empty_cache()

        stream.stop()
        stream.close()
        runner.clear_enrollment_cache()
    except FileNotFoundError as e:
        monitor.add_event({"type": "error", "time": _now(), "message": str(e)})
    except Exception as e:
        import traceback

        print(traceback.format_exc(), flush=True)
        monitor.add_event({"type": "error", "time": _now(), "message": str(e)})
    finally:
        monitor.running = False
        monitor.add_event({"type": "system", "time": _now(), "message": "Monitor stopped."})


# When a second browser tab/device connects, the previous WebSocket is closed. Otherwise every
# open tab receives the same PCM and you hear the same chunk repeated N times (not fixable in the queue alone).
_WS_SUPERSEDED_CODE = 4000


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    with monitor.ws_lock:
        previous = list(monitor.ws_clients)
        monitor.ws_clients.clear()
    for other in previous:
        try:
            await other.close(code=_WS_SUPERSEDED_CODE)
        except Exception:
            pass
    with monitor.ws_lock:
        monitor.ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        with monitor.ws_lock:
            try:
                monitor.ws_clients.remove(ws)
            except ValueError:
                pass


@app.get("/")
def dashboard():
    p = BASE_DIR / "dashboard.html"
    if not p.exists():
        return HTMLResponse("<p>dashboard.html missing</p>", status_code=404)
    return HTMLResponse(p.read_text())


@app.on_event("startup")
async def startup():
    global loop
    loop = asyncio.get_event_loop()


def main():
    cfg = load_config()
    h = cfg.get("server", {}).get("host", "0.0.0.0")
    p = int(cfg.get("server", {}).get("port", 8042))
    uvicorn.run(app, host=h, port=p)


if __name__ == "__main__":
    main()
