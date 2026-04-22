from __future__ import annotations

import collections
import copy
import ctypes
import gc
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from . import settings
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

try:
    import torchvision.transforms.functional_tensor
except ImportError:
    sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional


def to_normalized_float_tensor(vid):
    return vid


CS_CLASSES = [
    "Cook.Cleandishes",
    "Cook.Cleanup",
    "Cook.Cut",
    "Cook.Stir",
    "Cook.Usestove",
    "Cutbread",
    "Drink.Frombottle",
    "Drink.Fromcan",
    "Drink.Fromcup",
    "Drink.Fromglass",
    "Eat.Attable",
    "Eat.Snack",
    "Enter",
    "Getup",
    "Laydown",
    "Leave",
    "Makecoffee.Pourgrains",
    "Makecoffee.Pourwater",
    "Maketea.Boilwater",
    "Maketea.Insertteabag",
    "Pour.Frombottle",
    "Pour.Fromcan",
    "Pour.Fromkettle",
    "Readbook",
    "Sitdown",
    "Takepills",
    "Uselaptop",
    "Usetablet",
    "Usetelephone",
    "Walk",
    "WatchTV",
]

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

N_FRAMES = 16
SCALE = 256
CROP = 224
BUFFER_W = 640
BUFFER_H = 360
CONTEXT_SECONDS = 4.26


def pretty_action(label: str) -> str:
    return label.replace(".", " ")


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def scale_short_side(frames: torch.Tensor, target: int) -> torch.Tensor:
    _, _, height, width = frames.shape
    if width <= height:
        if width == target:
            return frames
        new_width = target
        new_height = int(math.floor(float(height) / width * target))
    else:
        if height == target:
            return frames
        new_height = target
        new_width = int(math.floor(float(width) / height * target))
    return F.interpolate(
        frames,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )


def centre_crop(frames: torch.Tensor, size: int) -> torch.Tensor:
    _, _, height, width = frames.shape
    y_off = int(math.ceil((height - size) / 2))
    x_off = int(math.ceil((width - size) / 2))
    return frames[:, :, y_off : y_off + size, x_off : x_off + size]


def preprocess_clip(frames_bgr: list) -> Optional[torch.Tensor]:
    if not frames_bgr:
        return None
    indices = np.linspace(0, len(frames_bgr) - 1, N_FRAMES, dtype=int)
    frames_rgb = []
    for idx in indices:
        bgr = frames_bgr[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).float() / 255.0
        frames_rgb.append(tensor)
    clip = torch.stack(frames_rgb).permute(0, 3, 1, 2)
    clip = (clip - MEAN.view(1, 3, 1, 1)) / STD.view(1, 3, 1, 1)
    clip = scale_short_side(clip, SCALE)
    clip = centre_crop(clip, CROP)
    return clip.unsqueeze(0)


def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_hp = dict(ckpt.get("hyper_parameters", {}))
    dm_hp = dict(ckpt.get("datamodule_hyper_parameters", {}))
    hparams = {**dm_hp, **model_hp}

    project_root = str(settings.POGUISE_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from models.poguise import POGUISE

    hparams["pretrained"] = None
    hparams["mode"] = "test"

    clean_hparams = {}
    for key, value in hparams.items():
        if hasattr(value, "__dict__"):
            value = vars(value)
        if isinstance(value, (int, float, str, bool, list, tuple, type(None))):
            clean_hparams[key] = value
        elif isinstance(value, dict) and type(value) is dict:
            clean_hparams[key] = value

    model = POGUISE(**clean_hparams)
    raw_sd = ckpt["state_dict"]
    poguise_sd = {
        key[6:]: value
        for key, value in raw_sd.items()
        if key.startswith("model.")
    }
    model.load_state_dict(poguise_sd, strict=True)
    model.to(device)
    model.eval()
    del poguise_sd
    del raw_sd
    del clean_hparams
    del hparams
    del dm_hp
    del model_hp
    del ckpt
    gc.collect()
    trim_process_heap()
    return model


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    clip: torch.Tensor,
    device: torch.device,
    use_fp16: bool,
    return_heatmap: bool,
):
    clip = clip.to(device)
    if use_fp16:
        clip = clip.half()
    out = model(clip)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    probs = torch.softmax(logits.float(), dim=-1)
    hm_tensor = None
    if return_heatmap and isinstance(out, (tuple, list)) and len(out) > 1 and out[1] is not None:
        hm_tensor = out[1].detach().float().cpu().numpy()
    return probs.squeeze(0).cpu().numpy(), hm_tensor


class PoguiseService:
    def __init__(self, gpu_lock: threading.Lock, policy, voice_running_getter):
        self.gpu_lock = gpu_lock
        self.policy = policy
        self.voice_running_getter = voice_running_getter
        self.loop = None
        self.event_bus: Optional[EventBus] = None
        self._lock = threading.Lock()
        self._show_heatmap = False
        # Lazy-JPEG state.  The old design composited (heatmap blend) and
        # JPEG-encoded every raw frame at 30 fps whether or not anyone was
        # watching — ~25-30% of one core.  Now we just stash a shallow ref
        # to the latest raw frame + ts and defer the blend + imencode to
        # ``latest_frame_bytes()``, which is called by the MJPEG endpoint
        # and the 1-Hz nexus-agent thumbnail loop.  Concurrent callers
        # share one encode per frame via the (_latest_raw_ts == _latest_jpeg_ts)
        # cache check.
        self._latest_raw_frame = None
        self._latest_raw_ts: float = 0.0
        self._latest_frame_jpeg = b""
        self._latest_frame_jpeg_ts: float = -1.0
        self._last_heatmap = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self._stats = self._fresh_stats()
        self._model = None
        self._capture = None
        self._cleanup_lock = threading.Lock()
        # Frame-sink fan-out: external consumers (the Nexus vision service) can
        # register a callback that receives every raw BGR frame + capture ts
        # without opening a second cv2.VideoCapture on /dev/video0. Sinks are
        # called synchronously on the camera thread, so they MUST be cheap —
        # stash the frame in a latest-wins slot and do the heavy work elsewhere.
        self._frame_sinks: list = []
        self._frame_sinks_lock = threading.Lock()

    def register_frame_sink(self, sink) -> None:
        """Subscribe to every raw BGR frame read from the camera.

        ``sink(frame_bgr: np.ndarray, capture_ts: float) -> None``. Must be
        cheap (microseconds). Called on the camera thread.
        """
        with self._frame_sinks_lock:
            if sink not in self._frame_sinks:
                self._frame_sinks.append(sink)

    def unregister_frame_sink(self, sink) -> None:
        with self._frame_sinks_lock:
            try:
                self._frame_sinks.remove(sink)
            except ValueError:
                pass

    def set_loop(self, loop) -> None:
        self.loop = loop

    def set_event_bus(self, bus: EventBus) -> None:
        self.event_bus = bus

    def _publish_brain_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.event_bus is None:
            return
        self.event_bus.publish_sync(Event(type=event_type, data=data), loop=self.loop)

    def _fresh_stats(self) -> dict:
        return {
            "running": False,
            "fps_cam": 0.0,
            "inference_ms": 0.0,
            "buffer_pct": 0.0,
            "current_action": "Waiting...",
            "action_history": [],
            "predictions": [],
            "policy_mode": "idle",
            "policy_reason": "PO-GUISE is stopped",
            "effective_infer_every": None,
            "skipped_due_to_busy_gpu": 0,
            "skipped_due_to_policy": 0,
            "heatmap_enabled": self._show_heatmap,
            "camera_open": False,
            "checkpoint": settings.load_section_config("poguise")["checkpoint"],
            "phase": "idle",
            "status_message": "PO-GUISE is stopped",
            "error": None,
        }

    def config_payload(self) -> dict:
        return settings.editor_payload("poguise")

    def update_config(self, values: Dict[str, Any]) -> dict:
        if self.running:
            raise ValueError("Stop PO-GUISE before changing settings")
        return settings.apply_editor_update("poguise", values)

    def reset_config(self) -> dict:
        if self.running:
            raise ValueError("Stop PO-GUISE before resetting settings")
        return settings.reset_editor_section("poguise")

    def status(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._stats)

    def toggle_heatmap(self) -> dict:
        with self._lock:
            self._show_heatmap = not self._show_heatmap
            self._stats["heatmap_enabled"] = self._show_heatmap
            return {"show_heatmap": self._show_heatmap}

    def video_feed(self):
        # ~30 fps ceiling.  The MJPEG pull path now composites + encodes
        # on demand via latest_frame_bytes(), so we need to pace ourselves
        # instead of spinning on the lock.
        last_ts = -1.0
        while True:
            with self._lock:
                active = self.running
                raw_ts = self._latest_raw_ts
            if active and raw_ts > last_ts:
                frame_bytes = self.latest_frame_bytes()
                last_ts = raw_ts
                if frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                    continue
            time.sleep(0.033 if active else 0.15)

    def latest_frame_bytes(self) -> bytes:
        """Lazy: return the most-recent raw frame as a JPEG (with heatmap
        overlay if enabled), encoding on demand and caching by frame ts.

        No encoding happens when nobody is polling this.  Concurrent callers
        share one encode per frame.
        """
        with self._lock:
            if self._latest_raw_ts == self._latest_frame_jpeg_ts:
                return bytes(self._latest_frame_jpeg)
            raw_frame = self._latest_raw_frame
            raw_ts = self._latest_raw_ts
            show_heatmap = self._show_heatmap
            heatmap = self._last_heatmap
        if raw_frame is None:
            return b""
        display_frame = raw_frame.copy()
        if show_heatmap and heatmap is not None:
            hm_full = cv2.resize(
                heatmap,
                (display_frame.shape[1], display_frame.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            colored_hm = cv2.applyColorMap(hm_full, cv2.COLORMAP_JET)
            display_frame = cv2.addWeighted(display_frame, 0.6, colored_hm, 0.4, 0)
        ok, buf = cv2.imencode(
            ".jpg", display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            return b""
        jpeg = buf.tobytes()
        with self._lock:
            if raw_ts >= self._latest_frame_jpeg_ts:
                self._latest_frame_jpeg = jpeg
                self._latest_frame_jpeg_ts = raw_ts
        return jpeg

    def start(self) -> dict:
        if self.running:
            raise ValueError("PO-GUISE is already running")
        self.stop_event = threading.Event()
        self.running = True
        self._publish_brain_event("system_status", {"vision_running": True})
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return {"status": "started"}

    def stop(self, wait: bool = True, timeout: float = 15.0) -> dict:
        if not self.running:
            raise ValueError("PO-GUISE is not running")
        with self._lock:
            self._stats["status_message"] = "Stopping PO-GUISE..."
            self._stats["phase"] = "stopping"
        self.stop_event.set()
        capture = self._capture
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if wait and self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
        self._finalize_runtime_refs()
        return {"status": "stopped" if not self.running else "stopping"}

    def shutdown(self, timeout: float = 15.0) -> None:
        if self.running:
            self.stop_event.set()
        capture = self._capture
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
        self._finalize_runtime_refs()

    def _finalize_runtime_refs(self) -> None:
        with self._cleanup_lock:
            self._model = None
            self._capture = None
            self._last_heatmap = None
            self._latest_raw_frame = None
            self._latest_raw_ts = 0.0
            self._latest_frame_jpeg = b""
            self._latest_frame_jpeg_ts = -1.0
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

    def _run(self) -> None:
        config = settings.load_section_config("poguise")
        ckpt_path = settings.POGUISE_ROOT / config["checkpoint"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        configure_cuda_runtime(device)

        with self._lock:
            self._stats = self._fresh_stats()
            self._stats["running"] = True
            self._stats["checkpoint"] = config["checkpoint"]
            self._stats["policy_mode"] = "starting"
            self._stats["policy_reason"] = "Loading PO-GUISE..."
            self._stats["phase"] = "loading_model"
            self._stats["status_message"] = "Loading PO-GUISE model..."
            self._stats["device"] = str(device)

        model = None
        init_error: Optional[Exception] = None
        try:
            with self.gpu_lock:
                model = load_model(str(ckpt_path), device)
                if config["fp16"] and device.type == "cuda":
                    model = model.half()
                self._model = model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            gc.collect()
            trim_process_heap()
        except Exception as exc:
            init_error = exc
            if device.type == "cuda":
                with self._lock:
                    self._stats["status_message"] = (
                        f"CUDA init failed ({exc}). Retrying on CPU..."
                    )
                    self._stats["device"] = "cpu"
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                gc.collect()
                trim_process_heap()
                device = torch.device("cpu")
                try:
                    model = load_model(str(ckpt_path), device)
                    self._model = model
                    init_error = None
                    gc.collect()
                    trim_process_heap()
                except Exception as cpu_exc:
                    init_error = cpu_exc

        if init_error is not None:
            with self._lock:
                self._stats["error"] = f"Model init failed: {init_error}"
                self._stats["running"] = False
                self._stats["phase"] = "error"
                self._stats["status_message"] = f"Model init failed: {init_error}"
            self.running = False
            self._publish_brain_event("system_status", {"vision_running": False})
            self._finalize_runtime_refs()
            return

        with self._lock:
            self._stats["phase"] = "opening_camera"
            self._stats["status_message"] = "Model loaded. Opening camera..."
            self._stats["device"] = str(device)

        cap = cv2.VideoCapture(int(config["camera"]))
        self._capture = cap
        while not cap.isOpened() and not self.stop_event.is_set():
            time.sleep(1.0)
            cap = cv2.VideoCapture(int(config["camera"]))
            self._capture = cap

        # Camera resolution is config-driven (settings.default_poguise_config)
        # so the nexus-agent stack and standalone poguise dashboard stay in
        # lock-step.  Driver picks the closest supported mode if the request
        # is not natively available (typical USB-UVC behaviour).
        cam_w = int(config.get("camera_width", 640))
        cam_h = int(config.get("camera_height", 480))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        if BUFFER_W <= BUFFER_H:
            scaled_w = SCALE
            scaled_h = int(math.floor(float(BUFFER_H) / BUFFER_W * SCALE))
        else:
            scaled_h = SCALE
            scaled_w = int(math.floor(float(BUFFER_W) / BUFFER_H * SCALE))
        crop_y = int(math.ceil((scaled_h - CROP) / 2))
        crop_x = int(math.ceil((scaled_w - CROP) / 2))
        inv_sy = BUFFER_H / scaled_h
        inv_sx = BUFFER_W / scaled_w
        hm_y0 = int(round(crop_y * inv_sy))
        hm_x0 = int(round(crop_x * inv_sx))
        hm_y1 = min(int(round((crop_y + CROP) * inv_sy)), BUFFER_H)
        hm_x1 = min(int(round((crop_x + CROP) * inv_sx)), BUFFER_W)

        buffer = collections.deque(maxlen=300)
        pred_history = collections.deque(maxlen=3)
        current_top_count = 0
        last_top_pred = None
        confirmed_action = "Waiting..."
        action_history = []
        inference_ms = 0.0
        fps_cam = 0.0
        frame_count = 0
        t_cam_prev = time.time()
        skipped_busy_gpu = 0
        skipped_policy = 0

        with self._lock:
            self._stats["camera_open"] = bool(cap.isOpened())
            self._stats["phase"] = "warming"
            self._stats["status_message"] = (
                f"Camera {config['camera']} ready. Buffering frames..."
                if cap.isOpened()
                else f"Waiting for camera {config['camera']}..."
            )

        try:
            while not self.stop_event.is_set():
                ret, raw_frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                t_now = time.time()
                fps_cam = 0.9 * fps_cam + 0.1 * (1.0 / max(t_now - t_cam_prev, 1e-6))
                t_cam_prev = t_now
                frame_count += 1

                # Fan out the raw frame to registered sinks (e.g. the Nexus
                # VisionService). Sinks are supposed to be cheap (stash +
                # return); we swallow errors so a misbehaving sink can't wedge
                # the camera loop.
                if self._frame_sinks:
                    with self._frame_sinks_lock:
                        sinks = list(self._frame_sinks)
                    for sink in sinks:
                        try:
                            sink(raw_frame, t_now)
                        except Exception:
                            pass

                small = cv2.resize(
                    raw_frame,
                    (BUFFER_W, BUFFER_H),
                    interpolation=cv2.INTER_AREA,
                )
                buffer.append((t_now, small))

                cutoff_exact = t_now - CONTEXT_SECONDS
                valid_frames = [frame for ts, frame in buffer if ts >= cutoff_exact]
                inferred_this_loop = False

                plan = self.policy.plan_poguise(
                    int(config["infer_every"]),
                    bool(self.voice_running_getter()),
                )
                effective_infer_every = int(plan["effective_infer_every"])
                startup_gate = getattr(self, "startup_gate", None)
                startup_ready = startup_gate is None or startup_gate.is_set()

                if (
                    startup_ready
                    and frame_count % effective_infer_every == 0
                    and len(valid_frames) >= N_FRAMES
                ):
                    if plan["should_pause"]:
                        skipped_policy += 1
                    else:
                        clip = preprocess_clip(valid_frames)
                        if clip is not None:
                            acquired = self.gpu_lock.acquire(blocking=False)
                            if not acquired:
                                skipped_busy_gpu += 1
                            else:
                                try:
                                    want_heatmap = self._show_heatmap
                                    t0 = time.time()
                                    raw_probs, hm_tensor = run_inference(
                                        model,
                                        clip,
                                        device,
                                        bool(config["fp16"]) and device.type == "cuda",
                                        return_heatmap=want_heatmap,
                                    )
                                    inference_ms = (time.time() - t0) * 1000.0
                                    inferred_this_loop = True
                                finally:
                                    self.gpu_lock.release()

                                pred_history.append(raw_probs)
                                self.policy.record_poguise_step(inference_ms)

                                if hm_tensor is not None:
                                    hm_np = np.asarray(hm_tensor).squeeze(0)
                                    while len(hm_np.shape) > 2:
                                        hm_np = np.max(hm_np, axis=0)
                                    hm_np = np.clip(hm_np, 0, None)
                                    if hm_np.max() > 0:
                                        hm_np = (hm_np / hm_np.max()) * 255.0
                                    hm_uint8 = hm_np.astype(np.uint8)
                                    hm_roi = cv2.resize(
                                        hm_uint8,
                                        (hm_x1 - hm_x0, hm_y1 - hm_y0),
                                        interpolation=cv2.INTER_CUBIC,
                                    )
                                    hm_full = np.zeros((BUFFER_H, BUFFER_W), dtype=np.uint8)
                                    hm_full[hm_y0:hm_y1, hm_x0:hm_x1] = hm_roi
                                    self._last_heatmap = hm_full

                # Publish shallow ref for lazy JPEG encode on demand
                # (see latest_frame_bytes()).  The old code blended heatmap +
                # imencode'd every frame here — at 30 fps that was a full CPU
                # core of composite-and-encode work that got thrown away any
                # time no browser or thumbnail poller was connected.
                with self._lock:
                    self._latest_raw_frame = raw_frame
                    self._latest_raw_ts = t_now
                buffer_pct = min(
                    100.0,
                    100.0 * len(valid_frames) / (fps_cam * CONTEXT_SECONDS + 1e-5),
                )

                if pred_history:
                    smoothed_probs = np.mean(pred_history, axis=0)
                    top1_idx = int(np.argmax(smoothed_probs))
                    top1_prob = float(smoothed_probs[top1_idx])
                    top1_label = CS_CLASSES[top1_idx]
                    if inferred_this_loop and top1_prob > float(config["confidence_threshold"]):
                        self._publish_brain_event(
                            "action_detected",
                            {
                                "action": top1_label,
                                "confidence": top1_prob,
                            },
                        )
                    if top1_prob > float(config["confidence_threshold"]):
                        if top1_label == last_top_pred:
                            current_top_count += 1
                        else:
                            last_top_pred = top1_label
                            current_top_count = 1
                        if (
                            current_top_count >= int(config["debounce_frames"])
                            and top1_label != confirmed_action
                        ):
                            if confirmed_action != "Waiting...":
                                action_history.insert(
                                    0,
                                    {
                                        "time": time.strftime("%H:%M:%S", time.localtime()),
                                        "label": pretty_action(confirmed_action),
                                    },
                                )
                                if len(action_history) > 6:
                                    action_history.pop()
                            confirmed_action = top1_label
                else:
                    smoothed_probs = np.ones(len(CS_CLASSES), dtype=np.float32) / len(CS_CLASSES)

                top5_idx = np.argsort(smoothed_probs)[::-1][:5]
                top_preds = [
                    {
                        "label": pretty_action(CS_CLASSES[idx]),
                        "raw_label": CS_CLASSES[idx],
                        "prob": float(smoothed_probs[idx]) * 100.0,
                    }
                    for idx in top5_idx
                ]

                with self._lock:
                    # JPEG is now produced lazily by latest_frame_bytes().
                    self._stats = {
                        "running": True,
                        "fps_cam": float(fps_cam),
                        "inference_ms": float(inference_ms),
                        "buffer_pct": float(buffer_pct),
                        "current_action": pretty_action(confirmed_action),
                        "action_history": copy.deepcopy(action_history),
                        "predictions": top_preds,
                        "policy_mode": plan["mode"],
                        "policy_reason": plan["reason"],
                        "effective_infer_every": effective_infer_every,
                        "skipped_due_to_busy_gpu": skipped_busy_gpu,
                        "skipped_due_to_policy": skipped_policy,
                        "heatmap_enabled": self._show_heatmap,
                        "camera_open": bool(cap.isOpened()),
                        "checkpoint": config["checkpoint"],
                        "phase": "live" if inference_ms > 0 else "warming",
                        "status_message": (
                            "PO-GUISE live."
                            if inference_ms > 0
                            else "Camera ready. Waiting for the first inference..."
                        ),
                        "error": None,
                    }
        except Exception as exc:
            with self._lock:
                self._stats["error"] = str(exc)
                self._stats["phase"] = "error"
                self._stats["status_message"] = f"PO-GUISE error: {exc}"
        finally:
            try:
                cap.release()
            except Exception:
                pass
            self.running = False
            self._publish_brain_event("system_status", {"vision_running": False})
            with self._lock:
                self._stats = self._fresh_stats()
                self._stats["checkpoint"] = config["checkpoint"]
            self._finalize_runtime_refs()
