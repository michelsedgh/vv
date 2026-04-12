from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from dual_dashboard import settings
    from dual_dashboard.policy import SharedInferencePolicy
    from dual_dashboard.poguise_service import PoguiseService
    from dual_dashboard.voice_service import VoiceService
else:
    from . import settings
    from .policy import SharedInferencePolicy
    from .poguise_service import PoguiseService
    from .voice_service import VoiceService

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Voice + PO-GUISE Dual Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

gpu_lock = threading.Lock()
policy = SharedInferencePolicy(settings.load_section_config("scheduler"))
voice_service = VoiceService(gpu_lock, policy)
poguise_service = PoguiseService(
    gpu_lock,
    policy,
    voice_running_getter=lambda: voice_service.monitor.running,
)


class ConfigUpdateRequest(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict)


class EnrollRequest(BaseModel):
    name: str
    duration: int = 30


class DiagRecordRequest(BaseModel):
    label: str
    duration: int = 5


def _raise_bad_request(exc: Exception):
    raise HTTPException(400, str(exc))


@app.on_event("startup")
async def on_startup() -> None:
    voice_service.set_loop(asyncio.get_running_loop())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    poguise_service.shutdown(timeout=20.0)
    voice_service.shutdown(timeout=20.0)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/bootstrap")
async def api_bootstrap():
    return {
        "voice": {
            "status": voice_service.status(),
            "editor": voice_service.config_payload(),
            "speakers": voice_service.list_speakers(),
            "diag_clips": voice_service.diag_clips(),
        },
        "poguise": {
            "status": poguise_service.status(),
            "editor": poguise_service.config_payload(),
        },
        "scheduler": {
            "status": policy.snapshot(),
            "editor": settings.editor_payload("scheduler"),
        },
        "system": {
            "voice_running": voice_service.monitor.running,
            "poguise_running": poguise_service.running,
        },
    }


@app.get("/api/system/status")
async def api_system_status():
    return {
        "voice": voice_service.status(),
        "poguise": poguise_service.status(),
        "scheduler": policy.snapshot(),
    }


@app.post("/api/system/start")
async def api_system_start():
    if not voice_service.monitor.running:
        try:
            voice_service.start()
        except Exception as exc:
            _raise_bad_request(exc)
    if not poguise_service.running:
        try:
            poguise_service.start()
        except Exception as exc:
            _raise_bad_request(exc)
    return {"status": "started"}


@app.post("/api/system/stop")
async def api_system_stop():
    if poguise_service.running:
        poguise_service.stop(wait=True, timeout=20.0)
    if voice_service.monitor.running:
        voice_service.stop(wait=True, timeout=20.0)
    return {"status": "stopped"}


@app.get("/api/voice/status")
async def api_voice_status():
    return voice_service.status()


@app.post("/api/voice/start")
async def api_voice_start():
    try:
        return voice_service.start()
    except Exception as exc:
        _raise_bad_request(exc)


@app.post("/api/voice/stop")
async def api_voice_stop():
    try:
        return voice_service.stop(wait=True, timeout=20.0)
    except Exception as exc:
        _raise_bad_request(exc)


@app.get("/api/voice/speakers")
async def api_voice_speakers():
    return voice_service.list_speakers()


@app.delete("/api/voice/speakers/{name}")
async def api_voice_delete_speaker(name: str):
    try:
        return voice_service.delete_speaker(name)
    except FileNotFoundError:
        raise HTTPException(404, f"Speaker '{name}' not found")


@app.get("/api/voice/config/editor")
async def api_voice_editor():
    return voice_service.config_payload()


@app.put("/api/voice/config")
async def api_voice_update_config(req: ConfigUpdateRequest):
    try:
        return voice_service.update_config(req.values)
    except Exception as exc:
        _raise_bad_request(exc)


@app.delete("/api/voice/config")
async def api_voice_reset_config():
    try:
        return voice_service.reset_config()
    except Exception as exc:
        _raise_bad_request(exc)


@app.post("/api/voice/enroll")
async def api_voice_enroll(req: EnrollRequest):
    try:
        return voice_service.enroll(req.name, req.duration)
    except Exception as exc:
        _raise_bad_request(exc)


@app.get("/api/voice/enroll/status/{name}")
async def api_voice_enroll_status(name: str):
    return voice_service.enroll_status(name)


@app.post("/api/voice/diag/record")
async def api_voice_diag_record(req: DiagRecordRequest):
    return voice_service.record_diag(req.label, req.duration)


@app.get("/api/voice/diag/clips")
async def api_voice_diag_clips():
    return voice_service.diag_clips()


@app.websocket("/ws/live")
async def api_voice_ws(ws: WebSocket):
    await voice_service.websocket_loop(ws)


@app.get("/api/poguise/status")
async def api_poguise_status():
    return poguise_service.status()


@app.post("/api/poguise/start")
async def api_poguise_start():
    try:
        return poguise_service.start()
    except Exception as exc:
        _raise_bad_request(exc)


@app.post("/api/poguise/stop")
async def api_poguise_stop():
    try:
        return poguise_service.stop(wait=True, timeout=20.0)
    except Exception as exc:
        _raise_bad_request(exc)


@app.get("/api/poguise/config/editor")
async def api_poguise_editor():
    return poguise_service.config_payload()


@app.put("/api/poguise/config")
async def api_poguise_update_config(req: ConfigUpdateRequest):
    try:
        return poguise_service.update_config(req.values)
    except Exception as exc:
        _raise_bad_request(exc)


@app.delete("/api/poguise/config")
async def api_poguise_reset_config():
    try:
        return poguise_service.reset_config()
    except Exception as exc:
        _raise_bad_request(exc)


@app.post("/api/poguise/heatmap/toggle")
async def api_poguise_toggle_heatmap():
    return poguise_service.toggle_heatmap()


@app.get("/api/poguise/video_feed")
async def api_poguise_video_feed():
    return StreamingResponse(
        poguise_service.video_feed(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/poguise/frame.jpg")
async def api_poguise_frame():
    frame = poguise_service.latest_frame_bytes()
    if not frame:
        return Response(status_code=204)
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/scheduler/config/editor")
async def api_scheduler_editor():
    return settings.editor_payload("scheduler")


@app.put("/api/scheduler/config")
async def api_scheduler_update(req: ConfigUpdateRequest):
    if voice_service.monitor.running or poguise_service.running:
        raise HTTPException(400, "Stop both models before changing scheduler settings")
    try:
        payload = settings.apply_editor_update("scheduler", req.values)
    except Exception as exc:
        _raise_bad_request(exc)
    policy.update_config(settings.load_section_config("scheduler"))
    return payload


@app.delete("/api/scheduler/config")
async def api_scheduler_reset():
    if voice_service.monitor.running or poguise_service.running:
        raise HTTPException(400, "Stop both models before resetting scheduler settings")
    payload = settings.reset_editor_section("scheduler")
    policy.update_config(settings.load_section_config("scheduler"))
    return payload


@app.get("/healthz")
async def healthz():
    return JSONResponse({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8055)
    args = parser.parse_args()
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.asgi").setLevel(logging.WARNING)
    try:
        uvicorn.run(
            "dual_dashboard.app:app",
            host=args.host,
            port=args.port,
            reload=False,
            log_level="warning",
            access_log=False,
            ws_ping_interval=30,
            ws_ping_timeout=10,
            timeout_graceful_shutdown=20,
        )
    finally:
        poguise_service.shutdown(timeout=20.0)
        voice_service.shutdown(timeout=20.0)


if __name__ == "__main__":
    main()
