from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict


class SharedInferencePolicy:
    """Small voice-first scheduler for GPU sharing inside one process."""

    def __init__(self, config: Dict[str, Any]):
        self._lock = threading.Lock()
        self._config = dict(config)
        self._voice_ms = deque(maxlen=12)
        self._poguise_ms = deque(maxlen=12)
        self._hold_until = 0.0

    def update_config(self, config: Dict[str, Any]) -> None:
        with self._lock:
            self._config = dict(config)

    def record_voice_step(self, infer_ms: float) -> None:
        with self._lock:
            self._voice_ms.append(float(infer_ms))

    def record_poguise_step(self, infer_ms: float) -> None:
        with self._lock:
            self._poguise_ms.append(float(infer_ms))

    def plan_poguise(self, base_infer_every: int, voice_running: bool) -> Dict[str, Any]:
        with self._lock:
            cfg = dict(self._config)
            recent_voice = list(self._voice_ms)
            now = time.time()
            last_voice = recent_voice[-1] if recent_voice else None
            avg_voice = (
                sum(recent_voice) / len(recent_voice)
                if recent_voice
                else None
            )

            effective = max(1, int(base_infer_every))
            mode = "voice-idle"
            reason = "Voice is not running"

            if not cfg.get("enabled", True):
                mode = "scheduler-off"
                reason = "Voice protection disabled"
            elif voice_running:
                mode = "balanced"
                reason = "Voice is within target"
                if last_voice is not None and (
                    last_voice >= float(cfg["voice_hard_limit_ms"])
                    or (
                        avg_voice is not None
                        and avg_voice >= float(cfg["voice_hard_limit_ms"])
                    )
                ):
                    self._hold_until = max(
                        self._hold_until,
                        now + float(cfg["pause_after_voice_sec"]),
                    )
                    effective = max(
                        effective,
                        int(cfg["max_infer_every"]),
                    )
                    mode = "protect-hard"
                    reason = f"Voice spike at {last_voice:.1f} ms"
                elif last_voice is not None and (
                    last_voice >= float(cfg["voice_target_ms"])
                    or (
                        avg_voice is not None
                        and avg_voice >= float(cfg["voice_target_ms"])
                    )
                ):
                    effective = min(
                        int(cfg["max_infer_every"]),
                        max(effective, int(base_infer_every) + 2),
                    )
                    mode = "protect-soft"
                    reason = f"Voice above target at {last_voice:.1f} ms"

            should_pause = (
                bool(cfg.get("enabled", True))
                and voice_running
                and now < self._hold_until
            )
            if should_pause:
                mode = "paused"
                reason = f"Cooling down after Voice spike for {self._hold_until - now:.2f}s"

            return {
                "effective_infer_every": int(effective),
                "mode": mode,
                "reason": reason,
                "should_pause": should_pause,
                "last_voice_ms": last_voice,
                "avg_voice_ms": avg_voice,
                "hold_remaining_ms": max(0.0, (self._hold_until - now) * 1000.0),
            }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            voice_vals = list(self._voice_ms)
            pog_vals = list(self._poguise_ms)
            now = time.time()
            return {
                "config": dict(self._config),
                "last_voice_ms": voice_vals[-1] if voice_vals else None,
                "avg_voice_ms": (
                    sum(voice_vals) / len(voice_vals)
                    if voice_vals
                    else None
                ),
                "last_poguise_ms": pog_vals[-1] if pog_vals else None,
                "avg_poguise_ms": (
                    sum(pog_vals) / len(pog_vals)
                    if pog_vals
                    else None
                ),
                "hold_remaining_ms": max(0.0, (self._hold_until - now) * 1000.0),
            }
