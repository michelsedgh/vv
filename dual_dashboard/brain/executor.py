"""
Layer 5: Action Executor

Executes actions from triggered rules. Currently a stub for:
    - smart_home: Will connect to Home Assistant / AppDaemon later
    - tts: Text-to-speech feedback (stub)
    - notify: Dashboard notifications via WebSocket

All actions are logged for the dashboard to display.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Coroutine

log = logging.getLogger("brain.executor")


class ActionExecutor:
    """
    Executes actions from the rule engine.
    
    Currently stubs smart_home and TTS actions.
    Notifications are sent to the dashboard via a callback.
    """

    def __init__(self):
        self._action_log: List[dict] = []
        self._notification_callback: Optional[Callable] = None
        self._pending_speech: List[dict] = []  # For dashboard display

        # Home Assistant config (for later)
        self.ha_url: Optional[str] = None
        self.ha_token: Optional[str] = None

    def set_notification_callback(self, callback: Callable) -> None:
        """Set callback for sending notifications to the dashboard."""
        self._notification_callback = callback

    async def execute(self, action: dict, rule_id: str = "", rule_description: str = "") -> dict:
        """
        Execute an action. Returns a result dict with execution details.
        """
        action_type = action.get("type", "unknown")
        result = {
            "rule_id": rule_id,
            "rule_description": rule_description,
            "action_type": action_type,
            "action": action,
            "executed_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp": time.time(),
            "status": "pending",
            "details": "",
        }

        try:
            if action_type == "smart_home":
                result.update(await self._execute_smart_home(action))

            elif action_type == "tts":
                result.update(await self._execute_tts(action))

            elif action_type == "notify":
                result.update(await self._execute_notify(action))

            else:
                result["status"] = "error"
                result["details"] = f"Unknown action type: {action_type}"

        except Exception as exc:
            result["status"] = "error"
            result["details"] = f"Execution failed: {exc}"
            log.error("Action execution failed: %s", exc)

        # Log the action
        self._action_log.append(result)
        if len(self._action_log) > 200:
            self._action_log = self._action_log[-200:]

        log.info(
            "Executed action: [%s] %s — %s (%s)",
            rule_id or "direct",
            action_type,
            action.get("command", action.get("message", "")),
            result["status"],
        )

        return result

    async def _execute_smart_home(self, action: dict) -> dict:
        """
        Execute a smart home command.
        Currently a stub — logs the command for dashboard display.
        Will connect to Home Assistant REST API later.
        """
        command = action.get("command", "unknown")
        params = action.get("params", {})

        if self.ha_url and self.ha_token:
            # TODO: actual Home Assistant API call
            return {
                "status": "executed",
                "details": f"[HA] {command}({params})",
            }

        # Stub mode: just log it
        return {
            "status": "simulated",
            "details": f"[STUB] smart_home.{command}({params}) — Home Assistant not connected",
        }

    async def _execute_tts(self, action: dict) -> dict:
        """
        Execute text-to-speech action.
        Currently a stub — stores the message for dashboard display.
        """
        message = action.get("message", "")
        self._pending_speech.append({
            "message": message,
            "at": datetime.now().isoformat(timespec="seconds"),
        })
        if len(self._pending_speech) > 20:
            self._pending_speech = self._pending_speech[-20:]

        # Also send as notification to dashboard
        if self._notification_callback:
            await self._notification_callback({
                "type": "tts",
                "message": message,
            })

        return {
            "status": "simulated",
            "details": f"[STUB] TTS: \"{message}\"",
        }

    async def _execute_notify(self, action: dict) -> dict:
        """Send a notification to the dashboard."""
        message = action.get("message", "")
        level = action.get("level", "info")

        if self._notification_callback:
            await self._notification_callback({
                "type": "notification",
                "level": level,
                "message": message,
            })

        return {
            "status": "executed",
            "details": f"Dashboard notification: {message}",
        }

    async def ask(self, question: str, rule_id: str) -> dict:
        """
        Ask the user a question via TTS + dashboard notification.
        The response will come back as a user_response event on the bus.
        """
        result = await self.execute(
            {
                "type": "tts",
                "message": question,
            },
            rule_id=rule_id,
            rule_description=f"Asking: {question}",
        )

        # Also send a confirmation request to dashboard
        if self._notification_callback:
            await self._notification_callback({
                "type": "confirmation_request",
                "rule_id": rule_id,
                "question": question,
            })

        return result

    async def notify(self, message: str) -> None:
        """Send a simple notification."""
        await self.execute(
            {"type": "notify", "message": message},
            rule_description=message,
        )

    def get_action_log(self, count: int = 30) -> List[dict]:
        """Get recent action log for dashboard."""
        return self._action_log[-count:]

    def get_pending_speech(self) -> List[dict]:
        """Get pending TTS messages (for dashboard display)."""
        return list(self._pending_speech)

    def clear_pending_speech(self) -> None:
        """Clear pending TTS messages."""
        self._pending_speech.clear()

    def stats(self) -> Dict[str, Any]:
        """Get executor stats for dashboard."""
        total = len(self._action_log)
        executed = sum(1 for a in self._action_log if a["status"] in ("executed", "simulated"))
        errors = sum(1 for a in self._action_log if a["status"] == "error")
        return {
            "total_actions": total,
            "executed": executed,
            "errors": errors,
            "ha_connected": bool(self.ha_url and self.ha_token),
            "pending_speech": len(self._pending_speech),
        }
