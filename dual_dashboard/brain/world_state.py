"""
Layer 2: World State Tracker

Maintains a single dict that always represents "what is happening right now."
Every layer can read it. Only the tracker writes to it.

This is the "short-term memory" of the house — when the LLM needs to decide
something, it gets this snapshot instead of raw sensor data.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .event_bus import Event

log = logging.getLogger("brain.world_state")

# How long before a person is considered "gone" if no updates received
PRESENCE_TIMEOUT_SEC = 300  # 5 minutes


def _time_of_day(hour: Optional[int] = None) -> str:
    """Return human-readable time of day period."""
    if hour is None:
        hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


class WorldState:
    """
    Fused world state from all sensors.

    Thread-safe read via snapshot(). Only update_from_event() writes.
    """

    def __init__(self):
        self.state: Dict[str, Any] = {
            # Who is here
            "people_present": {},       # {"michel": {"enrolled": True, "since": "...", "speaking": False, "last_seen": <ts>}}
            "people_count": 0,

            # What is happening (from PO-GUISE)
            "current_action": None,     # "Eat.Attable"
            "action_confidence": 0.0,
            "action_since": None,       # ISO timestamp when this action started
            "action_stable": False,     # True if action held for > debounce period

            # Speech tracking
            "last_speech": {},          # {"michel": {"text": "...", "at": "..."}}
            "recent_speech": [],        # last 10 utterances

            # Action history
            "recent_actions": [],       # last 10 action transitions

            # Time context
            "time_of_day": _time_of_day(),
            "current_time": datetime.now().isoformat(timespec="seconds"),

            # System state
            "voice_running": False,
            "vision_running": False,
        }

        # Internal tracking
        self._action_debounce_start: Optional[float] = None
        self._action_debounce_target: Optional[str] = None
        self._action_debounce_sec = 3.0  # seconds an action must be stable

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy of the current world state (thread-safe read)."""
        # Update time on every snapshot
        now = datetime.now()
        self.state["time_of_day"] = _time_of_day(now.hour)
        self.state["current_time"] = now.isoformat(timespec="seconds")
        self.state["people_count"] = len(self.state["people_present"])

        # Expire stale presence entries
        cutoff = time.time() - PRESENCE_TIMEOUT_SEC
        expired = [
            name for name, info in self.state["people_present"].items()
            if info.get("last_seen", 0) < cutoff
        ]
        for name in expired:
            log.info("Presence expired for %s (no update for %ds)", name, PRESENCE_TIMEOUT_SEC)
            del self.state["people_present"][name]
            self.state["people_count"] = len(self.state["people_present"])

        import copy
        return copy.deepcopy(self.state)

    def update_from_event(self, event: Event) -> List[Dict[str, Any]]:
        """
        Update world state from an event. Returns a list of derived events
        that should be published back to the bus (e.g., action_changed).
        """
        derived_events: List[Dict[str, Any]] = []

        if event.type == "speaker_active":
            who = event.data.get("who", "unknown")
            enrolled = event.data.get("enrolled", False)
            now_ts = event.timestamp

            if who not in self.state["people_present"]:
                # New person detected
                derived_events.append({
                    "type": "person_entered",
                    "data": {"who": who, "enrolled": enrolled},
                })

            self.state["people_present"][who] = {
                "enrolled": enrolled,
                "since": datetime.fromtimestamp(now_ts).isoformat(timespec="seconds"),
                "speaking": True,
                "last_seen": now_ts,
                "confidence": event.data.get("confidence", 0.0),
            }
            self.state["people_count"] = len(self.state["people_present"])

        elif event.type == "speaker_silent":
            who = event.data.get("who", "unknown")
            if who in self.state["people_present"]:
                self.state["people_present"][who]["speaking"] = False
                self.state["people_present"][who]["last_seen"] = event.timestamp

        elif event.type == "action_detected":
            new_action = event.data.get("action")
            confidence = event.data.get("confidence", 0.0)

            # Update confidence even if same action
            self.state["action_confidence"] = confidence

            if new_action != self.state["current_action"]:
                old_action = self.state["current_action"]

                # Track debounce: action must be stable for N seconds
                if new_action != self._action_debounce_target:
                    self._action_debounce_target = new_action
                    self._action_debounce_start = event.timestamp
                    self.state["action_stable"] = False
                elif (self._action_debounce_start and
                      event.timestamp - self._action_debounce_start >= self._action_debounce_sec):
                    # Action is stable — commit the transition
                    self.state["action_stable"] = True
                    self.state["current_action"] = new_action
                    self.state["action_since"] = datetime.fromtimestamp(event.timestamp).isoformat(timespec="seconds")

                    # Record in history
                    transition = {
                        "from": old_action,
                        "to": new_action,
                        "at": datetime.fromtimestamp(event.timestamp).isoformat(timespec="seconds"),
                        "confidence": confidence,
                    }
                    self.state["recent_actions"].append(transition)
                    self.state["recent_actions"] = self.state["recent_actions"][-10:]

                    # Publish action_changed event
                    derived_events.append({
                        "type": "action_changed",
                        "data": {
                            "from": old_action,
                            "to": new_action,
                            "confidence": confidence,
                        },
                    })
                    log.info("Action changed: %s → %s (%.0f%%)", old_action, new_action, confidence * 100)
            else:
                # Same action, refresh stability
                self.state["action_stable"] = True
                self._action_debounce_target = new_action

        elif event.type == "speech_text":
            who = event.data.get("who", "unknown")
            text = event.data.get("text", "")
            ts_iso = datetime.fromtimestamp(event.timestamp).isoformat(timespec="seconds")

            self.state["last_speech"][who] = {
                "text": text,
                "at": ts_iso,
            }
            self.state["recent_speech"].append({
                "who": who,
                "text": text,
                "at": ts_iso,
            })
            self.state["recent_speech"] = self.state["recent_speech"][-10:]

            # Update presence
            if who in self.state["people_present"]:
                self.state["people_present"][who]["last_seen"] = event.timestamp

        elif event.type == "person_left":
            who = event.data.get("who")
            if who and who in self.state["people_present"]:
                del self.state["people_present"][who]
                self.state["people_count"] = len(self.state["people_present"])

        elif event.type == "system_status":
            if "voice_running" in event.data:
                self.state["voice_running"] = event.data["voice_running"]
            if "vision_running" in event.data:
                self.state["vision_running"] = event.data["vision_running"]

        return derived_events

    def get_llm_context(self) -> str:
        """Format world state for LLM consumption."""
        snap = self.snapshot()
        lines = [
            f"Current time: {snap['current_time']} ({snap['time_of_day']})",
            f"People present: {list(snap['people_present'].keys()) or 'nobody'}",
        ]

        for name, info in snap["people_present"].items():
            status = "speaking" if info.get("speaking") else "quiet"
            kind = "enrolled" if info.get("enrolled") else "unknown"
            lines.append(f"  - {name}: {kind}, {status}")

        if snap["current_action"]:
            lines.append(f"Current activity: {snap['current_action']} ({snap['action_confidence']:.0%} confidence, since {snap['action_since']})")
        else:
            lines.append("Current activity: none detected")

        if snap["recent_actions"]:
            lines.append("Recent activity transitions:")
            for t in snap["recent_actions"][-5:]:
                lines.append(f"  {t['at']}: {t['from']} → {t['to']}")

        if snap["recent_speech"]:
            lines.append("Recent speech:")
            for s in snap["recent_speech"][-5:]:
                lines.append(f"  {s['at']} [{s['who']}]: \"{s['text']}\"")

        return "\n".join(lines)
