"""
Layer 3: Rule Engine

Evaluates automation rules against world state and incoming events.
Rules are stored in YAML and can be created/modified by the LLM.

Rule lifecycle:
    1. Event arrives on the bus
    2. Engine checks each active rule's trigger type against event type
    3. If trigger matches, evaluate conditions against world state
    4. If conditions met, fire the rule (respecting permission model)

Permission model:
    - auto:    Execute immediately, no feedback
    - notify:  Execute and tell the user
    - ask:     Ask user for yes/no before executing
    - suggest: Only mention it, don't execute
"""
from __future__ import annotations

import copy
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .event_bus import Event
from .world_state import WorldState, _time_of_day

log = logging.getLogger("brain.rules")


def _default_rules() -> List[dict]:
    """Default rules that ship with the system."""
    return [
        {
            "id": "dinner_lights",
            "description": "Dim lights when eating at table in the evening",
            "trigger": {
                "type": "action_changed",
                "action": "Eat.Attable",
                "conditions": [
                    {"time_of_day": ["evening", "night"]},
                    {"people_present_any": True},
                ],
            },
            "action": {
                "type": "smart_home",
                "command": "dim_lights",
                "params": {"brightness": 40, "area": "dining"},
            },
            "permission": "notify",
            "active": True,
            "cooldown_sec": 300,
            "expires": None,
            "created_by": "system",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "leave_lights_off",
            "description": "Turn off all lights when everyone leaves",
            "trigger": {
                "type": "person_left",
                "conditions": [
                    {"people_present_count": 0},
                ],
            },
            "action": {
                "type": "smart_home",
                "command": "lights_off",
                "params": {},
            },
            "permission": "auto",
            "active": True,
            "cooldown_sec": 60,
            "expires": None,
            "created_by": "system",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "cooking_ventilation",
            "description": "Turn on kitchen ventilation when cooking",
            "trigger": {
                "type": "action_changed",
                "action_in": ["Cook.Stir", "Cook.Usestove", "Cook.Cut"],
                "conditions": [],
            },
            "action": {
                "type": "smart_home",
                "command": "ventilation_on",
                "params": {"area": "kitchen"},
            },
            "permission": "notify",
            "active": True,
            "cooldown_sec": 600,
            "expires": None,
            "created_by": "system",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        {
            "id": "tv_ambiance",
            "description": "Set TV ambiance lighting when watching TV in the evening",
            "trigger": {
                "type": "action_changed",
                "action": "WatchTV",
                "conditions": [
                    {"time_of_day": ["evening", "night"]},
                ],
            },
            "action": {
                "type": "smart_home",
                "command": "scene_activate",
                "params": {"scene": "tv_ambiance"},
            },
            "permission": "suggest",
            "active": True,
            "cooldown_sec": 600,
            "expires": None,
            "created_by": "system",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    ]


class RuleEngine:
    """
    Evaluates rules against events and world state.
    """

    def __init__(self, rules_path: str, world: WorldState):
        self.rules_path = Path(rules_path)
        self.world = world
        self._rules: List[dict] = []
        self._pending_confirmations: Dict[str, dict] = {}  # rule_id -> {action, rule, asked_at}
        self._last_fired: Dict[str, float] = {}  # rule_id -> timestamp of last fire
        self._fire_history: List[dict] = []  # For dashboard display
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from YAML file, or create with defaults."""
        if self.rules_path.exists():
            try:
                with open(self.rules_path) as f:
                    data = yaml.safe_load(f) or {}
                self._rules = data.get("rules", [])
                log.info("Loaded %d rules from %s", len(self._rules), self.rules_path)
                return
            except Exception as exc:
                log.error("Failed to load rules: %s", exc)

        # Create default rules
        self._rules = _default_rules()
        self._save_rules()
        log.info("Created default rules file at %s", self.rules_path)

    def _save_rules(self) -> None:
        """Persist rules to YAML file."""
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rules_path, "w") as f:
            yaml.safe_dump({"rules": self._rules}, f, sort_keys=False, default_flow_style=False)

    @property
    def rules(self) -> List[dict]:
        return copy.deepcopy(self._rules)

    def get_rule(self, rule_id: str) -> Optional[dict]:
        for rule in self._rules:
            if rule["id"] == rule_id:
                return copy.deepcopy(rule)
        return None

    def add_rule(self, rule: dict) -> dict:
        """Add a new rule. Returns the added rule."""
        # Ensure required fields
        if "id" not in rule:
            rule["id"] = f"rule_{uuid.uuid4().hex[:8]}"
        rule.setdefault("active", True)
        rule.setdefault("permission", "ask")
        rule.setdefault("cooldown_sec", 60)
        rule.setdefault("expires", None)
        rule.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        # Check for duplicate ID
        existing_ids = {r["id"] for r in self._rules}
        if rule["id"] in existing_ids:
            rule["id"] = f"{rule['id']}_{uuid.uuid4().hex[:4]}"

        self._rules.append(rule)
        self._save_rules()
        log.info("Added rule: %s — %s", rule["id"], rule.get("description", ""))
        return copy.deepcopy(rule)

    def update_rule(self, rule_id: str, changes: dict) -> Optional[dict]:
        """Update fields of an existing rule."""
        for i, rule in enumerate(self._rules):
            if rule["id"] == rule_id:
                rule.update(changes)
                self._rules[i] = rule
                self._save_rules()
                log.info("Updated rule %s: %s", rule_id, changes)
                return copy.deepcopy(rule)
        return None

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r["id"] != rule_id]
        if len(self._rules) < before:
            self._save_rules()
            log.info("Deleted rule: %s", rule_id)
            return True
        return False

    def toggle_rule(self, rule_id: str) -> Optional[dict]:
        """Toggle a rule's active state."""
        for rule in self._rules:
            if rule["id"] == rule_id:
                rule["active"] = not rule["active"]
                self._save_rules()
                return copy.deepcopy(rule)
        return None

    def evaluate(self, event: Event) -> List[dict]:
        """
        Evaluate all rules against an event. Returns a list of
        rule fire results with their decisions.
        """
        results = []
        now = time.time()

        for rule in self._rules:
            if not rule.get("active", True):
                continue

            # Check expiry
            if rule.get("expires"):
                if rule["expires"] == "fired":
                    continue
                try:
                    exp_dt = datetime.fromisoformat(rule["expires"])
                    if datetime.now() > exp_dt:
                        continue
                except (ValueError, TypeError):
                    pass

            # Check trigger match
            if not self._trigger_matches(rule.get("trigger", {}), event):
                continue

            # Check conditions against world state
            conditions = rule.get("trigger", {}).get("conditions", [])
            state = self.world.state
            if not self._conditions_met(conditions, state):
                result = {
                    "rule_id": rule["id"],
                    "rule_description": rule.get("description", ""),
                    "decision": "conditions_not_met",
                    "event": event.to_dict(),
                    "timestamp": now,
                    "details": "Trigger matched but conditions not satisfied",
                }
                results.append(result)
                continue

            # Check cooldown
            cooldown = rule.get("cooldown_sec", 0)
            last_fired = self._last_fired.get(rule["id"], 0)
            if cooldown and (now - last_fired) < cooldown:
                remaining = cooldown - (now - last_fired)
                result = {
                    "rule_id": rule["id"],
                    "rule_description": rule.get("description", ""),
                    "decision": "cooldown",
                    "event": event.to_dict(),
                    "timestamp": now,
                    "details": f"In cooldown, {remaining:.0f}s remaining",
                }
                results.append(result)
                continue

            # Rule should fire!
            permission = rule.get("permission", "ask")
            result = {
                "rule_id": rule["id"],
                "rule_description": rule.get("description", ""),
                "decision": f"fire_{permission}",
                "permission": permission,
                "action": rule.get("action", {}),
                "event": event.to_dict(),
                "timestamp": now,
                "details": f"Rule triggered with permission={permission}",
            }
            results.append(result)

            # Record fire time
            self._last_fired[rule["id"]] = now

            # Handle one-time rules
            if rule.get("expires") == "once":
                rule["expires"] = "fired"
                self._save_rules()

            # Store in fire history
            self._fire_history.append(result)
            if len(self._fire_history) > 100:
                self._fire_history = self._fire_history[-100:]

        return results

    def confirm_pending(self, rule_id: str, approved: bool) -> Optional[dict]:
        """Handle user confirmation for an 'ask' permission rule."""
        pending = self._pending_confirmations.pop(rule_id, None)
        if pending:
            return {
                "rule_id": rule_id,
                "approved": approved,
                "action": pending["action"] if approved else None,
            }
        return None

    def add_pending_confirmation(self, rule_id: str, action: dict, rule_description: str) -> None:
        """Store a pending confirmation for 'ask' permission rules."""
        self._pending_confirmations[rule_id] = {
            "action": action,
            "description": rule_description,
            "asked_at": time.time(),
        }

    def get_pending_confirmations(self) -> List[dict]:
        """Get all pending confirmations for the dashboard."""
        now = time.time()
        result = []
        expired = []
        for rule_id, info in self._pending_confirmations.items():
            age = now - info["asked_at"]
            if age > 120:  # 2 minute timeout
                expired.append(rule_id)
                continue
            result.append({
                "rule_id": rule_id,
                "description": info["description"],
                "asked_at": datetime.fromtimestamp(info["asked_at"]).isoformat(timespec="seconds"),
                "age_sec": round(age),
            })
        for rule_id in expired:
            self._pending_confirmations.pop(rule_id, None)
        return result

    def get_fire_history(self, count: int = 20) -> List[dict]:
        """Get recent rule fire history for dashboard."""
        return self._fire_history[-count:]

    def rules_summary_for_llm(self, *, max_rules: Optional[int] = None, unique_only: bool = False) -> str:
        """Format rules summary for LLM context."""
        if not self._rules:
            return "No rules defined yet."
        rules = list(self._rules)
        if unique_only:
            seen = set()
            unique_rules = []
            for rule in rules:
                fingerprint = (
                    str(rule.get("description", "")).strip().lower(),
                    str(rule.get("trigger", {}).get("type", "")).strip().lower(),
                    str(rule.get("action", {}).get("command", "")).strip().lower(),
                    str(rule.get("permission", "ask")).strip().lower(),
                )
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                unique_rules.append(rule)
            rules = unique_rules
        if max_rules is not None:
            rules = rules[:max_rules]
        lines = []
        for rule in rules:
            status = "active" if rule.get("active") else "disabled"
            lines.append(
                f"- [{rule['id']}] {rule.get('description', 'no description')} "
                f"(permission={rule.get('permission', 'ask')}, {status})"
            )
        skipped = len(self._rules) - len(rules)
        if skipped > 0:
            lines.append(f"- ... {skipped} additional rules omitted for brevity")
        return "\n".join(lines)

    def _trigger_matches(self, trigger: dict, event: Event) -> bool:
        """Check if an event matches a rule's trigger."""
        trigger_type = trigger.get("type")
        if not trigger_type:
            return False

        # Type must match
        if trigger_type != event.type:
            return False

        # Check specific action match
        if "action" in trigger:
            if event.data.get("action") != trigger["action"]:
                # Also check the "to" field for action_changed events
                if event.data.get("to") != trigger["action"]:
                    return False

        # Check action_in (matches any in a list)
        if "action_in" in trigger:
            action = event.data.get("action") or event.data.get("to")
            if action not in trigger["action_in"]:
                return False

        # Check speaker match
        if "who" in trigger:
            if event.data.get("who") != trigger["who"]:
                return False

        return True

    def _conditions_met(self, conditions: list, state: dict) -> bool:
        """Check if all conditions are met against the world state."""
        for cond in conditions:
            if not isinstance(cond, dict):
                continue
            for key, expected in cond.items():
                if key == "time_of_day":
                    if isinstance(expected, list):
                        if _time_of_day() not in expected:
                            return False
                    elif isinstance(expected, str):
                        if _time_of_day() != expected:
                            return False

                elif key == "people_present_any":
                    if expected and not state.get("people_present"):
                        return False

                elif key == "people_present":
                    present = set(state.get("people_present", {}).keys())
                    if not set(expected).issubset(present):
                        return False

                elif key == "people_present_count":
                    if len(state.get("people_present", {})) != expected:
                        return False

                elif key == "people_present_min":
                    if len(state.get("people_present", {})) < expected:
                        return False

                elif key == "action_is":
                    if state.get("current_action") != expected:
                        return False

                elif key == "action_in":
                    if state.get("current_action") not in expected:
                        return False

                elif key == "action_confidence_min":
                    if state.get("action_confidence", 0) < expected:
                        return False

                elif key == "speaker_is_enrolled":
                    # Check if any present person is enrolled
                    has_enrolled = any(
                        info.get("enrolled")
                        for info in state.get("people_present", {}).values()
                    )
                    if expected and not has_enrolled:
                        return False

        return True
