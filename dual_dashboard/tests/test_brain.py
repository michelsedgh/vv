"""
Brain Decision System — Comprehensive Tests

Tests realistic scenarios that a user would encounter in daily life.
Each test simulates events flowing through the system and verifies
the correct decisions are made.

Usage:
    cd /home/michel/Documents/Voice
    python -m pytest dual_dashboard/tests/test_brain.py -v
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dual_dashboard.brain.event_bus import Event, EventBus
from dual_dashboard.brain.world_state import WorldState, _time_of_day
from dual_dashboard.brain.rule_engine import RuleEngine
from dual_dashboard.brain.llm_client import LLMClient
from dual_dashboard.brain.executor import ActionExecutor
from dual_dashboard.brain.decision_loop import DecisionSystem
from dual_dashboard.brain.dashboard_stream import DashboardStreamHub


# ─── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def world():
    return WorldState()


@pytest.fixture
def tmp_rules_file(tmp_path):
    rules_path = tmp_path / "test_rules.yaml"
    return str(rules_path)


@pytest.fixture
def rule_engine(world, tmp_rules_file):
    return RuleEngine(tmp_rules_file, world)


@pytest.fixture
def executor():
    return ActionExecutor()


# ─── Event Bus Tests ───────────────────────────────────────────────

class TestEventBus:
    """Layer 1: Event Bus tests."""

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self, bus):
        """Events published to the bus are received by subscribers."""
        queue = bus.subscribe()
        event = Event(type="speaker_active", data={"who": "michel", "enrolled": True})
        await bus.publish(event)

        received = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert received.type == "speaker_active"
        assert received.data["who"] == "michel"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus):
        """Multiple subscribers each get a copy of the event."""
        q1 = bus.subscribe()
        q2 = bus.subscribe()
        event = Event(type="action_detected", data={"action": "Walk"})
        await bus.publish(event)

        r1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        r2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert r1.type == r2.type == "action_detected"

    @pytest.mark.asyncio
    async def test_queue_overflow_drops_oldest(self, bus):
        """When a queue is full, oldest events are dropped."""
        q = bus.subscribe(maxsize=3)
        for i in range(5):
            await bus.publish(Event(type="test", data={"i": i}))

        events = []
        while not q.empty():
            events.append(await q.get())
        # Should have the 3 most recent
        assert len(events) == 3
        assert events[-1].data["i"] == 4

    @pytest.mark.asyncio
    async def test_event_history(self, bus):
        """Bus maintains event history."""
        for i in range(5):
            await bus.publish(Event(type="test", data={"i": i}))

        history = bus.recent_events(count=3)
        assert len(history) == 3
        assert history[-1]["data"]["i"] == 4

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus):
        """Unsubscribed queues no longer receive events."""
        q = bus.subscribe()
        bus.unsubscribe(q)
        await bus.publish(Event(type="test"))
        assert q.empty()

    @pytest.mark.asyncio
    async def test_event_serialization(self):
        """Events can be serialized to dict."""
        event = Event(type="speaker_active", data={"who": "michel"})
        d = event.to_dict()
        assert d["type"] == "speaker_active"
        assert d["data"]["who"] == "michel"
        assert "event_id" in d
        assert "iso_time" in d


class TestDashboardStreaming:
    """Dashboard stream and callback payload tests."""

    @pytest.mark.asyncio
    async def test_dashboard_stream_hub_broadcasts(self):
        hub = DashboardStreamHub()
        queue = hub.subscribe()
        payload = {"type": "trace", "value": 1}

        await hub.publish(payload)
        received = await asyncio.wait_for(queue.get(), timeout=1.0)

        assert received == payload

    @pytest.mark.asyncio
    async def test_decision_system_emits_trace_callback(self, tmp_path):
        bus = EventBus()
        system = DecisionSystem(
            bus,
            rules_path=str(tmp_path / "rules.yaml"),
            lm_studio_url="http://fake:1234",
        )
        system.llm._available = False
        pushed = []

        async def callback(payload):
            pushed.append(payload)

        system.set_dashboard_callback(callback)
        await system._process_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))

        assert pushed
        assert pushed[-1]["type"] == "trace"
        assert pushed[-1]["trace"]["event"]["type"] == "speaker_active"
        assert pushed[-1]["world"]["people_count"] == 1


# ─── World State Tests ─────────────────────────────────────────────

class TestWorldState:
    """Layer 2: World State Tracker tests."""

    def test_speaker_active_creates_presence(self, world):
        """Speaker becoming active adds them to people_present."""
        event = Event(type="speaker_active", data={"who": "michel", "enrolled": True, "confidence": 0.95})
        world.update_from_event(event)
        
        snap = world.snapshot()
        assert "michel" in snap["people_present"]
        assert snap["people_present"]["michel"]["enrolled"] is True
        assert snap["people_present"]["michel"]["speaking"] is True

    def test_speaker_silent_updates_speaking(self, world):
        """Speaker going silent updates their speaking status."""
        world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        world.update_from_event(Event(type="speaker_silent", data={"who": "michel"}))

        snap = world.snapshot()
        assert snap["people_present"]["michel"]["speaking"] is False

    def test_new_speaker_generates_person_entered(self, world):
        """A new speaker generates a person_entered derived event."""
        event = Event(type="speaker_active", data={"who": "alice", "enrolled": False})
        derived = world.update_from_event(event)

        assert any(d["type"] == "person_entered" for d in derived)

    def test_action_debounce(self, world):
        """Actions must be stable for N seconds before committing."""
        # First detection doesn't immediately change
        world.update_from_event(Event(type="action_detected", data={"action": "Eat.Attable", "confidence": 0.85}))
        snap = world.snapshot()
        assert snap["current_action"] is None  # Not committed yet

        # After debounce period, it should commit
        world._action_debounce_sec = 0  # Disable for test
        world._action_debounce_target = None  # Reset
        world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.9}))
        world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.9}))
        
        snap = world.snapshot()
        assert snap["current_action"] == "Walk"

    def test_action_changed_generates_event(self, world):
        """When action changes (after debounce), a derived event is generated."""
        world._action_debounce_sec = 0
        # First, commit Walk as the current action
        world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.9}))
        world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.9}))
        assert world.state["current_action"] == "Walk"

        # Now change to Eat.Attable — should produce a derived event
        derived = world.update_from_event(Event(type="action_detected", data={"action": "Eat.Attable", "confidence": 0.85}))
        # First call with new action sets debounce target; second triggers the change
        derived2 = world.update_from_event(Event(type="action_detected", data={"action": "Eat.Attable", "confidence": 0.85}))
        all_derived = derived + derived2

        action_changed_events = [d for d in all_derived if d["type"] == "action_changed"]
        assert len(action_changed_events) == 1
        assert action_changed_events[0]["data"]["to"] == "Eat.Attable"

    def test_speech_tracking(self, world):
        """Speech events are tracked in world state."""
        world.update_from_event(Event(
            type="speech_text",
            data={"who": "michel", "text": "turn on the lights"},
        ))

        snap = world.snapshot()
        assert snap["last_speech"]["michel"]["text"] == "turn on the lights"
        assert len(snap["recent_speech"]) == 1

    def test_person_left(self, world):
        """person_left event removes from people_present."""
        world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        world.update_from_event(Event(type="person_left", data={"who": "michel"}))

        snap = world.snapshot()
        assert "michel" not in snap["people_present"]

    def test_time_of_day(self):
        """Time of day classification works correctly."""
        assert _time_of_day(8) == "morning"
        assert _time_of_day(14) == "afternoon"
        assert _time_of_day(19) == "evening"
        assert _time_of_day(23) == "night"
        assert _time_of_day(3) == "night"

    def test_llm_context_format(self, world):
        """LLM context is formatted as a readable string."""
        world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        world._action_debounce_sec = 0
        # Need two calls for debounce: first sets target, second commits
        world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.7}))
        world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.7}))

        ctx = world.get_llm_context()
        assert "michel" in ctx
        assert "Walk" in ctx


# ─── Rule Engine Tests ─────────────────────────────────────────────

class TestRuleEngine:
    """Layer 3: Rule Engine tests."""

    def _make_engine(self, world, tmp_path, rules=None):
        """Helper to create a rule engine with specific rules."""
        path = str(tmp_path / "rules.yaml")
        engine = RuleEngine(path, world)
        if rules is not None:
            engine._rules = rules
            engine._save_rules()
        return engine

    def test_default_rules_created(self, world, tmp_path):
        """Default rules are created when no rules file exists."""
        engine = self._make_engine(world, tmp_path)
        assert len(engine.rules) > 0

    def test_trigger_matching(self, world, tmp_path):
        """Rules trigger on matching event types."""
        rules = [{
            "id": "test_walk",
            "description": "Test Walk detection",
            "trigger": {
                "type": "action_changed",
                "action": "Walk",
                "conditions": [],
            },
            "action": {"type": "notify", "message": "Walking!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)

        # Should match
        event = Event(type="action_changed", data={"to": "Walk"})
        results = engine.evaluate(event)
        assert len(results) == 1
        assert results[0]["decision"] == "fire_auto"

    def test_trigger_no_match(self, world, tmp_path):
        """Rules don't trigger on non-matching events."""
        rules = [{
            "id": "test_walk",
            "description": "Test Walk detection",
            "trigger": {
                "type": "action_changed",
                "action": "Walk",
                "conditions": [],
            },
            "action": {"type": "notify", "message": "Walking!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)

        # Different action — should not match
        event = Event(type="action_changed", data={"to": "Eat.Attable"})
        results = engine.evaluate(event)
        assert len(results) == 0

    def test_conditions_time_of_day(self, world, tmp_path):
        """Time-of-day conditions are evaluated correctly."""
        rules = [{
            "id": "evening_only",
            "description": "Only fires in evening",
            "trigger": {
                "type": "action_changed",
                "action": "WatchTV",
                "conditions": [
                    {"time_of_day": ["evening", "night"]},
                ],
            },
            "action": {"type": "notify", "message": "TV time!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)
        event = Event(type="action_changed", data={"to": "WatchTV"})

        # Mock time of day
        with patch("dual_dashboard.brain.rule_engine._time_of_day", return_value="evening"):
            results = engine.evaluate(event)
            assert len(results) == 1
            assert results[0]["decision"] == "fire_auto"

        with patch("dual_dashboard.brain.rule_engine._time_of_day", return_value="morning"):
            results = engine.evaluate(event)
            has_fire = any(r["decision"].startswith("fire_") for r in results)
            assert not has_fire

    def test_conditions_people_present(self, world, tmp_path):
        """People-present conditions check world state."""
        rules = [{
            "id": "needs_people",
            "description": "Requires people present",
            "trigger": {
                "type": "action_changed",
                "action": "Eat.Attable",
                "conditions": [
                    {"people_present_any": True},
                ],
            },
            "action": {"type": "notify", "message": "Dinner!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)
        event = Event(type="action_changed", data={"to": "Eat.Attable"})

        # No people — should not fire
        results = engine.evaluate(event)
        has_fire = any(r["decision"].startswith("fire_") for r in results)
        assert not has_fire

        # Add a person
        world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        results = engine.evaluate(event)
        fired = [r for r in results if r["decision"].startswith("fire_")]
        assert len(fired) == 1

    def test_cooldown(self, world, tmp_path):
        """Rules respect cooldown periods."""
        rules = [{
            "id": "test_cooldown",
            "description": "Has 60s cooldown",
            "trigger": {
                "type": "action_changed",
                "action": "Walk",
                "conditions": [],
            },
            "action": {"type": "notify", "message": "Walking!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 60,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)
        event = Event(type="action_changed", data={"to": "Walk"})

        # First fire should work
        results = engine.evaluate(event)
        assert any(r["decision"] == "fire_auto" for r in results)

        # Second fire should be blocked by cooldown
        results = engine.evaluate(event)
        assert any(r["decision"] == "cooldown" for r in results)

    def test_disabled_rule(self, world, tmp_path):
        """Disabled rules are skipped."""
        rules = [{
            "id": "disabled_rule",
            "description": "Disabled",
            "trigger": {
                "type": "action_changed",
                "action": "Walk",
                "conditions": [],
            },
            "action": {"type": "notify", "message": "Walking!"},
            "permission": "auto",
            "active": False,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)
        event = Event(type="action_changed", data={"to": "Walk"})
        results = engine.evaluate(event)
        assert len(results) == 0

    def test_permission_models(self, world, tmp_path):
        """Different permission models produce correct decisions."""
        for perm in ["auto", "notify", "ask", "suggest"]:
            rules = [{
                "id": f"test_{perm}",
                "description": f"Test {perm}",
                "trigger": {"type": "test_event", "conditions": []},
                "action": {"type": "notify", "message": "test"},
                "permission": perm,
                "active": True,
                "cooldown_sec": 0,
                "expires": None,
            }]
            engine = self._make_engine(world, tmp_path, rules)
            results = engine.evaluate(Event(type="test_event"))
            assert results[0]["decision"] == f"fire_{perm}"

    def test_add_rule(self, world, tmp_path):
        """Adding a new rule persists it."""
        engine = self._make_engine(world, tmp_path, [])
        new_rule = {
            "id": "new_rule",
            "description": "Newly added rule",
            "trigger": {"type": "test_event", "conditions": []},
            "action": {"type": "notify", "message": "new!"},
            "permission": "auto",
        }
        added = engine.add_rule(new_rule)
        assert added["id"] == "new_rule"
        assert len(engine.rules) == 1

        # Reload and verify persistence
        engine2 = RuleEngine(str(tmp_path / "rules.yaml"), world)
        assert len(engine2.rules) == 1

    def test_toggle_rule(self, world, tmp_path):
        """Toggling a rule flips its active state."""
        rules = [{
            "id": "toggle_me",
            "description": "Toggle test",
            "trigger": {"type": "test", "conditions": []},
            "action": {"type": "notify", "message": "test"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)
        toggled = engine.toggle_rule("toggle_me")
        assert toggled["active"] is False

        toggled = engine.toggle_rule("toggle_me")
        assert toggled["active"] is True

    def test_action_in_trigger(self, world, tmp_path):
        """action_in trigger matches any action in a list."""
        rules = [{
            "id": "cooking_any",
            "description": "Any cooking activity",
            "trigger": {
                "type": "action_changed",
                "action_in": ["Cook.Stir", "Cook.Cut", "Cook.Usestove"],
                "conditions": [],
            },
            "action": {"type": "notify", "message": "Cooking!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)

        # Should match Cook.Stir
        results = engine.evaluate(Event(type="action_changed", data={"to": "Cook.Stir"}))
        assert any(r["decision"] == "fire_auto" for r in results)

        # Should NOT match Walk
        results = engine.evaluate(Event(type="action_changed", data={"to": "Walk"}))
        assert not any(r["decision"].startswith("fire_") for r in results)

    def test_one_time_rule(self, world, tmp_path):
        """One-time rules only fire once."""
        rules = [{
            "id": "once_only",
            "description": "One-time rule",
            "trigger": {"type": "test_event", "conditions": []},
            "action": {"type": "notify", "message": "once!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": "once",
        }]
        engine = self._make_engine(world, tmp_path, rules)

        # First fire
        results = engine.evaluate(Event(type="test_event"))
        assert any(r["decision"] == "fire_auto" for r in results)

        # Second should not fire (expires = "fired")
        results = engine.evaluate(Event(type="test_event"))
        assert not any(r["decision"].startswith("fire_") for r in results)

    def test_pending_confirmations(self, world, tmp_path):
        """Ask-permission rules create pending confirmations."""
        engine = self._make_engine(world, tmp_path, [])
        action = {"type": "smart_home", "command": "lights_off"}
        engine.add_pending_confirmation("rule_1", action, "Turn off lights")

        pending = engine.get_pending_confirmations()
        assert len(pending) == 1
        assert pending[0]["rule_id"] == "rule_1"

        # Confirm it
        result = engine.confirm_pending("rule_1", True)
        assert result["approved"] is True
        assert result["action"] == action

    def test_specific_person_condition(self, world, tmp_path):
        """Rules can require specific people to be present."""
        rules = [{
            "id": "michel_only",
            "description": "Only when Michel is here",
            "trigger": {
                "type": "test_event",
                "conditions": [
                    {"people_present": ["michel"]},
                ],
            },
            "action": {"type": "notify", "message": "Hi Michel!"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        engine = self._make_engine(world, tmp_path, rules)

        # No Michel — should not fire
        results = engine.evaluate(Event(type="test_event"))
        has_fire = any(r["decision"].startswith("fire_") for r in results)
        assert not has_fire

        # Add Michel
        world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        results = engine.evaluate(Event(type="test_event"))
        fired = [r for r in results if r["decision"].startswith("fire_")]
        assert len(fired) == 1


# ─── Executor Tests ────────────────────────────────────────────────

class TestActionExecutor:
    """Layer 5: Action Executor tests."""

    @pytest.mark.asyncio
    async def test_smart_home_stub(self, executor):
        """Smart home actions are simulated when HA not connected."""
        result = await executor.execute(
            {"type": "smart_home", "command": "lights_off", "params": {}},
            rule_id="test",
        )
        assert result["status"] == "simulated"
        assert "STUB" in result["details"]

    @pytest.mark.asyncio
    async def test_tts_stub(self, executor):
        """TTS actions store messages for dashboard display."""
        result = await executor.execute(
            {"type": "tts", "message": "Hello, Michel!"},
            rule_id="test",
        )
        assert result["status"] == "simulated"
        assert len(executor.get_pending_speech()) == 1

    @pytest.mark.asyncio
    async def test_notification(self, executor):
        """Notifications are sent via callback."""
        received = []
        async def mock_callback(data):
            received.append(data)

        executor.set_notification_callback(mock_callback)
        await executor.execute(
            {"type": "notify", "message": "Test notification"},
            rule_id="test",
        )
        assert len(received) == 1
        assert received[0]["message"] == "Test notification"

    @pytest.mark.asyncio
    async def test_action_log(self, executor):
        """All actions are logged."""
        await executor.execute({"type": "smart_home", "command": "test"})
        await executor.execute({"type": "tts", "message": "test"})
        await executor.execute({"type": "notify", "message": "test"})

        log = executor.get_action_log()
        assert len(log) == 3

    @pytest.mark.asyncio
    async def test_stats(self, executor):
        """Stats track execution counts."""
        await executor.execute({"type": "smart_home", "command": "test"})
        await executor.execute({"type": "unknown_type"})

        stats = executor.stats()
        assert stats["total_actions"] == 2
        assert stats["executed"] >= 1


# ─── Scenario Tests (Integration) ─────────────────────────────────

class TestScenarios:
    """
    End-to-end scenario tests simulating real-life usage.
    These test multiple layers working together.
    """

    def _make_system(self, tmp_path, rules=None):
        """Create a DecisionSystem with test rules."""
        bus = EventBus()
        rules_path = str(tmp_path / "test_rules.yaml")
        system = DecisionSystem(bus, rules_path=rules_path, lm_studio_url="http://fake:1234")
        system.llm._available = False  # Disable LLM for most tests

        if rules is not None:
            system.rules._rules = rules
            system.rules._save_rules()

        return system, bus

    @pytest.mark.asyncio
    async def test_scenario_dinner_time(self, tmp_path):
        """
        Scenario: Michel comes home, starts eating dinner.
        Expected: Dinner lights should dim.
        """
        rules = [{
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
            "action": {"type": "smart_home", "command": "dim_lights", "params": {"brightness": 40}},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        system, bus = self._make_system(tmp_path, rules)

        # Michel enters
        system.world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))

        # Action changes to eating
        event = Event(type="action_changed", data={"to": "Eat.Attable", "confidence": 0.85})

        with patch("dual_dashboard.brain.rule_engine._time_of_day", return_value="evening"):
            results = system.rules.evaluate(event)

        fired = [r for r in results if r["decision"].startswith("fire_")]
        assert len(fired) == 1
        assert fired[0]["action"]["command"] == "dim_lights"

    @pytest.mark.asyncio
    async def test_scenario_everyone_leaves(self, tmp_path):
        """
        Scenario: Last person leaves the house.
        Expected: All lights turn off.
        """
        rules = [{
            "id": "leave_lights",
            "description": "Turn off lights when everyone leaves",
            "trigger": {
                "type": "person_left",
                "conditions": [{"people_present_count": 0}],
            },
            "action": {"type": "smart_home", "command": "lights_off"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        system, bus = self._make_system(tmp_path, rules)

        # Michel and Alice are present
        system.world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        system.world.update_from_event(Event(type="speaker_active", data={"who": "alice", "enrolled": False}))

        # Alice leaves — Michel still here
        system.world.update_from_event(Event(type="person_left", data={"who": "alice"}))
        results = system.rules.evaluate(Event(type="person_left", data={"who": "alice"}))
        fired = [r for r in results if r["decision"].startswith("fire_")]
        assert len(fired) == 0  # Michel still present

        # Michel leaves
        system.world.update_from_event(Event(type="person_left", data={"who": "michel"}))
        results = system.rules.evaluate(Event(type="person_left", data={"who": "michel"}))
        fired = [r for r in results if r["decision"].startswith("fire_")]
        assert len(fired) == 1
        assert fired[0]["action"]["command"] == "lights_off"

    @pytest.mark.asyncio
    async def test_scenario_cooking_ventilation(self, tmp_path):
        """
        Scenario: User starts cooking — multiple cooking actions should trigger ventilation.
        """
        rules = [{
            "id": "cooking_vent",
            "description": "Ventilation when cooking",
            "trigger": {
                "type": "action_changed",
                "action_in": ["Cook.Stir", "Cook.Usestove", "Cook.Cut"],
                "conditions": [],
            },
            "action": {"type": "smart_home", "command": "ventilation_on"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        system, bus = self._make_system(tmp_path, rules)

        # Cook.Stir detected
        results = system.rules.evaluate(Event(type="action_changed", data={"to": "Cook.Stir"}))
        assert any(r["decision"] == "fire_auto" for r in results)

        # Cook.Cut detected  
        results = system.rules.evaluate(Event(type="action_changed", data={"to": "Cook.Cut"}))
        assert any(r["decision"] == "fire_auto" for r in results)

        # Walk detected — should NOT trigger
        results = system.rules.evaluate(Event(type="action_changed", data={"to": "Walk"}))
        assert not any(r["decision"].startswith("fire_") for r in results)

    @pytest.mark.asyncio
    async def test_scenario_rule_cooldown_prevents_spam(self, tmp_path):
        """
        Scenario: Action flaps between Eat.Attable and Sitdown.
        Expected: Rule only fires once within cooldown period.
        """
        rules = [{
            "id": "dinner_lights",
            "description": "Dim lights for dinner",
            "trigger": {
                "type": "action_changed",
                "action": "Eat.Attable",
                "conditions": [],
            },
            "action": {"type": "smart_home", "command": "dim_lights"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 300,
            "expires": None,
        }]
        system, bus = self._make_system(tmp_path, rules)

        # First detection fires
        results = system.rules.evaluate(Event(type="action_changed", data={"to": "Eat.Attable"}))
        assert any(r["decision"] == "fire_auto" for r in results)

        # Flap to Sitdown and back — should hit cooldown
        results = system.rules.evaluate(Event(type="action_changed", data={"to": "Eat.Attable"}))
        assert any(r["decision"] == "cooldown" for r in results)

    @pytest.mark.asyncio
    async def test_scenario_night_vs_day(self, tmp_path):
        """
        Scenario: TV ambiance only activates in the evening.
        """
        rules = [{
            "id": "tv_ambiance",
            "description": "TV ambiance in evening",
            "trigger": {
                "type": "action_changed",
                "action": "WatchTV",
                "conditions": [{"time_of_day": ["evening", "night"]}],
            },
            "action": {"type": "smart_home", "command": "tv_scene"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        system, bus = self._make_system(tmp_path, rules)
        event = Event(type="action_changed", data={"to": "WatchTV"})

        # 3pm — afternoon, should NOT fire
        with patch("dual_dashboard.brain.rule_engine._time_of_day", return_value="afternoon"):
            results = system.rules.evaluate(event)
            has_fire = any(r["decision"].startswith("fire_") for r in results)
            assert not has_fire

        # 8pm — evening, SHOULD fire
        with patch("dual_dashboard.brain.rule_engine._time_of_day", return_value="evening"):
            results = system.rules.evaluate(event)
            assert any(r["decision"] == "fire_auto" for r in results)

    @pytest.mark.asyncio
    async def test_scenario_ask_permission(self, tmp_path):
        """
        Scenario: Rule with 'ask' permission waits for user confirmation.
        """
        rules = [{
            "id": "roomba",
            "description": "Start roomba when everyone leaves",
            "trigger": {
                "type": "person_left",
                "conditions": [{"people_present_count": 0}],
            },
            "action": {"type": "smart_home", "command": "start_roomba"},
            "permission": "ask",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        system, bus = self._make_system(tmp_path, rules)

        # Fire the rule
        results = system.rules.evaluate(Event(type="person_left"))
        assert any(r["decision"] == "fire_ask" for r in results)

        # Simulate pending confirmation
        system.rules.add_pending_confirmation("roomba", {"type": "smart_home", "command": "start_roomba"}, "Start roomba")
        pending = system.rules.get_pending_confirmations()
        assert len(pending) == 1

        # User says yes
        result = system.rules.confirm_pending("roomba", True)
        assert result["approved"] is True
        assert result["action"]["command"] == "start_roomba"

    @pytest.mark.asyncio
    async def test_scenario_suggest_mode(self, tmp_path):
        """
        Scenario: Rule with 'suggest' permission just notifies, doesn't execute.
        """
        rules = [{
            "id": "suggest_lights",
            "description": "Suggest table lighting for dinner",
            "trigger": {
                "type": "action_changed",
                "action": "Eat.Attable",
                "conditions": [],
            },
            "action": {"type": "smart_home", "command": "dim_lights"},
            "permission": "suggest",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]
        system, bus = self._make_system(tmp_path, rules)

        results = system.rules.evaluate(Event(type="action_changed", data={"to": "Eat.Attable"}))
        assert any(r["decision"] == "fire_suggest" for r in results)

    @pytest.mark.asyncio
    async def test_scenario_multiple_rules_same_event(self, tmp_path):
        """
        Scenario: Multiple rules can match the same event.
        Expected: All matching rules fire independently.
        """
        rules = [
            {
                "id": "rule_a",
                "description": "Rule A",
                "trigger": {"type": "test_event", "conditions": []},
                "action": {"type": "notify", "message": "A"},
                "permission": "auto",
                "active": True,
                "cooldown_sec": 0,
                "expires": None,
            },
            {
                "id": "rule_b",
                "description": "Rule B",
                "trigger": {"type": "test_event", "conditions": []},
                "action": {"type": "notify", "message": "B"},
                "permission": "auto",
                "active": True,
                "cooldown_sec": 0,
                "expires": None,
            },
        ]
        system, bus = self._make_system(tmp_path, rules)

        results = system.rules.evaluate(Event(type="test_event"))
        fired = [r for r in results if r["decision"].startswith("fire_")]
        assert len(fired) == 2

    @pytest.mark.asyncio
    async def test_scenario_world_state_consistency(self, tmp_path):
        """
        Scenario: World state stays consistent through many events.
        """
        system, bus = self._make_system(tmp_path, [])

        # Rapid fire events
        system.world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        system.world.update_from_event(Event(type="speaker_active", data={"who": "alice", "enrolled": False}))
        system.world._action_debounce_sec = 0
        # Two same-action events needed: first sets target, second commits
        system.world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.9}))
        system.world.update_from_event(Event(type="action_detected", data={"action": "Walk", "confidence": 0.9}))
        # Transition to Eat.Attable
        system.world.update_from_event(Event(type="action_detected", data={"action": "Eat.Attable", "confidence": 0.8}))
        system.world.update_from_event(Event(type="action_detected", data={"action": "Eat.Attable", "confidence": 0.8}))
        system.world.update_from_event(Event(type="speech_text", data={"who": "michel", "text": "hello"}))
        system.world.update_from_event(Event(type="speaker_silent", data={"who": "michel"}))

        snap = system.world.snapshot()
        assert snap["people_count"] == 2
        assert snap["current_action"] == "Eat.Attable"
        assert snap["people_present"]["michel"]["speaking"] is False
        assert snap["people_present"]["alice"]["speaking"] is True
        assert snap["last_speech"]["michel"]["text"] == "hello"


# ─── LLM Client Tests ─────────────────────────────────────────────

class TestLLMClient:
    """Layer 4: LLM Client tests (with mocked responses)."""

    @pytest.mark.asyncio
    async def test_json_parsing_clean(self):
        """Clean JSON responses are parsed correctly."""
        client = LLMClient(base_url="http://fake:1234")
        result = client._parse_json('{"intent": "create_rule", "confidence": 0.95}')
        assert result["intent"] == "create_rule"

    @pytest.mark.asyncio
    async def test_json_parsing_markdown(self):
        """JSON wrapped in markdown code blocks is parsed."""
        client = LLMClient(base_url="http://fake:1234")
        result = client._parse_json('```json\n{"intent": "create_rule"}\n```')
        assert result["intent"] == "create_rule"

    @pytest.mark.asyncio
    async def test_json_parsing_with_text(self):
        """JSON embedded in explanatory text is extracted."""
        client = LLMClient(base_url="http://fake:1234")
        result = client._parse_json('Here is the result: {"intent": "question", "confidence": 0.8} Hope that helps!')
        assert result["intent"] == "question"

    @pytest.mark.asyncio
    async def test_json_parsing_invalid(self):
        """Invalid JSON returns None."""
        client = LLMClient(base_url="http://fake:1234")
        result = client._parse_json("this is not json at all")
        assert result is None

    @pytest.mark.asyncio
    async def test_stats(self):
        """Stats are tracked correctly."""
        client = LLMClient(base_url="http://fake:1234")
        stats = client.stats()
        assert stats["call_count"] == 0
        assert stats["available"] is False


# ─── Edge Case Tests ───────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases identified in the architecture plan."""

    def test_edge_stale_presence_expires(self):
        """People who haven't been seen recently are removed."""
        world = WorldState()
        # Add someone with a very old timestamp
        world.state["people_present"]["ghost"] = {
            "enrolled": False,
            "since": "2020-01-01",
            "speaking": False,
            "last_seen": time.time() - 999999,  # Very old
        }

        snap = world.snapshot()
        assert "ghost" not in snap["people_present"]

    def test_edge_empty_event_data(self):
        """Events with missing data fields don't crash."""
        world = WorldState()
        # These shouldn't crash
        world.update_from_event(Event(type="speaker_active", data={}))
        world.update_from_event(Event(type="action_detected", data={}))
        world.update_from_event(Event(type="speech_text", data={}))

    def test_edge_rapid_action_changes(self):
        """Rapid action changes don't destabilize the state."""
        world = WorldState()
        world._action_debounce_sec = 2.0  # 2 second debounce

        actions = ["Walk", "Sitdown", "Walk", "Eat.Attable", "Walk", "Sitdown"]
        for action in actions:
            world.update_from_event(Event(
                type="action_detected",
                data={"action": action, "confidence": 0.7},
            ))

        # State should NOT have committed any action (too rapid)
        snap = world.snapshot()
        # The rapid changes mean debounce hasn't settled
        assert snap["action_stable"] is False or snap["current_action"] is None

    def test_edge_concurrent_speakers(self):
        """Multiple speakers active simultaneously."""
        world = WorldState()
        world.update_from_event(Event(type="speaker_active", data={"who": "michel", "enrolled": True}))
        world.update_from_event(Event(type="speaker_active", data={"who": "alice", "enrolled": False}))
        world.update_from_event(Event(type="speaker_active", data={"who": "bob", "enrolled": False}))

        snap = world.snapshot()
        assert snap["people_count"] == 3
        assert all(
            snap["people_present"][name]["speaking"] is True
            for name in ["michel", "alice", "bob"]
        )

    @pytest.mark.asyncio
    async def test_edge_rule_with_no_conditions(self, tmp_path):
        """Rules with empty conditions still work."""
        world = WorldState()
        engine = RuleEngine(str(tmp_path / "rules.yaml"), world)
        engine._rules = [{
            "id": "no_cond",
            "description": "No conditions",
            "trigger": {"type": "test", "conditions": []},
            "action": {"type": "notify", "message": "fired"},
            "permission": "auto",
            "active": True,
            "cooldown_sec": 0,
            "expires": None,
        }]

        results = engine.evaluate(Event(type="test"))
        assert any(r["decision"] == "fire_auto" for r in results)

    @pytest.mark.asyncio
    async def test_edge_rule_with_missing_fields(self, tmp_path):
        """Rules with missing optional fields still work."""
        world = WorldState()
        engine = RuleEngine(str(tmp_path / "rules.yaml"), world)
        engine._rules = [{
            "id": "minimal",
            "trigger": {"type": "test"},
            "action": {"type": "notify"},
            "active": True,
        }]

        results = engine.evaluate(Event(type="test"))
        # Should fire without crashing
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_edge_person_left_with_nobody_present(self, tmp_path):
        """person_left event when nobody is tracked doesn't crash."""
        world = WorldState()
        world.update_from_event(Event(type="person_left", data={"who": "nobody"}))
        snap = world.snapshot()
        assert snap["people_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
