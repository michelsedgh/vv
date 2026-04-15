"""
Decision Loop — The Main Orchestrator

Ties all 5 layers together:
    1. Subscribes to the Event Bus
    2. Updates World State from every event
    3. Evaluates Rules against events
    4. Calls LLM for speech-based commands and ambiguous situations
    5. Executes actions through the Executor

Also provides a comprehensive API for the Brain Dashboard
to visualize decision flow.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .event_bus import Event, EventBus
from .executor import ActionExecutor
from .llm_client import LLMClient
from .rule_engine import RuleEngine
from .world_state import WorldState

log = logging.getLogger("brain.decision")


class DecisionTrace:
    """A single decision trace showing how an event flowed through the system."""

    def __init__(self, event: Event):
        self.event = event
        self.start_time = time.time()
        self.layers: List[dict] = []
        self.final_decision: Optional[str] = None

    def add_layer(self, layer: str, status: str, details: str, data: Optional[dict] = None) -> None:
        self.layers.append({
            "layer": layer,
            "status": status,
            "details": details,
            "data": data or {},
            "timestamp": time.time(),
            "elapsed_ms": round((time.time() - self.start_time) * 1000, 1),
        })

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "start_time": self.start_time,
            "total_ms": round((time.time() - self.start_time) * 1000, 1),
            "layers": self.layers,
            "final_decision": self.final_decision,
        }


class DecisionSystem:
    """
    Main decision system orchestrator.
    
    Subscribes to the event bus, processes events through all layers,
    and provides dashboard API endpoints.
    """

    def __init__(
        self,
        bus: EventBus,
        rules_path: str = "",
        lm_studio_url: str = "http://192.168.1.198:1234",
    ):
        self.bus = bus
        self.world = WorldState()
        self.executor = ActionExecutor()
        self.llm = LLMClient(base_url=lm_studio_url)

        # Default rules path
        if not rules_path:
            rules_path = str(Path(__file__).parent.parent / "rules.yaml")
        self.rules = RuleEngine(rules_path, self.world)

        # Decision trace history (for dashboard)
        self._traces: List[dict] = []
        self._max_traces = 200

        # System status
        self._running = False
        self._events_processed = 0
        self._rules_fired = 0
        self._llm_calls = 0

        # Dashboard notification callback
        self._dashboard_callback: Optional[Any] = None

        # Queue for the main loop
        self._queue: Optional[asyncio.Queue] = None

    async def start(self) -> None:
        """Start the decision loop."""
        if self._running:
            return

        self._running = True
        self._queue = self.bus.subscribe()

        # Check LLM health
        llm_ok = await self.llm.check_health()
        if llm_ok:
            log.info("LLM connected: %s (model: %s)", self.llm.base_url, self.llm.model)
        else:
            log.warning("LLM not available at %s — rule creation via speech will be disabled", self.llm.base_url)

        # Set up executor notification callback
        async def notify_dashboard(data: dict) -> None:
            if self._dashboard_callback:
                await self._dashboard_callback(data)

        self.executor.set_notification_callback(notify_dashboard)

        log.info("Decision system started")

        # Start main loop as background task
        asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """Stop the decision loop."""
        self._running = False
        if self._queue:
            self.bus.unsubscribe(self._queue)
            self._queue = None
        await self.llm.close()
        log.info("Decision system stopped")

    async def _run(self) -> None:
        """Main event processing loop."""
        while self._running and self._queue:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception:
                if not self._running:
                    break
                continue

            try:
                await self._process_event(event)
            except Exception as exc:
                log.error("Error processing event %s: %s", event.type, exc, exc_info=True)

    async def _process_event(self, event: Event) -> None:
        """Process a single event through all layers."""
        trace = DecisionTrace(event)
        self._events_processed += 1

        # ── Layer 1: Event Bus (already done — event is here) ──
        trace.add_layer("event_bus", "received", f"Event: {event.type}", event.to_dict())

        # ── Layer 2: World State ──
        derived_events = self.world.update_from_event(event)
        trace.add_layer(
            "world_state",
            "updated",
            f"State updated, {len(derived_events)} derived events",
            {"derived_events": [e.get("type") for e in derived_events]},
        )

        # Publish derived events (e.g., person_entered, action_changed)
        for de in derived_events:
            derived_event = Event(type=de["type"], data=de.get("data", {}))
            await self.bus.publish(derived_event)

        # ── Layer 3: Rule Engine ──
        rule_results = self.rules.evaluate(event)

        # Also evaluate derived events
        for de in derived_events:
            derived_event = Event(type=de["type"], data=de.get("data", {}))
            rule_results.extend(self.rules.evaluate(derived_event))

        if rule_results:
            trace.add_layer(
                "rule_engine",
                "evaluated",
                f"{len(rule_results)} rule(s) matched",
                {"results": rule_results},
            )
        else:
            trace.add_layer("rule_engine", "no_match", "No rules triggered")

        # ── Process rule results ──
        for result in rule_results:
            decision = result.get("decision", "")

            if decision.startswith("fire_"):
                permission = result.get("permission", "ask")
                action = result.get("action", {})
                rule_id = result.get("rule_id", "")
                rule_desc = result.get("rule_description", "")

                if permission == "auto":
                    exec_result = await self.executor.execute(action, rule_id, rule_desc)
                    trace.add_layer(
                        "executor",
                        "executed",
                        f"Auto-executed: {rule_desc}",
                        exec_result,
                    )
                    self._rules_fired += 1

                elif permission == "notify":
                    exec_result = await self.executor.execute(action, rule_id, rule_desc)
                    await self.executor.notify(f"✓ {rule_desc}")
                    trace.add_layer(
                        "executor",
                        "executed_notified",
                        f"Executed with notification: {rule_desc}",
                        exec_result,
                    )
                    self._rules_fired += 1

                elif permission == "ask":
                    self.rules.add_pending_confirmation(rule_id, action, rule_desc)
                    await self.executor.ask(f"Should I {rule_desc.lower()}?", rule_id)
                    trace.add_layer(
                        "executor",
                        "asked",
                        f"Asking user: {rule_desc}",
                        {"rule_id": rule_id},
                    )

                elif permission == "suggest":
                    await self.executor.notify(f"💡 Suggestion: {rule_desc}")
                    trace.add_layer(
                        "executor",
                        "suggested",
                        f"Suggestion sent: {rule_desc}",
                    )

            elif decision == "cooldown":
                trace.add_layer(
                    "rule_engine",
                    "cooldown",
                    f"Rule {result['rule_id']} in cooldown: {result.get('details', '')}",
                )

            elif decision == "conditions_not_met":
                trace.add_layer(
                    "rule_engine",
                    "conditions_unmet",
                    f"Rule {result['rule_id']} conditions not met",
                )

        # ── Layer 4: LLM (for speech events) ──
        if event.type == "speech_text" and self.llm._available:
            await self._handle_speech(event, trace)

        # ── Handle user responses to confirmation requests ──
        if event.type == "user_response":
            rule_id = event.data.get("rule_id", "")
            approved = event.data.get("answer", "").lower() in ("yes", "yeah", "yep", "go ahead", "do it", "sure", "ok")
            result = self.rules.confirm_pending(rule_id, approved)
            if result and result.get("approved") and result.get("action"):
                exec_result = await self.executor.execute(result["action"], rule_id)
                trace.add_layer("executor", "confirmed", f"User confirmed rule {rule_id}", exec_result)
                self._rules_fired += 1
            elif result:
                trace.add_layer("executor", "rejected", f"User rejected rule {rule_id}")

        # ── Finalize trace ──
        fired = any(r.get("decision", "").startswith("fire_") for r in rule_results)
        if fired:
            trace.final_decision = "rule_fired"
        elif event.type == "speech_text":
            trace.final_decision = "speech_processed"
        else:
            trace.final_decision = "state_updated"

        self._traces.append(trace.to_dict())
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

        if self._dashboard_callback:
            await self._dashboard_callback({
                "type": "trace",
                "trace": trace.to_dict(),
                "status": self.get_system_status(),
                "world": self.get_world_state(),
                "pending": self.rules.get_pending_confirmations(),
                "actions": self.get_action_log(),
                "fire_history": self.get_fire_history(),
            })

    async def _handle_speech(self, event: Event, trace: DecisionTrace) -> None:
        """Process speech through the LLM for intent classification and rule management."""
        text = event.data.get("text", "")
        speaker = event.data.get("who", "unknown")

        if not text or len(text.strip()) < 3:
            return

        self._llm_calls += 1

        # Step 1: Classify intent
        context = self.world.get_llm_context()
        intent_result = await self.llm.classify_intent(text, speaker, context)
        intent = intent_result.get("intent", "conversation")
        confidence = intent_result.get("confidence", 0)

        trace.add_layer(
            "llm_reasoner",
            "intent_classified",
            f"Intent: {intent} ({confidence:.0%})",
            intent_result,
        )

        if intent == "conversation":
            return  # Not a command, ignore

        if intent == "create_rule":
            # Parse into a structured rule
            rule_result = await self.llm.parse_rule(
                text, speaker,
                self.world.get_llm_context(),
                self.rules.rules_summary_for_llm(),
            )

            if not rule_result:
                trace.add_layer("llm_reasoner", "parse_failed", "Failed to parse rule from speech")
                return

            if "clarify" in rule_result:
                await self.executor.ask(rule_result["clarify"], "clarify")
                trace.add_layer("llm_reasoner", "needs_clarification", rule_result["clarify"])
                return

            if rule_result.get("direct_action"):
                # Execute immediately
                exec_result = await self.executor.execute(
                    rule_result["action"],
                    rule_description=f"Direct command from {speaker}: {text}",
                )
                trace.add_layer("executor", "direct_executed", f"Direct command executed", exec_result)
                self._rules_fired += 1
                return

            # Add as a new rule
            rule_result.setdefault("created_by", speaker)
            new_rule = self.rules.add_rule(rule_result)
            await self.executor.notify(f"✓ New rule created: {new_rule.get('description', '')}")
            trace.add_layer(
                "rule_engine",
                "rule_created",
                f"New rule: {new_rule.get('description', '')}",
                new_rule,
            )

        elif intent == "modify_rule":
            change = await self.llm.parse_override(
                text, speaker,
                self.rules.rules_summary_for_llm(),
            )

            if not change:
                trace.add_layer("llm_reasoner", "parse_failed", "Failed to parse rule modification")
                return

            if "clarify" in change:
                await self.executor.ask(change["clarify"], "clarify")
                trace.add_layer("llm_reasoner", "needs_clarification", change["clarify"])
                return

            if "modify" in change:
                updated = self.rules.update_rule(change["modify"], change.get("changes", {}))
                if updated:
                    await self.executor.notify(f"✓ Rule updated: {updated.get('description', '')}")
                    trace.add_layer("rule_engine", "rule_modified", f"Updated: {change['modify']}", updated)

            elif "suspend" in change:
                updated = self.rules.update_rule(change["suspend"], {"active": False})
                if updated:
                    await self.executor.notify(f"✓ Rule suspended: {updated.get('description', '')}")
                    trace.add_layer("rule_engine", "rule_suspended", f"Suspended: {change['suspend']}")

            elif "delete" in change:
                deleted = self.rules.delete_rule(change["delete"])
                if deleted:
                    await self.executor.notify(f"✓ Rule deleted: {change['delete']}")
                    trace.add_layer("rule_engine", "rule_deleted", f"Deleted: {change['delete']}")

        elif intent == "direct_command":
            # Try to parse as a direct action
            rule_result = await self.llm.parse_rule(
                text, speaker,
                self.world.get_llm_context(),
                self.rules.rules_summary_for_llm(),
            )
            if rule_result and rule_result.get("direct_action"):
                exec_result = await self.executor.execute(
                    rule_result["action"],
                    rule_description=f"Direct: {text}",
                )
                trace.add_layer("executor", "direct_executed", "Direct command", exec_result)
                self._rules_fired += 1
            elif rule_result and "action" in rule_result:
                exec_result = await self.executor.execute(
                    rule_result["action"],
                    rule_description=f"Direct: {text}",
                )
                trace.add_layer("executor", "direct_executed", "Direct command (from rule parse)", exec_result)
                self._rules_fired += 1

        elif intent == "confirmation":
            # Check if there's a pending confirmation
            pending = self.rules.get_pending_confirmations()
            if pending:
                # Confirm the most recent one
                last = pending[-1]
                is_yes = any(w in text.lower() for w in ["yes", "yeah", "yep", "sure", "ok", "go ahead", "do it"])
                result = self.rules.confirm_pending(last["rule_id"], is_yes)
                if result and result.get("approved") and result.get("action"):
                    exec_result = await self.executor.execute(result["action"], last["rule_id"])
                    await self.executor.notify(f"✓ Done!")
                    trace.add_layer("executor", "confirmed", "User confirmed", exec_result)
                    self._rules_fired += 1
                else:
                    await self.executor.notify("Cancelled.")
                    trace.add_layer("executor", "rejected", "User rejected")

        elif intent == "question":
            # For now, just log it
            trace.add_layer("llm_reasoner", "question", f"Question from {speaker}: {text}")

    def set_dashboard_callback(self, callback) -> None:
        """Set callback for pushing updates to the dashboard."""
        self._dashboard_callback = callback

    # ─── Dashboard API ──────────────────────────────────────────────

    def get_traces(self, count: int = 50) -> List[dict]:
        """Get recent decision traces for dashboard visualization."""
        return self._traces[-count:]

    def get_system_status(self) -> Dict[str, Any]:
        """Get full system status for dashboard."""
        return {
            "running": self._running,
            "events_processed": self._events_processed,
            "rules_fired": self._rules_fired,
            "llm_calls": self._llm_calls,
            "world_state": self.world.snapshot(),
            "rules": {
                "total": len(self.rules.rules),
                "active": sum(1 for r in self.rules.rules if r.get("active")),
                "pending_confirmations": len(self.rules.get_pending_confirmations()),
            },
            "llm": self.llm.stats(),
            "executor": self.executor.stats(),
        }

    def get_rules(self) -> List[dict]:
        """Get all rules."""
        return self.rules.rules

    def get_world_state(self) -> Dict[str, Any]:
        """Get current world state."""
        return self.world.snapshot()

    def get_fire_history(self) -> List[dict]:
        """Get rule fire history."""
        return self.rules.get_fire_history()

    def get_action_log(self) -> List[dict]:
        """Get executor action log."""
        return self.executor.get_action_log()

    # ─── Manual event injection (for testing via dashboard) ─────

    async def inject_event(self, event_type: str, data: dict) -> dict:
        """Inject a test event into the system."""
        event = Event(type=event_type, data=data)
        await self.bus.publish(event)
        return event.to_dict()
