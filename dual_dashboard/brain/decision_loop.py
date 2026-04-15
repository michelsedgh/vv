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
from copy import deepcopy
import logging
import re
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
            "data": deepcopy(data) if data is not None else {},
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
        self._queue_items: Dict[str, Dict[str, Any]] = {}
        self._max_queue_items = 80
        self._stream_progress_interval_sec = 0.18

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

    def _iso(self, timestamp: Optional[float] = None) -> str:
        return datetime.fromtimestamp(timestamp or time.time()).isoformat(timespec="milliseconds")

    def _base_llm_state(self) -> Dict[str, Any]:
        return {
            "active": False,
            "phase": None,
            "status": "idle",
            "stream_text": "",
            "latest_output": "",
            "latency_ms": None,
            "chunks": 0,
            "phase_started_ts": None,
            "phase_started_at": None,
            "updated_ts": None,
            "updated_at": None,
            "history": [],
        }

    def _fast_intent_guess(self, text: str) -> Optional[Dict[str, Any]]:
        normalized = " ".join(text.lower().strip().split())
        if not normalized:
            return None

        confirmation_values = {
            "yes", "yeah", "yep", "sure", "ok", "okay", "no", "nope", "nah",
            "go ahead", "do it", "cancel", "stop that", "not now",
        }
        if normalized in confirmation_values:
            return {
                "intent": "confirmation",
                "confidence": 0.99,
                "reasoning": "Matched a direct yes/no confirmation phrase.",
                "_raw_response": "",
                "_latency_ms": 0.0,
                "_fast_path": True,
            }

        question_starts = ("what ", "why ", "how ", "which ", "when ", "who ", "is ", "are ", "do ", "does ", "did ", "can ", "could ", "would ", "will ")
        if normalized.endswith("?") or normalized.startswith(question_starts):
            return {
                "intent": "question",
                "confidence": 0.95,
                "reasoning": "Matched a direct question pattern.",
                "_raw_response": "",
                "_latency_ms": 0.0,
                "_fast_path": True,
            }

        modify_phrases = (
            "disable rule", "pause rule", "resume rule", "delete rule", "remove rule",
            "suspend rule", "turn that rule off", "dont do that anymore", "don't do that anymore",
        )
        if normalized.startswith(("disable ", "pause ", "resume ", "delete ", "remove ", "suspend ")) or any(phrase in normalized for phrase in modify_phrases):
            return {
                "intent": "modify_rule",
                "confidence": 0.94,
                "reasoning": "Matched a rule-management phrase.",
                "_raw_response": "",
                "_latency_ms": 0.0,
                "_fast_path": True,
            }

        if normalized.startswith(("when ", "if ", "whenever ", "after ", "before ", "once ")) or re.search(r"\b(when|if|whenever|after i|before i|once|every time|any time)\b", normalized):
            return {
                "intent": "create_rule",
                "confidence": 0.97,
                "reasoning": "Matched an automation trigger phrase.",
                "_raw_response": "",
                "_latency_ms": 0.0,
                "_fast_path": True,
            }

        direct_command_starts = (
            "turn on ", "turn off ", "set ", "dim ", "brighten ", "open ", "close ",
            "lock ", "unlock ", "activate ", "deactivate ", "play ",
        )
        if normalized.startswith(direct_command_starts) or normalized.endswith(" now") or " right now" in normalized:
            return {
                "intent": "direct_command",
                "confidence": 0.92,
                "reasoning": "Matched an immediate command phrase.",
                "_raw_response": "",
                "_latency_ms": 0.0,
                "_fast_path": True,
            }

        return None

    def _ensure_queue_item(self, event: Event) -> Dict[str, Any]:
        entry = self._queue_items.get(event.event_id)
        if entry is None:
            entry = {
                "event_id": event.event_id,
                "event": event.to_dict(),
                "status": "queued",
                "current_stage": "queued",
                "stage_status": "queued",
                "details": "Waiting to enter decision loop.",
                "queued_ts": event.timestamp,
                "queued_at": event.iso_time,
                "started_ts": None,
                "started_at": None,
                "stage_started_ts": event.timestamp,
                "stage_started_at": event.iso_time,
                "updated_ts": event.timestamp,
                "updated_at": event.iso_time,
                "completed_ts": None,
                "completed_at": None,
                "llm": self._base_llm_state(),
            }
            self._queue_items[event.event_id] = entry
            self._trim_queue_items()
        return entry

    def _trim_queue_items(self) -> None:
        if len(self._queue_items) <= self._max_queue_items:
            return
        active = [item for item in self._queue_items.values() if item.get("status") in ("queued", "processing")]
        completed = [item for item in self._queue_items.values() if item.get("status") not in ("queued", "processing")]
        completed.sort(key=lambda item: item.get("updated_ts", 0), reverse=True)
        keep = active + completed[: max(0, self._max_queue_items - len(active))]
        self._queue_items = {item["event_id"]: item for item in keep}

    def _set_queue_stage(
        self,
        event: Event,
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        stage_status: Optional[str] = None,
        details: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = self._ensure_queue_item(event)
        now = time.time()
        if status is not None:
            entry["status"] = status
        if stage is not None:
            if entry.get("current_stage") != stage:
                entry["stage_started_ts"] = now
                entry["stage_started_at"] = self._iso(now)
            entry["current_stage"] = stage
        if stage_status is not None:
            entry["stage_status"] = stage_status
        if details is not None:
            entry["details"] = details
        entry["updated_ts"] = now
        entry["updated_at"] = self._iso(now)
        return entry

    def _start_llm_phase(self, event: Event, phase: str) -> Dict[str, Any]:
        entry = self._set_queue_stage(
            event,
            status="processing",
            stage="llm_reasoner",
            stage_status="streaming",
            details=f"LLM {phase.replace('_', ' ')}",
        )
        now = time.time()
        phase_entry = {
            "phase": phase,
            "status": "streaming",
            "text": "",
            "latency_ms": None,
            "started_ts": now,
            "started_at": self._iso(now),
            "updated_ts": now,
            "updated_at": self._iso(now),
        }
        llm_state = entry.setdefault("llm", self._base_llm_state())
        llm_state["active"] = True
        llm_state["phase"] = phase
        llm_state["status"] = "streaming"
        llm_state["stream_text"] = ""
        llm_state["latency_ms"] = None
        llm_state["chunks"] = 0
        llm_state["phase_started_ts"] = now
        llm_state["phase_started_at"] = self._iso(now)
        llm_state["updated_ts"] = now
        llm_state["updated_at"] = self._iso(now)
        llm_state.setdefault("history", []).append(phase_entry)
        entry["_last_stream_publish_ts"] = 0.0
        return entry

    def _append_llm_text(self, event: Event, chunk: str) -> Dict[str, Any]:
        entry = self._ensure_queue_item(event)
        llm_state = entry.setdefault("llm", self._base_llm_state())
        now = time.time()
        llm_state["active"] = True
        llm_state["status"] = "streaming"
        llm_state["stream_text"] = f"{llm_state.get('stream_text', '')}{chunk}"
        llm_state["latest_output"] = llm_state["stream_text"]
        llm_state["chunks"] = int(llm_state.get("chunks", 0)) + 1
        llm_state["updated_ts"] = now
        llm_state["updated_at"] = self._iso(now)
        if llm_state.get("history"):
            llm_state["history"][-1]["text"] = llm_state["stream_text"]
            llm_state["history"][-1]["updated_ts"] = now
            llm_state["history"][-1]["updated_at"] = llm_state["updated_at"]
        return entry

    def _finish_llm_phase(
        self,
        event: Event,
        *,
        raw_output: str,
        latency_ms: Optional[float],
        status: str = "complete",
    ) -> Dict[str, Any]:
        entry = self._ensure_queue_item(event)
        llm_state = entry.setdefault("llm", self._base_llm_state())
        now = time.time()
        llm_state["active"] = False
        llm_state["status"] = status
        llm_state["stream_text"] = raw_output or llm_state.get("stream_text", "")
        llm_state["latest_output"] = raw_output or llm_state.get("latest_output", "")
        llm_state["latency_ms"] = latency_ms
        llm_state["updated_ts"] = now
        llm_state["updated_at"] = self._iso(now)
        if llm_state.get("history"):
            llm_state["history"][-1]["text"] = llm_state["stream_text"]
            llm_state["history"][-1]["status"] = status
            llm_state["history"][-1]["latency_ms"] = latency_ms
            llm_state["history"][-1]["updated_ts"] = now
            llm_state["history"][-1]["updated_at"] = llm_state["updated_at"]
        return entry

    async def _publish_dashboard(self, payload: Dict[str, Any]) -> None:
        if self._dashboard_callback:
            await self._dashboard_callback(payload)

    async def _publish_queue_update(self) -> None:
        await self._publish_dashboard({
            "type": "queue",
            "queue": self.get_queue_state(),
        })

    async def note_event_enqueued(self, event: Event) -> None:
        self._set_queue_stage(
            event,
            status="queued",
            stage="queued",
            stage_status="queued",
            details="Waiting to enter decision loop.",
        )
        await self._publish_queue_update()

    async def _publish_trace_progress(
        self,
        trace: DecisionTrace,
        current_stage: str,
        status: str,
        details: str,
    ) -> None:
        entry = self._queue_items.get(trace.event.event_id)
        await self._publish_dashboard({
            "type": "trace_progress",
            "trace": trace.to_dict(),
            "progress": {
                "event_id": trace.event.event_id,
                "current_stage": current_stage,
                "status": status,
                "details": details,
                "llm": deepcopy(entry.get("llm", {})) if entry else {},
            },
            "queue": self.get_queue_state(),
        })

    def _should_publish_stream_progress(self, event: Event, *, force: bool = False) -> bool:
        entry = self._ensure_queue_item(event)
        now = time.time()
        last = float(entry.get("_last_stream_publish_ts") or 0.0)
        if not force and (now - last) < self._stream_progress_interval_sec:
            return False
        entry["_last_stream_publish_ts"] = now
        return True

    async def _record_layer(
        self,
        trace: DecisionTrace,
        layer: str,
        status: str,
        details: str,
        data: Optional[dict] = None,
    ) -> None:
        trace.add_layer(layer, status, details, data)
        self._set_queue_stage(
            trace.event,
            status="processing",
            stage=layer,
            stage_status=status,
            details=details,
        )
        await self._publish_trace_progress(trace, layer, status, details)

    async def _process_rule_results(self, trace: DecisionTrace, rule_results: List[dict]) -> None:
        """Apply deterministic rule outcomes to the executor."""
        for result in rule_results:
            decision = result.get("decision", "")

            if decision.startswith("fire_"):
                permission = result.get("permission", "ask")
                action = result.get("action", {})
                rule_id = result.get("rule_id", "")
                rule_desc = result.get("rule_description", "")

                if permission == "auto":
                    exec_result = await self.executor.execute(action, rule_id, rule_desc)
                    await self._record_layer(
                        trace,
                        "executor",
                        "executed",
                        f"Auto-executed: {rule_desc}",
                        exec_result,
                    )
                    self._rules_fired += 1

                elif permission == "notify":
                    exec_result = await self.executor.execute(action, rule_id, rule_desc)
                    await self.executor.notify(f"✓ {rule_desc}")
                    await self._record_layer(
                        trace,
                        "executor",
                        "executed_notified",
                        f"Executed with notification: {rule_desc}",
                        exec_result,
                    )
                    self._rules_fired += 1

                elif permission == "ask":
                    self.rules.add_pending_confirmation(rule_id, action, rule_desc)
                    await self.executor.ask(f"Should I {rule_desc.lower()}?", rule_id)
                    await self._record_layer(
                        trace,
                        "executor",
                        "asked",
                        f"Asking user: {rule_desc}",
                        {"rule_id": rule_id},
                    )

                elif permission == "suggest":
                    await self.executor.notify(f"💡 Suggestion: {rule_desc}")
                    await self._record_layer(
                        trace,
                        "executor",
                        "suggested",
                        f"Suggestion sent: {rule_desc}",
                    )

            elif decision == "cooldown":
                await self._record_layer(
                    trace,
                    "rule_engine",
                    "cooldown",
                    f"Rule {result['rule_id']} in cooldown: {result.get('details', '')}",
                )

            elif decision == "conditions_not_met":
                await self._record_layer(
                    trace,
                    "rule_engine",
                    "conditions_unmet",
                    f"Rule {result['rule_id']} conditions not met",
                )

    async def _handle_user_response(self, event: Event, trace: DecisionTrace) -> None:
        """Resolve a pending ask/suggest style confirmation."""
        rule_id = event.data.get("rule_id", "")
        approved = event.data.get("answer", "").lower() in ("yes", "yeah", "yep", "go ahead", "do it", "sure", "ok")
        result = self.rules.confirm_pending(rule_id, approved)
        if result and result.get("approved") and result.get("action"):
            exec_result = await self.executor.execute(result["action"], rule_id)
            await self._record_layer(trace, "executor", "confirmed", f"User confirmed rule {rule_id}", exec_result)
            self._rules_fired += 1
        elif result:
            await self._record_layer(trace, "executor", "rejected", f"User rejected rule {rule_id}")

    def _complete_queue_item(self, trace: DecisionTrace, status: str, details: str) -> None:
        entry = self._set_queue_stage(
            trace.event,
            status=status,
            stage=trace.layers[-1]["layer"] if trace.layers else "done",
            stage_status=status,
            details=details,
        )
        now = time.time()
        entry["completed_ts"] = now
        entry["completed_at"] = self._iso(now)

    def _fail_queue_item(self, event: Event, details: str) -> None:
        current_stage = self._queue_items.get(event.event_id, {}).get("current_stage") or "processing"
        entry = self._set_queue_stage(
            event,
            status="error",
            stage=current_stage,
            stage_status="error",
            details=details,
        )
        now = time.time()
        entry["completed_ts"] = now
        entry["completed_at"] = self._iso(now)

    def get_queue_state(self) -> Dict[str, Any]:
        queued = []
        processing = []
        finished = []
        for item in self._queue_items.values():
            payload = deepcopy(item)
            if item.get("status") == "queued":
                queued.append(payload)
            elif item.get("status") == "processing":
                processing.append(payload)
            else:
                finished.append(payload)

        queued.sort(key=lambda item: item.get("queued_ts") or 0)
        processing.sort(key=lambda item: item.get("started_ts") or 0, reverse=True)
        finished.sort(key=lambda item: item.get("updated_ts") or 0, reverse=True)

        visible = processing + queued + finished[:30]
        return {
            "pending_count": len(queued),
            "active_count": len(processing),
            "worker_busy": bool(processing),
            "queue_depth": self._queue.qsize() if self._queue else 0,
            "items": visible,
        }

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
                self._fail_queue_item(event, f"{type(exc).__name__}: {exc}")
                await self._publish_queue_update()
                log.error("Error processing event %s: %s", event.type, exc, exc_info=True)

    async def _process_event(self, event: Event) -> None:
        """Process a single event through all layers."""
        trace = DecisionTrace(event)
        self._events_processed += 1
        entry = self._set_queue_stage(
            event,
            status="processing",
            stage="event_bus",
            stage_status="received",
            details=f"Event: {event.type}",
        )
        if entry.get("started_ts") is None:
            now = time.time()
            entry["started_ts"] = now
            entry["started_at"] = self._iso(now)
        await self._publish_queue_update()

        # ── Layer 1: Event Bus (already done — event is here) ──
        await self._record_layer(trace, "event_bus", "received", f"Event: {event.type}", event.to_dict())

        # ── Layer 2: World State ──
        derived_events = self.world.update_from_event(event)
        await self._record_layer(
            trace,
            "world_state",
            "updated",
            f"State updated, {len(derived_events)} derived events",
            {"derived_events": [e.get("type") for e in derived_events]},
        )

        # Publish derived events (e.g., person_entered, action_changed)
        for de in derived_events:
            derived_event = Event(type=de["type"], data=de.get("data", {}))
            await self.bus.publish(derived_event)

        rule_results: List[dict] = []

        # Speech needs to be classified before generic rule evaluation.
        if event.type == "speech_text":
            if self.llm._available:
                await self._handle_speech(event, trace)
            else:
                await self._record_layer(trace, "llm_reasoner", "offline", "Speech arrived, but the LLM is offline.")

        elif event.type == "user_response":
            await self._handle_user_response(event, trace)

        else:
            # ── Layer 3: Rule Engine ──
            rule_results = self.rules.evaluate(event)

            # Also evaluate derived events
            for de in derived_events:
                derived_event = Event(type=de["type"], data=de.get("data", {}))
                rule_results.extend(self.rules.evaluate(derived_event))

            if rule_results:
                await self._record_layer(
                    trace,
                    "rule_engine",
                    "evaluated",
                    f"{len(rule_results)} rule(s) matched",
                    {"results": rule_results},
                )
            else:
                await self._record_layer(trace, "rule_engine", "no_match", "No rules triggered")

            await self._process_rule_results(trace, rule_results)

        # ── Finalize trace ──
        fired = any(r.get("decision", "").startswith("fire_") for r in rule_results)
        if fired:
            trace.final_decision = "rule_fired"
        elif event.type == "speech_text":
            trace.final_decision = "speech_processed"
        elif event.type == "user_response":
            trace.final_decision = "confirmation_processed"
        else:
            trace.final_decision = "state_updated"

        trace_payload = trace.to_dict()
        self._traces.append(trace_payload)
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

        self._complete_queue_item(trace, "completed", trace.final_decision or "completed")
        await self._publish_dashboard({
            "type": "trace",
            "trace": trace_payload,
            "status": self.get_system_status(),
            "world": self.get_world_state(),
            "pending": self.rules.get_pending_confirmations(),
            "actions": self.get_action_log(),
            "fire_history": self.get_fire_history(),
            "queue": self.get_queue_state(),
        })

    async def _handle_speech(self, event: Event, trace: DecisionTrace) -> None:
        """Process speech through the LLM for intent classification and rule management."""
        text = event.data.get("text", "")
        speaker = event.data.get("who", "unknown")

        if not text or len(text.strip()) < 3:
            return

        # Step 1: Classify intent
        context = self.world.get_llm_context()
        intent_result = self._fast_intent_guess(text)
        if intent_result is None:
            self._llm_calls += 1
            self._start_llm_phase(event, "intent_classification")
            await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM intent classification")

            async def on_intent_delta(chunk: str) -> None:
                self._append_llm_text(event, chunk)
                if self._should_publish_stream_progress(event):
                    await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM intent classification")

            intent_result = await self.llm.classify_intent(text, speaker, context, on_delta=on_intent_delta)
            self._finish_llm_phase(
                event,
                raw_output=intent_result.get("_raw_response", ""),
                latency_ms=intent_result.get("_latency_ms"),
            )
        intent = intent_result.get("intent", "conversation")
        confidence = intent_result.get("confidence", 0)
        fast_path = bool(intent_result.get("_fast_path"))

        await self._record_layer(
            trace,
            "llm_reasoner",
            "intent_classified",
            f"Intent: {intent} ({confidence:.0%}){' via fast-path' if fast_path else ''}",
            intent_result,
        )

        if intent == "conversation":
            return  # Not a command, ignore

        if intent == "create_rule":
            # Parse into a structured rule
            self._llm_calls += 1
            self._start_llm_phase(event, "rule_creation")
            await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM rule creation")

            async def on_rule_delta(chunk: str) -> None:
                self._append_llm_text(event, chunk)
                if self._should_publish_stream_progress(event):
                    await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM rule creation")

            rule_result = await self.llm.parse_rule(
                text, speaker,
                self.world.get_llm_context(),
                self.rules.rules_summary_for_llm(max_rules=8, unique_only=True),
                on_delta=on_rule_delta,
            )

            if not rule_result:
                latest_output = self._queue_items.get(event.event_id, {}).get("llm", {}).get("stream_text", "")
                self._finish_llm_phase(
                    event,
                    raw_output=latest_output,
                    latency_ms=self.llm.stats().get("last_latency_ms"),
                    status="error",
                )
                await self._record_layer(trace, "llm_reasoner", "parse_failed", "Failed to parse rule from speech")
                return

            self._finish_llm_phase(
                event,
                raw_output=rule_result.get("_raw_response", ""),
                latency_ms=rule_result.get("_latency_ms"),
            )
            await self._record_layer(trace, "llm_reasoner", "rule_parsed", "Structured rule parsed from speech.", rule_result)

            if "clarify" in rule_result:
                await self.executor.ask(rule_result["clarify"], "clarify")
                await self._record_layer(trace, "llm_reasoner", "needs_clarification", rule_result["clarify"], rule_result)
                return

            if rule_result.get("direct_action"):
                # Execute immediately
                exec_result = await self.executor.execute(
                    rule_result["action"],
                    rule_description=f"Direct command from {speaker}: {text}",
                )
                await self._record_layer(trace, "executor", "direct_executed", "Direct command executed", exec_result)
                self._rules_fired += 1
                return

            # Add as a new rule
            rule_result.pop("_raw_response", None)
            rule_result.pop("_latency_ms", None)
            upsert_result = self.rules.upsert_rule(rule_result, speaker=speaker, source_text=text)
            final_rule = upsert_result.get("rule", {})
            upsert_status = upsert_result.get("status", "created")

            if upsert_status == "created":
                await self.executor.notify(f"✓ New rule created: {final_rule.get('description', '')}")
                await self._record_layer(
                    trace,
                    "rule_engine",
                    "rule_created",
                    f"New rule: {final_rule.get('description', '')}",
                    {**upsert_result, "rule": final_rule},
                )
            elif upsert_status == "reactivated":
                await self.executor.notify(f"✓ Existing rule reactivated: {final_rule.get('description', '')}")
                await self._record_layer(
                    trace,
                    "rule_engine",
                    "rule_reactivated",
                    f"Existing rule reactivated: {final_rule.get('description', '')}",
                    {**upsert_result, "rule": final_rule},
                )
            else:
                await self.executor.notify(f"✓ Rule already existed: {final_rule.get('description', '')}")
                await self._record_layer(
                    trace,
                    "rule_engine",
                    "rule_existing",
                    f"Matched existing rule: {final_rule.get('description', '')}",
                    {**upsert_result, "rule": final_rule},
                )

        elif intent == "modify_rule":
            self._llm_calls += 1
            self._start_llm_phase(event, "rule_override")
            await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM rule override")

            async def on_override_delta(chunk: str) -> None:
                self._append_llm_text(event, chunk)
                if self._should_publish_stream_progress(event):
                    await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM rule override")

            change = await self.llm.parse_override(
                text, speaker,
                self.rules.rules_summary_for_llm(max_rules=12, unique_only=True),
                on_delta=on_override_delta,
            )

            if not change:
                latest_output = self._queue_items.get(event.event_id, {}).get("llm", {}).get("stream_text", "")
                self._finish_llm_phase(
                    event,
                    raw_output=latest_output,
                    latency_ms=self.llm.stats().get("last_latency_ms"),
                    status="error",
                )
                await self._record_layer(trace, "llm_reasoner", "parse_failed", "Failed to parse rule modification")
                return

            self._finish_llm_phase(
                event,
                raw_output=change.get("_raw_response", ""),
                latency_ms=change.get("_latency_ms"),
            )
            await self._record_layer(trace, "llm_reasoner", "override_parsed", "Rule modification parsed from speech.", change)

            if "clarify" in change:
                await self.executor.ask(change["clarify"], "clarify")
                await self._record_layer(trace, "llm_reasoner", "needs_clarification", change["clarify"], change)
                return

            change.pop("_raw_response", None)
            change.pop("_latency_ms", None)
            if "modify" in change:
                updated = self.rules.update_rule(change["modify"], change.get("changes", {}))
                if updated:
                    await self.executor.notify(f"✓ Rule updated: {updated.get('description', '')}")
                    await self._record_layer(trace, "rule_engine", "rule_modified", f"Updated: {change['modify']}", updated)

            elif "suspend" in change:
                updated = self.rules.update_rule(change["suspend"], {"active": False})
                if updated:
                    await self.executor.notify(f"✓ Rule suspended: {updated.get('description', '')}")
                    await self._record_layer(trace, "rule_engine", "rule_suspended", f"Suspended: {change['suspend']}")

            elif "delete" in change:
                deleted = self.rules.delete_rule(change["delete"])
                if deleted:
                    await self.executor.notify(f"✓ Rule deleted: {change['delete']}")
                    await self._record_layer(trace, "rule_engine", "rule_deleted", f"Deleted: {change['delete']}")

        elif intent == "direct_command":
            # Try to parse as a direct action
            self._llm_calls += 1
            self._start_llm_phase(event, "direct_command")
            await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM direct command parse")

            async def on_command_delta(chunk: str) -> None:
                self._append_llm_text(event, chunk)
                if self._should_publish_stream_progress(event):
                    await self._publish_trace_progress(trace, "llm_reasoner", "streaming", "LLM direct command parse")

            rule_result = await self.llm.parse_rule(
                text, speaker,
                self.world.get_llm_context(),
                self.rules.rules_summary_for_llm(max_rules=8, unique_only=True),
                on_delta=on_command_delta,
            )
            latest_output = self._queue_items.get(event.event_id, {}).get("llm", {}).get("stream_text", "")
            if rule_result:
                self._finish_llm_phase(
                    event,
                    raw_output=rule_result.get("_raw_response", latest_output),
                    latency_ms=rule_result.get("_latency_ms"),
                )
                await self._record_layer(trace, "llm_reasoner", "command_parsed", "Direct command parsed from speech.", rule_result)
            else:
                self._finish_llm_phase(
                    event,
                    raw_output=latest_output,
                    latency_ms=self.llm.stats().get("last_latency_ms"),
                    status="error",
                )
                await self._record_layer(trace, "llm_reasoner", "parse_failed", "Failed to parse direct command from speech.")
            if rule_result and rule_result.get("direct_action"):
                exec_result = await self.executor.execute(
                    rule_result["action"],
                    rule_description=f"Direct: {text}",
                )
                await self._record_layer(trace, "executor", "direct_executed", "Direct command", exec_result)
                self._rules_fired += 1
            elif rule_result and "action" in rule_result:
                exec_result = await self.executor.execute(
                    rule_result["action"],
                    rule_description=f"Direct: {text}",
                )
                await self._record_layer(trace, "executor", "direct_executed", "Direct command (from rule parse)", exec_result)
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
                    await self._record_layer(trace, "executor", "confirmed", "User confirmed", exec_result)
                    self._rules_fired += 1
                else:
                    await self.executor.notify("Cancelled.")
                    await self._record_layer(trace, "executor", "rejected", "User rejected")

        elif intent == "question":
            # For now, just log it
            await self._record_layer(trace, "llm_reasoner", "question", f"Question from {speaker}: {text}")

    def set_dashboard_callback(self, callback) -> None:
        """Set callback for pushing updates to the dashboard."""
        self._dashboard_callback = callback

    # ─── Dashboard API ──────────────────────────────────────────────

    def get_traces(self, count: int = 50) -> List[dict]:
        """Get recent decision traces for dashboard visualization."""
        return self._traces[-count:]

    def get_system_status(self) -> Dict[str, Any]:
        """Get full system status for dashboard."""
        queue = self.get_queue_state()
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
            "queue": {
                "pending_count": queue["pending_count"],
                "active_count": queue["active_count"],
                "queue_depth": queue["queue_depth"],
            },
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
