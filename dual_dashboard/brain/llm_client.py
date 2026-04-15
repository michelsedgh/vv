"""
Layer 4: LLM Reasoner (LM Studio)

Connects to LM Studio's OpenAI-compatible API for:
    1. Intent classification — is the user creating a rule, modifying one, asking a question?
    2. Rule parsing — convert natural language to structured rule JSON
    3. Override handling — parse modification/suspension commands
    4. Ambiguity resolution — should this rule fire given uncertain context?

All prompts return structured JSON for reliable machine parsing.
"""
from __future__ import annotations

import inspect
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

log = logging.getLogger("brain.llm")

# ─── Prompt Templates ──────────────────────────────────────────────

INTENT_CLASSIFICATION_PROMPT = """You are a smart home voice command classifier. Classify the user's speech into exactly ONE of these intents:

- "create_rule": User wants to create a new automation (e.g., "turn off lights when I leave")
- "modify_rule": User wants to change or disable an existing rule (e.g., "actually keep the lights on tonight")
- "direct_command": User wants something done RIGHT NOW (e.g., "turn on kitchen lights")
- "question": User is asking about the system (e.g., "what rules are active?")
- "confirmation": User is responding yes/no to a question (e.g., "yes", "go ahead", "no thanks")
- "conversation": Normal speech, not a command (e.g., "pass me the salt")

Current context:
{context}

User said: "{text}"
Speaker: {speaker}

Respond with ONLY a JSON object:
{{"intent": "one_of_the_above", "confidence": 0.0_to_1.0, "reasoning": "brief explanation"}}"""

RULE_CREATION_PROMPT = """You are a smart home rule creator. Convert the user's request into a structured automation rule.

Current situation:
{world_state}

Existing rules:
{existing_rules}

User said: "{text}"
Speaker: {speaker}

IMPORTANT edge cases to handle:
- If no duration specified (e.g., "turn on lights"), assume it stays until manually changed or a contrasting rule fires
- If "when I leave" is mentioned, use trigger type "person_left" with the speaker's name
- If time-based, use appropriate time_of_day conditions
- For "now" commands, create a direct_action instead of a rule

Available trigger types: action_changed, action_detected, speaker_active, speaker_silent, person_entered, person_left, time_trigger
Available action types: smart_home, tts, notify
Available permissions: auto, notify, ask, suggest

Respond with ONLY a JSON object:
{{
    "id": "unique_snake_case_id",
    "description": "human-readable description",
    "trigger": {{
        "type": "event_type",
        "action": "optional — for action triggers",
        "action_in": ["optional — list of actions"],
        "who": "optional — for speaker triggers",
        "conditions": [
            {{"time_of_day": ["evening", "night"]}},
            {{"people_present_any": true}}
        ]
    }},
    "action": {{
        "type": "smart_home|tts|notify",
        "command": "command_name",
        "params": {{}},
        "message": "optional — for tts/notify"
    }},
    "permission": "auto|notify|ask|suggest",
    "cooldown_sec": 60,
    "expires": null
}}

If the request is a direct command (do it NOW, not a rule), respond with:
{{"direct_action": true, "action": {{"type": "...", "command": "...", "params": {{}}}}}}

If unclear, respond with:
{{"clarify": "your question to the user"}}"""

OVERRIDE_PROMPT = """You are a smart home rule modifier. The user wants to change or temporarily override an automation rule.

Active rules:
{rules}

User said: "{text}"
Speaker: {speaker}

Respond with ONLY one of these JSON formats:

1. Permanent change:
{{"modify": "rule_id", "changes": {{"field": "new_value"}}}}

2. Temporary disable:
{{"suspend": "rule_id", "until": "ISO datetime or 'tomorrow' or 'manual'"}}

3. One-time override:
{{"one_time_override": "rule_id", "action": {{"type": "...", "command": "...", "params": {{}}}}}}

4. Delete rule:
{{"delete": "rule_id"}}

5. If ambiguous:
{{"clarify": "your question to the user"}}"""

AMBIGUITY_PROMPT = """Given the current situation, should this automation fire?

Rule: {rule_description}
Trigger event: {event_type} — {event_data}
Current activity: {current_action} ({action_confidence:.0%} confidence)
People present: {people_present}
Time: {current_time} ({time_of_day})
Recent actions: {recent_actions}

Consider:
- Is the confidence high enough to trust the activity detection?
- Is this the right time/context for this automation?
- Could this be a false positive (brief/transitional action)?

Respond with ONLY a JSON object:
{{"should_fire": true_or_false, "confidence": 0.0_to_1.0, "reasoning": "brief explanation"}}"""


class LLMClient:
    """
    LM Studio API client (OpenAI-compatible).
    
    Handles structured JSON responses with retry logic
    and response parsing.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.1.198:1234",
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

        # Stats for dashboard
        self._call_count = 0
        self._total_latency = 0.0
        self._last_latency = 0.0
        self._last_error: Optional[str] = None
        self._available = False

    async def check_health(self) -> bool:
        """Check if LM Studio is reachable and has a model loaded."""
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models and not self.model:
                    # Auto-select first available model
                    self.model = models[0].get("id", "")
                    log.info("Auto-selected LM Studio model: %s", self.model)
                self._available = len(models) > 0
                return self._available
        except Exception as exc:
            log.debug("LM Studio health check failed: %s", exc)
            self._available = False
        return False

    async def _emit_delta(
        self,
        callback: Optional[Callable[[str], Awaitable[None] | None]],
        chunk: str,
    ) -> None:
        if not callback or not chunk:
            return
        result = callback(chunk)
        if inspect.isawaitable(result):
            await result

    async def _chat(
        self,
        system: str,
        user: str,
        on_delta: Optional[Callable[[str], Awaitable[None] | None]] = None,
        max_tokens: Optional[int] = None,
    ) -> Optional[dict]:
        """Make a chat completion request to LM Studio."""
        t0 = time.time()
        self._call_count += 1

        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": bool(on_delta),
        }
        if self.model:
            payload["model"] = self.model

        try:
            content = ""
            if on_delta:
                parts: List[str] = []
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            choice = (data.get("choices") or [{}])[0]
                            delta = choice.get("delta") or {}
                            piece = (
                                delta.get("content")
                                or delta.get("text")
                            )
                            if piece:
                                parts.append(piece)
                                await self._emit_delta(on_delta, piece)
                        else:
                            # Fallback for non-streaming-compatible gateways.
                            content = line.strip()
                    if not content:
                        content = "".join(parts).strip()
            else:
                resp = await self._client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()

            latency = time.time() - t0
            self._last_latency = latency
            self._total_latency += latency
            self._last_error = None
            self._available = True

            log.debug("LLM response (%.1fs): %s", latency, content[:200])
            return {
                "text": content,
                "latency_ms": round(latency * 1000, 1),
            }

        except Exception as exc:
            latency = time.time() - t0
            self._last_latency = latency
            self._last_error = str(exc)
            log.error("LLM call failed (%.1fs): %s", latency, exc)
            return None

    def _parse_json(self, text: Optional[str]) -> Optional[dict]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        if not text:
            return None
        # Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` markers)
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            log.warning("Failed to parse LLM JSON response: %s", text[:300])
            return None

    async def classify_intent(
        self,
        text: str,
        speaker: str,
        context: str,
        on_delta: Optional[Callable[[str], Awaitable[None] | None]] = None,
    ) -> Dict[str, Any]:
        """Classify user speech intent."""
        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            context=context,
            text=text,
            speaker=speaker,
        )
        response = await self._chat(
            "You are a precise intent classifier. Always respond with valid JSON only.",
            prompt,
            on_delta=on_delta,
            max_tokens=96,
        )
        raw_text = response["text"] if response else None
        result = self._parse_json(raw_text)
        if result and "intent" in result:
            result["_raw_response"] = raw_text or ""
            result["_latency_ms"] = response["latency_ms"] if response else round(self._last_latency * 1000, 1)
            return result
        # Fallback
        return {
            "intent": "conversation",
            "confidence": 0.5,
            "reasoning": "Failed to classify",
            "_raw_response": raw_text or "",
            "_latency_ms": response["latency_ms"] if response else round(self._last_latency * 1000, 1),
        }

    async def parse_rule(
        self,
        text: str,
        speaker: str,
        world_state: str,
        existing_rules: str,
        on_delta: Optional[Callable[[str], Awaitable[None] | None]] = None,
    ) -> Optional[dict]:
        """Parse user speech into a structured rule."""
        prompt = RULE_CREATION_PROMPT.format(
            world_state=world_state,
            existing_rules=existing_rules,
            text=text,
            speaker=speaker,
        )
        response = await self._chat(
            "You are a smart home automation designer. Always respond with valid JSON only.",
            prompt,
            on_delta=on_delta,
            max_tokens=320,
        )
        raw_text = response["text"] if response else None
        result = self._parse_json(raw_text)
        if result is not None:
            result["_raw_response"] = raw_text or ""
            result["_latency_ms"] = response["latency_ms"] if response else round(self._last_latency * 1000, 1)
        return result

    async def parse_override(
        self,
        text: str,
        speaker: str,
        rules: str,
        on_delta: Optional[Callable[[str], Awaitable[None] | None]] = None,
    ) -> Optional[dict]:
        """Parse a rule modification command."""
        prompt = OVERRIDE_PROMPT.format(
            rules=rules,
            text=text,
            speaker=speaker,
        )
        response = await self._chat(
            "You are a smart home rule editor. Always respond with valid JSON only.",
            prompt,
            on_delta=on_delta,
            max_tokens=220,
        )
        raw_text = response["text"] if response else None
        result = self._parse_json(raw_text)
        if result is not None:
            result["_raw_response"] = raw_text or ""
            result["_latency_ms"] = response["latency_ms"] if response else round(self._last_latency * 1000, 1)
        return result

    async def should_fire(
        self,
        rule_description: str,
        event: dict,
        world_state: Dict[str, Any],
        on_delta: Optional[Callable[[str], Awaitable[None] | None]] = None,
    ) -> Dict[str, Any]:
        """Ask LLM if a rule should fire given ambiguous context."""
        prompt = AMBIGUITY_PROMPT.format(
            rule_description=rule_description,
            event_type=event.get("type", "unknown"),
            event_data=json.dumps(event.get("data", {})),
            current_action=world_state.get("current_action", "none"),
            action_confidence=world_state.get("action_confidence", 0),
            people_present=list(world_state.get("people_present", {}).keys()),
            current_time=world_state.get("current_time", "unknown"),
            time_of_day=world_state.get("time_of_day", "unknown"),
            recent_actions=world_state.get("recent_actions", [])[-3:],
        )
        response = await self._chat(
            "You are a smart home decision engine. Always respond with valid JSON only.",
            prompt,
            on_delta=on_delta,
            max_tokens=96,
        )
        raw_text = response["text"] if response else None
        result = self._parse_json(raw_text)
        if result and "should_fire" in result:
            result["_raw_response"] = raw_text or ""
            result["_latency_ms"] = response["latency_ms"] if response else round(self._last_latency * 1000, 1)
            return result
        return {
            "should_fire": False,
            "confidence": 0.0,
            "reasoning": "Failed to evaluate",
            "_raw_response": raw_text or "",
            "_latency_ms": response["latency_ms"] if response else round(self._last_latency * 1000, 1),
        }

    def stats(self) -> Dict[str, Any]:
        """Get LLM client stats for dashboard."""
        avg_latency = (self._total_latency / self._call_count) if self._call_count else 0
        return {
            "available": self._available,
            "model": self.model,
            "base_url": self.base_url,
            "call_count": self._call_count,
            "last_latency_ms": round(self._last_latency * 1000, 1),
            "avg_latency_ms": round(avg_latency * 1000, 1),
            "last_error": self._last_error,
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
