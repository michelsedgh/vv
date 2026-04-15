"""
Smart Home Decision System — Brain Module

5-layer architecture that fuses voice (speaker ID) and vision (action recognition)
into a single reasoning system for smart home automation.

Layers:
    1. Event Bus      — normalized event stream from all sensors
    2. World State    — fused "current situation" snapshot
    3. Rule Engine    — trigger/condition/action evaluation
    4. LLM Reasoner   — natural language rule creation & ambiguity resolution
    5. Action Executor — smart home APIs, TTS, notifications
"""
