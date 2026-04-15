"""
Layer 1: Event Bus

Central async pub/sub system. Every sensor publishes typed events to one
central queue. All other layers subscribe to it.

The bus also maintains an event history (last 500 events) for the dashboard
to replay recent system activity.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

log = logging.getLogger("brain.event_bus")


@dataclass
class Event:
    """A single typed event flowing through the system."""
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def iso_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp).isoformat(timespec="milliseconds")

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
            "iso_time": self.iso_time,
        }


class EventBus:
    """
    Central event bus with subscriber queues and event history.

    Subscribers get their own asyncio.Queue. When a queue is full,
    the oldest event is dropped to prevent memory buildup.
    """

    def __init__(self, history_size: int = 500):
        self._subscribers: List[asyncio.Queue] = []
        self._history: List[Event] = []
        self._history_size = history_size
        self._listeners: List[Callable[[Event], Coroutine]] = []
        self._lock = asyncio.Lock()

    def subscribe(self, maxsize: int = 200) -> asyncio.Queue:
        """Create a new subscriber queue."""
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def add_listener(self, callback: Callable[[Event], Coroutine]) -> None:
        """Add an async callback listener (called for every event)."""
        self._listeners.append(callback)

    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers and listeners."""
        # Store in history
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size:]

        # Push to subscriber queues
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event to make room
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        # Call async listeners
        for listener in self._listeners:
            try:
                await listener(event)
            except Exception as exc:
                log.warning("Event listener error: %s", exc)

    def publish_sync(self, event: Event, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Thread-safe publish from synchronous code (e.g. voice/poguise threads)."""
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.publish(event), loop)
        else:
            # Fallback: store in history only (no async delivery)
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

    def recent_events(self, count: int = 50, event_type: Optional[str] = None) -> List[dict]:
        """Get recent events for dashboard display."""
        events = self._history
        if event_type:
            events = [e for e in events if e.type == event_type]
        return [e.to_dict() for e in events[-count:]]

    def clear_history(self) -> None:
        """Clear event history."""
        self._history.clear()

    @property
    def total_events(self) -> int:
        return len(self._history)
