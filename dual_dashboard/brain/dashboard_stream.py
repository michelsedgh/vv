from __future__ import annotations

import asyncio
from typing import Any, Dict, List


class DashboardStreamHub:
    """Small fan-out hub for the Brain dashboard live stream."""

    def __init__(self) -> None:
        self._subscribers: List[asyncio.Queue[Dict[str, Any]]] = []

    def subscribe(self, maxsize: int = 200) -> asyncio.Queue[Dict[str, Any]]:
        queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Dict[str, Any]]) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    async def publish(self, payload: Dict[str, Any]) -> None:
        dead: List[asyncio.Queue[Dict[str, Any]]] = []
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(queue)
        for queue in dead:
            self.unsubscribe(queue)
