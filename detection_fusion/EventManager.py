from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List


@dataclass(frozen=True, slots=True)
class FusionEvent:
    event_type: str
    entity_id: str | None
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"event_type": self.event_type, "entity_id": self.entity_id,
                "timestamp": self.timestamp.isoformat(), "metadata": self.metadata}


class EventManager:
    def __init__(self) -> None:
        self._callbacks: List[Callable[[FusionEvent], None]] = []

    def subscribe(self, callback: Callable[[FusionEvent], None]) -> None:
        self._callbacks.append(callback)

    def emit(self, event_type: str, entity_id: str | None = None, **metadata) -> FusionEvent:
        event = FusionEvent(event_type, entity_id, datetime.now(timezone.utc), metadata)
        for callback in tuple(self._callbacks):
            callback(event)
        return event
