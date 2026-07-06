from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class Seat:
    """A role instance occupied by a human, model, system process, or script."""

    id: str
    role: str
    actor_kind: str = "llm"
    team_id: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class Action:
    """A structured move submitted by a seat."""

    type: str
    payload: dict[str, JsonValue] = field(default_factory=dict)

    @classmethod
    def of(cls, action_type: str, **payload: JsonValue) -> "Action":
        return cls(type=action_type, payload=payload)


@dataclass(frozen=True)
class Ruling:
    """Validation result for an action."""

    is_legal: bool
    rule_id: str
    message: str
    severity: str = "info"

    @classmethod
    def legal(cls, rule_id: str = "legal", message: str = "Action is legal.") -> "Ruling":
        return cls(is_legal=True, rule_id=rule_id, message=message)

    @classmethod
    def illegal(
        cls,
        rule_id: str,
        message: str,
        severity: str = "error",
    ) -> "Ruling":
        return cls(is_legal=False, rule_id=rule_id, message=message, severity=severity)


@dataclass(frozen=True)
class Observation:
    """Role-specific projection of state."""

    seat_id: str
    phase: str
    data: dict[str, JsonValue]
    allowed_actions: tuple[dict[str, JsonValue], ...] = ()


@dataclass(frozen=True)
class EventRecord:
    """Event before the event log assigns a sequence number."""

    type: str
    payload: dict[str, JsonValue] = field(default_factory=dict)
    visibility: tuple[str, ...] = ("replay",)


@dataclass(frozen=True)
class Event:
    """Append-only event with a stable sequence number."""

    seq: int
    type: str
    payload: dict[str, JsonValue]
    visibility: tuple[str, ...] = ("replay",)


@dataclass
class EventLog:
    """Append-only event log for gameplay, replay, and evaluation."""

    events: list[Event] = field(default_factory=list)

    def append(
        self,
        event_type: str,
        payload: dict[str, JsonValue] | None = None,
        visibility: tuple[str, ...] = ("replay",),
    ) -> Event:
        event = Event(
            seq=len(self.events),
            type=event_type,
            payload=payload or {},
            visibility=visibility,
        )
        self.events.append(event)
        return event

    def extend(self, records: tuple[EventRecord, ...] | list[EventRecord]) -> tuple[Event, ...]:
        return tuple(
            self.append(record.type, record.payload, record.visibility)
            for record in records
        )


@dataclass(frozen=True)
class Reward:
    """Reward assigned during rollout or postgame scoring."""

    seat_id: str | None
    name: str
    value: float
    metadata: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class StepTransition:
    """Game-specific transition output."""

    state: Any
    events: tuple[EventRecord, ...] = ()
    rewards: tuple[Reward, ...] = ()

