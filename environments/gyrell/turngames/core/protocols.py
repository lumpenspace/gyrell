from __future__ import annotations

from dataclasses import dataclass, field

from turngames.core.types import Action, JsonValue, Observation, Seat


@dataclass(frozen=True)
class ActionSchema:
    """Role-level action contract.

    This is intentionally lightweight so it can drive LLM prompts, human UI,
    and RL action spaces without committing to one validator library yet.
    """

    type: str
    schema: dict[str, JsonValue]
    description: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "type": self.type,
            "schema": self.schema,
            "description": self.description,
        }


@dataclass(frozen=True)
class RoleSpec:
    """Protocol for any actor occupying a role."""

    id: str
    description: str
    action_schemas: tuple[ActionSchema, ...]
    response_format: str = "json"
    public_deliberation: bool = True

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "description": self.description,
            "action_schemas": [schema.to_json() for schema in self.action_schemas],
            "response_format": self.response_format,
            "public_deliberation": self.public_deliberation,
        }


@dataclass(frozen=True)
class ActorInput:
    """Everything an adapter gives an actor before it speaks."""

    seat: Seat
    role: RoleSpec
    observation: Observation
    time_left: int | None = None
    constraints: dict[str, JsonValue] = field(default_factory=dict)
    # Recent audience-visible table talk (everything was said aloud on
    # stage), oldest first. Actors are otherwise stateless between turns.
    transcript: tuple[str, ...] = ()

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "seat": {
                "id": self.seat.id,
                "role": self.seat.role,
                "actor_kind": self.seat.actor_kind,
                "team_id": self.seat.team_id,
                "label": self.seat.label,
            },
            "role": self.role.to_json(),
            "observation": {
                "seat_id": self.observation.seat_id,
                "phase": self.observation.phase,
                "data": self.observation.data,
                "allowed_actions": list(self.observation.allowed_actions),
            },
            "time_left": self.time_left,
            "constraints": self.constraints,
            "transcript": list(self.transcript),
        }


@dataclass(frozen=True)
class ActorOutput:
    """Structured actor response after parsing.

    `reasoning` carries a reasoning model's hidden thinking trace. It is
    diagnostic sidecar material only — never game state, never broadcast,
    and kept out of replays so RL rollouts can't train on it.
    """

    public_deliberation: str
    action: Action
    raw_response: str | None = None
    reasoning: str | None = None
    # A short model-authored note to its future self, fed back on the seat's
    # next turn. A structured summary, deliberately NOT the raw reasoning
    # trace: replaying full CoT makes models resume stale conclusions.
    notes: str | None = None

