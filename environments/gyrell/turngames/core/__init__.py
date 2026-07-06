"""Core interfaces for turn-based games."""

from turngames.core.board import BoardCell, BoardLayer, BoardView, GridBoard
from turngames.core.protocols import ActionSchema, ActorInput, ActorOutput, RoleSpec
from turngames.core.state_machine import GameSpec, StateMachine
from turngames.core.timer import TimeSpend, clip_to_word_budget, count_time_words
from turngames.core.types import (
    Action,
    Event,
    EventLog,
    EventRecord,
    Observation,
    Reward,
    Ruling,
    Seat,
    StepTransition,
)
from turngames.core.visibility import VisibilityContext, VisibilityRule

__all__ = [
    "Action",
    "ActionSchema",
    "ActorInput",
    "ActorOutput",
    "BoardCell",
    "BoardLayer",
    "BoardView",
    "Event",
    "EventLog",
    "EventRecord",
    "GameSpec",
    "GridBoard",
    "Observation",
    "Reward",
    "RoleSpec",
    "Ruling",
    "Seat",
    "StateMachine",
    "StepTransition",
    "TimeSpend",
    "VisibilityContext",
    "VisibilityRule",
    "clip_to_word_budget",
    "count_time_words",
]
