from __future__ import annotations

from typing import Generic, Protocol, TypeVar

from turngames.core.types import (
    Action,
    EventLog,
    EventRecord,
    JsonValue,
    Observation,
    Reward,
    Ruling,
    Seat,
    StepTransition,
)

StateT = TypeVar("StateT")
ConfigT = TypeVar("ConfigT")


class GameSpec(Protocol[StateT, ConfigT]):
    """Protocol implemented by concrete turn-based games."""

    id: str

    def initial_state(self, seed: str, config: ConfigT) -> StateT:
        ...

    def seats(self, state: StateT) -> tuple[Seat, ...]:
        ...

    def acting_seat(self, state: StateT) -> str | None:
        ...

    def observe(self, state: StateT, seat_id: str) -> Observation:
        ...

    def legal_actions(self, state: StateT, seat_id: str) -> tuple[dict[str, JsonValue], ...]:
        ...

    def validate(self, state: StateT, seat_id: str, action: Action) -> Ruling:
        ...

    def step(self, state: StateT, seat_id: str, action: Action) -> StepTransition:
        ...

    def handle_invalid(
        self,
        state: StateT,
        seat_id: str,
        action: Action,
        ruling: Ruling,
    ) -> StepTransition:
        ...

    def is_terminal(self, state: StateT) -> bool:
        ...

    def score(self, log: EventLog, state: StateT) -> dict[str, JsonValue]:
        ...


class StateMachine(Generic[StateT, ConfigT]):
    """Generic executor for a GameSpec.

    The state machine owns event sequencing and delegates all game-specific
    legality, transition, and scoring behavior to the spec.
    """

    def __init__(
        self,
        spec: GameSpec[StateT, ConfigT],
        state: StateT,
        log: EventLog | None = None,
    ) -> None:
        self.spec = spec
        self.state = state
        self.log = log or EventLog()

    @classmethod
    def new(
        cls,
        spec: GameSpec[StateT, ConfigT],
        seed: str,
        config: ConfigT,
    ) -> "StateMachine[StateT, ConfigT]":
        state = spec.initial_state(seed, config)
        machine = cls(spec=spec, state=state)
        machine.log.append(
            "game_initialized",
            {"game_id": spec.id, "seed": seed},
            visibility=("replay", "audience"),
        )
        return machine

    def observe(self, seat_id: str) -> Observation:
        observation = self.spec.observe(self.state, seat_id)
        self.log.append(
            "observation_created",
            {"seat_id": seat_id, "phase": observation.phase},
            visibility=("replay",),
        )
        return observation

    def submit(
        self,
        seat_id: str,
        action: Action,
        public_deliberation: str | None = None,
        raw_response: str | None = None,
    ) -> tuple[Ruling, tuple[Reward, ...]]:
        if self.spec.is_terminal(self.state):
            raise RuntimeError("cannot submit action to a terminal game")

        # Give the spec a chance to cut the spoken text off where a running
        # clock (if any) expires, so an over-long deliberation trails off with
        # an ellipsis instead of being shown in full and then voided. The
        # clipped text is what we record and what the clock spends below.
        clip_public_deliberation = getattr(self.spec, "clip_public_deliberation", None)
        if clip_public_deliberation is not None and public_deliberation:
            public_deliberation = clip_public_deliberation(
                self.state,
                seat_id,
                public_deliberation,
            )

        self.log.append(
            "actor_responded",
            {
                "seat_id": seat_id,
                "action": {"type": action.type, "payload": action.payload},
                "public_deliberation": public_deliberation,
                "raw_response": raw_response,
            },
            visibility=("replay", "audience"),
        )

        setup_rewards: tuple[Reward, ...] = ()
        consume_public_deliberation = getattr(
            self.spec,
            "consume_public_deliberation",
            None,
        )
        if consume_public_deliberation is not None:
            was_acting = self.spec.acting_seat(self.state) == seat_id
            setup_transition = consume_public_deliberation(
                self.state,
                seat_id,
                public_deliberation or "",
            )
            setup_rewards = setup_transition.rewards
            self._apply_transition(setup_transition)

            # If the speaker's own deliberation ran out the clock, the right
            # to act may have moved to another seat mid-submit. That is not an
            # infraction — void the action without penalty and let the new
            # acting seat take over. (A seat that was never the acting seat
            # still gets the ordinary wrong_acting_seat ruling below.)
            expected = self.spec.acting_seat(self.state)
            if was_acting and expected is not None and expected != seat_id:
                ruling = Ruling.illegal(
                    "time_expired_while_talking",
                    "The clock ran out mid-speech; the action is void.",
                    severity="info",
                )
                # Replay-only: the audience already sees the speech trail off
                # (clipped with an ellipsis) and the "Time!" hand-off to the
                # quiet teammate, so surfacing a separate "action void" line
                # here would just read as a jarring, redundant penalty.
                self.log.append(
                    "action_voided",
                    {
                        "seat_id": seat_id,
                        "action_type": action.type,
                        "rule_id": ruling.rule_id,
                    },
                    visibility=("replay",),
                )
                return ruling, setup_rewards

        ruling = self.spec.validate(self.state, seat_id, action)
        self.log.append(
            "ruling_issued",
            {
                "seat_id": seat_id,
                "is_legal": ruling.is_legal,
                "rule_id": ruling.rule_id,
                "message": ruling.message,
                "severity": ruling.severity,
            },
            visibility=("replay", "audience"),
        )

        transition = (
            self.spec.step(self.state, seat_id, action)
            if ruling.is_legal
            else self.spec.handle_invalid(self.state, seat_id, action, ruling)
        )
        self._apply_transition(transition)
        if self.spec.is_terminal(self.state):
            self.log.append(
                "game_ended",
                self.spec.score(self.log, self.state),
                visibility=("replay", "audience"),
            )
        return ruling, setup_rewards + transition.rewards

    def score(self) -> dict[str, JsonValue]:
        return self.spec.score(self.log, self.state)

    def _apply_transition(self, transition: StepTransition) -> None:
        self.state = transition.state
        self.log.extend(transition.events)
        for reward in transition.rewards:
            self.log.append(
                "reward_assigned",
                {
                    "seat_id": reward.seat_id,
                    "name": reward.name,
                    "value": reward.value,
                    "metadata": reward.metadata,
                },
                visibility=("replay",),
            )
