from __future__ import annotations

import random
import re
from dataclasses import replace

from turngames.core.protocols import ActionSchema, RoleSpec
from turngames.core.timer import clip_to_word_budget, count_time_words
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
from turngames.core.visibility import VisibilityContext
from turngames.games.codewords.deck import DECKS
from turngames.games.codewords.types import (
    ARBITER,
    ASSASSIN,
    BLUE,
    CLUE_PROPOSAL,
    CLUEGIVER,
    CodewordsBoard,
    CodewordsConfig,
    CodewordsState,
    Clue,
    GAME_END,
    GUESSER,
    GUESSING,
    NEUTRAL,
    Proposal,
    RED,
)


class CodewordsSpec:
    """Codenames-like game spec built on the generic state machine."""

    id = "codewords"

    def initial_state(self, seed: str, config: CodewordsConfig) -> CodewordsState:
        word_count = config.width * config.height
        if config.words is None:
            deck = DECKS.get(config.deck)
            if deck is None:
                raise ValueError(f"unknown deck: {config.deck!r}")
            config = replace(config, words=deck.deal(seed, word_count))
        if len(config.words) != word_count:
            raise ValueError("config.words must contain width * height words")

        key = config.key or self._generate_key(seed, config)
        if len(key) != word_count:
            raise ValueError("config.key must contain width * height assignments")

        if config.guessers_per_team < 1:
            raise ValueError("config.guessers_per_team must be at least 1")
        labels = config.seat_labels or {}
        seats: list[Seat] = []
        for team in (RED, BLUE):
            cluegiver_id = f"{team}_cluegiver"
            seats.append(
                Seat(
                    id=cluegiver_id,
                    role=CLUEGIVER,
                    team_id=team,
                    label=labels.get(cluegiver_id),
                )
            )
            for index in range(config.guessers_per_team):
                guesser_id = f"{team}_guesser_{index + 1}"
                seats.append(
                    Seat(
                        id=guesser_id,
                        role=GUESSER,
                        team_id=team,
                        label=labels.get(guesser_id),
                    )
                )
        seats.append(Seat(id="canonical_arbiter", role=ARBITER, actor_kind="system"))
        seats = tuple(seats)
        return CodewordsState(
            seed=seed,
            config=config,
            board=CodewordsBoard.from_words_and_key(
                config.width,
                config.height,
                config.words,
                key,
            ),
            seats=seats,
            current_team=config.starting_team,
            time_left=config.turn_time_limit_words,
        )

    def role_specs(self) -> dict[str, RoleSpec]:
        return {
            CLUEGIVER: RoleSpec(
                id=CLUEGIVER,
                description="See the key and submit a legal one-word clue plus count.",
                action_schemas=(
                    ActionSchema(
                        type="say",
                        schema={},
                        description=(
                            "Think out loud about strategy before committing to a "
                            "clue. Spoken words count against the team's clock."
                        ),
                    ),
                    ActionSchema(
                        type="give_clue",
                        schema={"word": "string", "count": "integer"},
                        description="Give a single-word clue and target count.",
                    ),
                ),
            ),
            GUESSER: RoleSpec(
                id=GUESSER,
                description=(
                    "Discuss the clue with teammates, propose a tile, and confirm or "
                    "reject teammates' proposals. A tile is only revealed once a "
                    "different guesser confirms the proposal."
                ),
                action_schemas=(
                    ActionSchema(
                        type="say",
                        schema={},
                        description="Deliberate out loud without committing to anything.",
                    ),
                    ActionSchema(
                        type="propose",
                        schema={"word": "string"},
                        description="Propose one unrevealed board word to the team.",
                    ),
                    ActionSchema(
                        type="confirm",
                        schema={},
                        description=(
                            "Confirm the pending proposal; the tile is revealed and "
                            "you keep the floor to propose the next guess."
                        ),
                    ),
                    ActionSchema(
                        type="reject",
                        schema={},
                        description="Reject the pending proposal.",
                    ),
                    ActionSchema(
                        type="choose",
                        schema={"word": "string"},
                        description=(
                            "Directly choose one board word. Used by a lone guesser "
                            "instead of propose/confirm, or by any guesser once the "
                            "timer has expired."
                        ),
                    ),
                    ActionSchema(
                        type="stop",
                        schema={},
                        description="End the team's guessing turn.",
                    ),
                ),
            ),
            ARBITER: RoleSpec(
                id=ARBITER,
                description="Validate actions against rules and visibility policy.",
                action_schemas=(),
                public_deliberation=False,
            ),
        }

    def seats(self, state: CodewordsState) -> tuple[Seat, ...]:
        return state.seats

    def acting_seat(self, state: CodewordsState) -> str | None:
        if self.is_terminal(state):
            return None
        if state.phase == CLUE_PROPOSAL:
            return state.active_cluegiver_id()
        if state.phase == GUESSING:
            return state.active_guesser_id()
        return None

    def observe(self, state: CodewordsState, seat_id: str) -> Observation:
        seat = self._require_seat(state, seat_id)
        context = VisibilityContext(
            seat_id=seat.id,
            role=seat.role,
            team_id=seat.team_id,
            viewer="player",
        )
        data: dict[str, JsonValue] = self._shared_view(state)
        data["board"] = state.board.view_for(context)
        return Observation(
            seat_id=seat_id,
            phase=state.phase,
            data=data,
            allowed_actions=self.legal_actions(state, seat_id),
        )

    def observe_viewer(self, state: CodewordsState, viewer: str) -> dict[str, JsonValue]:
        context = VisibilityContext(viewer=viewer)
        data = self._shared_view(state)
        data["board"] = state.board.view_for(context)
        return data

    def _shared_view(self, state: CodewordsState) -> dict[str, JsonValue]:
        return {
            "game_id": self.id,
            "phase": state.phase,
            "current_team": state.current_team,
            "turn_number": state.turn_number,
            "current_clue": state.current_clue.to_json() if state.current_clue else None,
            "guesses_this_turn": state.guesses_this_turn,
            "winner": state.winner,
            "terminal_reason": state.terminal_reason,
            "time_left": state.time_left,
            "time_limit": state.config.turn_time_limit_words,
            "final_guess_only": state.final_guess_only,
            "bonus_guess": self._is_bonus_guess(state),
            "pending_proposal": (
                state.pending_proposal.to_json() if state.pending_proposal else None
            ),
            "acting_seat": self.acting_seat(state),
            "final_guess_seat": state.final_guess_seat,
            "seats": [
                {
                    "id": seat.id,
                    "role": seat.role,
                    "team_id": seat.team_id,
                    "label": seat.label,
                    "actor_kind": seat.actor_kind,
                }
                for seat in state.seats
                if seat.role in (CLUEGIVER, GUESSER)
            ],
        }

    def clip_public_deliberation(
        self,
        state: CodewordsState,
        seat_id: str,
        text: str,
    ) -> str:
        """Cut a speaker off at their team's remaining word clock.

        An over-long deliberation trails off with an ellipsis exactly where the
        clock runs out, instead of being spoken in full and then discarded. The
        clock still expires (the clipped text costs the last of it), which hands
        the single final word to the quiet teammate — but the moment now reads
        as trailing off mid-sentence rather than a jarring voided action.
        """
        if not text or self.is_terminal(state):
            return text
        seat = state.seat(seat_id)
        if seat is None or seat.team_id != state.current_team:
            return text
        budget = state.time_left
        # The bonus guess is a snap decision, not a debate: two spoken words,
        # tops — part of the anti-stall guarantee (see _is_bonus_guess).
        if self._is_bonus_guess(state):
            budget = 2 if budget is None else min(budget, 2)
        if budget is None:
            return text
        return clip_to_word_budget(text, budget)

    def consume_public_deliberation(
        self,
        state: CodewordsState,
        seat_id: str,
        public_deliberation: str,
    ) -> StepTransition:
        if state.time_left is None or self.is_terminal(state):
            return StepTransition(state)

        seat = state.seat(seat_id)
        if seat is None or seat.team_id != state.current_team:
            return StepTransition(state)

        spent = count_time_words(public_deliberation)
        before = state.time_left
        after = max(0, before - spent)
        expired = before > 0 and after == 0
        final_guess_only = state.final_guess_only
        spoken = dict(state.spoken_this_turn)
        spoken[seat_id] = spoken.get(seat_id, 0) + spent
        next_state = replace(state, time_left=after, spoken_this_turn=spoken)
        events = [
            EventRecord(
                "time_used",
                {
                    "seat_id": seat_id,
                    "team_id": seat.team_id,
                    "phase": state.phase,
                    "before": before,
                    "spent": spent,
                    "after": after,
                },
                visibility=("replay", "audience"),
            )
        ]
        final_guess_seat = state.final_guess_seat
        if state.phase == GUESSING and expired:
            final_guess_only = True
            # The final word goes to whoever wasn't burning the clock talking.
            final_guess_seat = next_state.quietest_guesser()
            events.append(
                EventRecord(
                    "timer_expired",
                    {
                        "seat_id": seat_id,
                        "team_id": seat.team_id,
                        "phase": state.phase,
                        "constraint": "final_guess_one_word",
                        "final_guess_seat": final_guess_seat,
                    },
                    visibility=("replay", "audience"),
                )
            )
        return StepTransition(
            replace(
                next_state,
                final_guess_only=final_guess_only,
                final_guess_seat=final_guess_seat,
                pending_proposal=None if final_guess_only else next_state.pending_proposal,
            ),
            tuple(events),
        )

    def legal_actions(
        self,
        state: CodewordsState,
        seat_id: str,
    ) -> tuple[dict[str, JsonValue], ...]:
        if self.is_terminal(state):
            return ()
        if state.phase == CLUE_PROPOSAL and seat_id == state.active_cluegiver_id():
            return (
                {"type": "say", "schema": {}},
                {
                    "type": "give_clue",
                    "schema": {
                        "word": "string",
                        "count": f"integer 0..{state.config.max_clue_count}",
                    },
                },
            )
        if state.phase == GUESSING and seat_id == state.active_guesser_id():
            if state.final_guess_only:
                return (
                    {
                        "type": "choose",
                        "schema": {"word": "single board word"},
                        "constraint": "final_guess_one_word",
                    },
                )
            if self._is_bonus_guess(state):
                # The free "+1" is a snap decision: no propose/reject cycles
                # to stall in — choose one more word or bank the clue.
                return (
                    {
                        "type": "choose",
                        "schema": {"word": "single board word"},
                        "constraint": "bonus_guess_two_words",
                    },
                    {"type": "stop", "schema": {}},
                )
            if len(state.guesser_ids()) == 1:
                return (
                    {"type": "say", "schema": {}},
                    {"type": "choose", "schema": {"word": "string"}},
                    {"type": "stop", "schema": {}},
                )
            if state.pending_proposal is not None:
                return (
                    {"type": "confirm", "schema": {}},
                    {"type": "reject", "schema": {}},
                    {"type": "say", "schema": {}},
                    {"type": "stop", "schema": {}},
                )
            return (
                {"type": "say", "schema": {}},
                {"type": "propose", "schema": {"word": "string"}},
                {"type": "stop", "schema": {}},
            )
        return ()

    def validate(self, state: CodewordsState, seat_id: str, action: Action) -> Ruling:
        expected = self.acting_seat(state)
        if expected is None:
            return Ruling.illegal("terminal_game", "The game is already terminal.")
        if seat_id != expected:
            return Ruling.illegal(
                "wrong_acting_seat",
                f"Expected {expected} to act, but received {seat_id}.",
            )
        if state.phase == CLUE_PROPOSAL:
            return self._validate_clue(state, action)
        if state.phase == GUESSING:
            return self._validate_guess(state, action)
        return Ruling.illegal("unknown_phase", f"Unknown phase: {state.phase}")

    def step(self, state: CodewordsState, seat_id: str, action: Action) -> StepTransition:
        if state.phase == CLUE_PROPOSAL:
            return self._step_clue(state, seat_id, action)
        if state.phase == GUESSING:
            return self._step_guess(state, seat_id, action)
        raise RuntimeError(f"cannot step phase {state.phase}")

    def handle_invalid(
        self,
        state: CodewordsState,
        seat_id: str,
        action: Action,
        ruling: Ruling,
    ) -> StepTransition:
        events = (
            EventRecord(
                "illegal_action_applied",
                {
                    "seat_id": seat_id,
                    "action_type": action.type,
                    "rule_id": ruling.rule_id,
                    "message": ruling.message,
                },
                visibility=("replay", "audience"),
            ),
        )
        rewards = (
            Reward(
                seat_id=seat_id,
                name="illegal_action",
                value=-1.0,
                metadata={"rule_id": ruling.rule_id},
            ),
        )
        if state.phase == CLUE_PROPOSAL and state.config.illegal_clue_ends_turn:
            penalties = dict(state.illegal_clues)
            penalties[state.current_team] = penalties.get(state.current_team, 0) + 1
            next_state = self._end_turn(replace(state, illegal_clues=penalties))
            return StepTransition(next_state, events, rewards)
        return StepTransition(state, events, rewards)

    def is_terminal(self, state: CodewordsState) -> bool:
        return state.phase == GAME_END or state.winner is not None

    def score(self, log: EventLog, state: CodewordsState) -> dict[str, JsonValue]:
        return {
            "winner": state.winner,
            "terminal_reason": state.terminal_reason,
            "turn_number": state.turn_number,
            "red_remaining": state.board.remaining_for_team(RED),
            "blue_remaining": state.board.remaining_for_team(BLUE),
            "illegal_clues": dict(state.illegal_clues),
            "event_count": len(log.events),
        }

    def _validate_clue(self, state: CodewordsState, action: Action) -> Ruling:
        if action.type == "say":
            return Ruling.legal("legal_say", "Cluegiver may think out loud.")
        if action.type != "give_clue":
            return Ruling.illegal("wrong_action_type", "Cluegiver must say or give a clue.")
        word = action.payload.get("word")
        count = action.payload.get("count")
        if not isinstance(word, str) or not word.strip():
            return Ruling.illegal("missing_clue_word", "Clue word must be a non-empty string.")
        clue_word = word.strip()
        if not re.fullmatch(r"[A-Za-z]+", clue_word):
            return Ruling.illegal(
                "clue_not_single_alpha_word",
                "Clue must be a single alphabetic word in this ruleset.",
            )
        if not isinstance(count, int):
            return Ruling.illegal("missing_clue_count", "Clue count must be an integer.")
        if count < 0 or count > state.config.max_clue_count:
            return Ruling.illegal(
                "clue_count_out_of_range",
                f"Clue count must be between 0 and {state.config.max_clue_count}.",
            )

        clue_norm = clue_word.casefold()
        for board_word in state.board.unrevealed_words():
            board_norm = board_word.casefold()
            if clue_norm == board_norm:
                return Ruling.illegal(
                    "clue_matches_unrevealed_word",
                    "Clue cannot exactly match an unrevealed board word.",
                )
            if clue_norm in board_norm or board_norm in clue_norm:
                return Ruling.illegal(
                    "clue_contains_unrevealed_word",
                    "Clue cannot contain or be contained by an unrevealed board word.",
                )
        return Ruling.legal("legal_clue", "Clue is legal.")

    def _validate_guess(self, state: CodewordsState, action: Action) -> Ruling:
        if state.final_guess_only:
            if action.type != "choose":
                return Ruling.illegal(
                    "final_guess_must_choose",
                    "Timer expired: guesser must choose one board word.",
                )
            word = action.payload.get("word")
            if not isinstance(word, str) or not word.strip():
                return Ruling.illegal("missing_guess_word", "Choose must include a word.")
            if count_time_words(word) != 1:
                return Ruling.illegal(
                    "final_guess_must_be_one_word",
                    "Timer expired: guesser may say only one word.",
                )
            return self._validate_board_word(state, word)

        if self._is_bonus_guess(state):
            if action.type == "stop":
                return Ruling.legal("legal_stop", "Guesser banks the clue.")
            if action.type == "choose":
                word = action.payload.get("word")
                if not isinstance(word, str) or not word.strip():
                    return Ruling.illegal("missing_guess_word", "Choose must include a word.")
                return self._validate_board_word(state, word)
            return Ruling.illegal(
                "bonus_guess_must_choose_or_stop",
                "Bonus guess: choose one more board word or stop.",
            )

        solo = len(state.guesser_ids()) == 1

        if action.type == "say":
            return Ruling.legal("legal_say", "Guesser may deliberate.")
        if action.type == "stop":
            return Ruling.legal("legal_stop", "Guesser may stop.")
        if action.type == "choose":
            if not solo:
                return Ruling.illegal(
                    "choose_requires_solo_guesser",
                    "Teams with multiple guessers must propose and confirm guesses.",
                )
            word = action.payload.get("word")
            if not isinstance(word, str) or not word.strip():
                return Ruling.illegal("missing_guess_word", "Choose must include a word.")
            return self._validate_board_word(state, word)
        if action.type == "propose":
            if solo:
                return Ruling.illegal(
                    "solo_guesser_must_choose",
                    "A lone guesser chooses directly instead of proposing.",
                )
            if state.pending_proposal is not None:
                return Ruling.illegal(
                    "proposal_already_pending",
                    "A proposal is already awaiting confirmation.",
                )
            word = action.payload.get("word")
            if not isinstance(word, str) or not word.strip():
                return Ruling.illegal("missing_guess_word", "Proposal must include a word.")
            return self._validate_board_word(state, word)
        if action.type in ("confirm", "reject"):
            if solo:
                return Ruling.illegal(
                    "solo_guesser_must_choose",
                    "A lone guesser chooses directly instead of confirming or rejecting.",
                )
            if state.pending_proposal is None:
                return Ruling.illegal(
                    "no_pending_proposal",
                    "There is no proposal to confirm or reject.",
                )
            return Ruling.legal(f"legal_{action.type}", f"Guesser may {action.type}.")
        return Ruling.illegal(
            "wrong_action_type",
            "Guesser must say, propose, confirm, reject, choose, or stop.",
        )

    def _validate_board_word(self, state: CodewordsState, word: str) -> Ruling:
        cell_id = state.board.cell_id_for_word(word)
        if cell_id is None:
            return Ruling.illegal("guess_not_on_board", "Guess must be a board word.")
        if cell_id in state.board.revealed:
            return Ruling.illegal("guess_already_revealed", "Guess must be unrevealed.")
        return Ruling.legal("legal_guess", "Guess is legal.")

    def _step_clue(
        self,
        state: CodewordsState,
        seat_id: str,
        action: Action,
    ) -> StepTransition:
        if action.type == "say":
            return StepTransition(
                state,
                (
                    EventRecord(
                        "cluegiver_said",
                        {"seat_id": seat_id, "team_id": state.current_team},
                        visibility=("replay", "audience"),
                    ),
                ),
            )
        clue = Clue(
            team_id=state.current_team,
            word=str(action.payload["word"]).strip(),
            count=int(action.payload["count"]),
        )
        expired_at_clue = state.time_left == 0
        next_state = replace(
            state,
            phase=GUESSING,
            current_clue=clue,
            guesses_this_turn=0,
            final_guess_only=expired_at_clue,
            final_guess_seat=state.quietest_guesser() if expired_at_clue else None,
            pending_proposal=None,
            guess_cursor=0,
        )
        events = (
            EventRecord(
                "clue_given",
                {"seat_id": seat_id, "clue": clue.to_json()},
                visibility=("replay", "audience"),
            ),
        )
        return StepTransition(next_state, events)

    def _step_guess(
        self,
        state: CodewordsState,
        seat_id: str,
        action: Action,
    ) -> StepTransition:
        guessers = state.guesser_ids()

        if action.type == "stop":
            next_state = self._end_turn(state)
            return StepTransition(
                next_state,
                (
                    EventRecord(
                        "guesser_stopped",
                        {"seat_id": seat_id, "team_id": state.current_team},
                        visibility=("replay", "audience"),
                    ),
                ),
            )

        if action.type == "say":
            next_state = state
            if state.pending_proposal is None and seat_id in guessers:
                next_state = replace(
                    state,
                    guess_cursor=(guessers.index(seat_id) + 1) % len(guessers),
                )
            return StepTransition(
                next_state,
                (
                    EventRecord(
                        "guesser_said",
                        {"seat_id": seat_id, "team_id": state.current_team},
                        visibility=("replay", "audience"),
                    ),
                ),
            )

        if action.type == "propose":
            word = str(action.payload["word"]).strip()
            return StepTransition(
                replace(state, pending_proposal=Proposal(seat_id=seat_id, word=word)),
                (
                    EventRecord(
                        "guess_proposed",
                        {
                            "seat_id": seat_id,
                            "team_id": state.current_team,
                            "word": word,
                        },
                        visibility=("replay", "audience"),
                    ),
                ),
            )

        if action.type == "reject":
            proposal = state.pending_proposal
            if proposal is None:
                raise RuntimeError("validated reject lost its proposal")
            return StepTransition(
                replace(
                    state,
                    pending_proposal=None,
                    guess_cursor=guessers.index(seat_id),
                ),
                (
                    EventRecord(
                        "guess_rejected",
                        {
                            "seat_id": seat_id,
                            "team_id": state.current_team,
                            "word": proposal.word,
                            "proposed_by": proposal.seat_id,
                        },
                        visibility=("replay", "audience"),
                    ),
                ),
            )

        if action.type == "confirm":
            proposal = state.pending_proposal
            if proposal is None:
                raise RuntimeError("validated confirm lost its proposal")
            base = replace(
                state,
                pending_proposal=None,
                guess_cursor=guessers.index(seat_id),
            )
            confirm_event = EventRecord(
                "guess_confirmed",
                {
                    "seat_id": seat_id,
                    "team_id": state.current_team,
                    "word": proposal.word,
                    "proposed_by": proposal.seat_id,
                },
                visibility=("replay", "audience"),
            )
            return self._reveal(base, proposal.seat_id, proposal.word, (confirm_event,))

        if action.type == "choose":
            word = str(action.payload["word"]).strip()
            return self._reveal(state, seat_id, word, ())

        raise RuntimeError(f"cannot step guess action {action.type}")

    def _reveal(
        self,
        state: CodewordsState,
        credited_seat: str,
        word: str,
        lead_events: tuple[EventRecord, ...],
    ) -> StepTransition:
        cell_id = state.board.cell_id_for_word(word)
        if cell_id is None:
            raise RuntimeError("validated guess lost its board cell")
        assignment = state.board.key[cell_id]
        board = state.board.reveal(cell_id)
        guesses_this_turn = state.guesses_this_turn + 1
        events: list[EventRecord] = [
            *lead_events,
            EventRecord(
                "word_revealed",
                {
                    "seat_id": credited_seat,
                    "team_id": state.current_team,
                    "word": state.board.word_for_cell(cell_id),
                    "cell_id": cell_id,
                    "assignment": assignment,
                },
                visibility=("replay", "audience"),
            ),
        ]
        rewards: list[Reward] = [
            Reward(
                seat_id=credited_seat,
                name="guess_assignment",
                value=self._guess_reward(state.current_team, assignment),
                metadata={"assignment": assignment},
            )
        ]

        next_state = replace(
            state,
            board=board,
            guesses_this_turn=guesses_this_turn,
        )

        if assignment == ASSASSIN:
            winner = self._other_team(state.current_team)
            return StepTransition(
                replace(
                    next_state,
                    phase=GAME_END,
                    winner=winner,
                    terminal_reason="assassin_revealed",
                ),
                tuple(events),
                tuple(rewards),
            )

        if board.remaining_for_team(state.current_team) == 0:
            return StepTransition(
                replace(
                    next_state,
                    phase=GAME_END,
                    winner=state.current_team,
                    terminal_reason="all_team_words_revealed",
                ),
                tuple(events),
                tuple(rewards),
            )

        if assignment == self._other_team(state.current_team) and board.remaining_for_team(assignment) == 0:
            return StepTransition(
                replace(
                    next_state,
                    phase=GAME_END,
                    winner=assignment,
                    terminal_reason="opponent_words_completed",
                ),
                tuple(events),
                tuple(rewards),
            )

        if assignment != state.current_team:
            next_state = self._end_turn(next_state)
            return StepTransition(next_state, tuple(events), tuple(rewards))

        if state.final_guess_only:
            next_state = self._end_turn(next_state)
            return StepTransition(next_state, tuple(events), tuple(rewards))

        clue_count = state.current_clue.count if state.current_clue else 0
        if clue_count > 0 and guesses_this_turn == clue_count:
            # The team just revealed the last card the clue paid for and still
            # holds the floor: the next guess is the free "+1" bonus guess.
            # Announce it so the audience and UI can mark the extra try.
            events.append(
                EventRecord(
                    "bonus_guess_reached",
                    {
                        "team_id": state.current_team,
                        "seat_id": credited_seat,
                        "clue_count": clue_count,
                    },
                    visibility=("replay", "audience"),
                )
            )

        if guesses_this_turn >= self._max_guesses_for_turn(state):
            # All guesses for this clue landed. If the clock still has time
            # and the team made more than one guess, the cluegiver keeps the
            # floor for a bonus clue instead of the turn ending.
            if (
                state.config.bonus_clue_when_time_left
                and guesses_this_turn > 1
                and next_state.time_left is not None
                and next_state.time_left > 0
            ):
                events.append(
                    EventRecord(
                        "bonus_clue_granted",
                        {
                            "team_id": state.current_team,
                            "cluegiver": state.active_cluegiver_id(),
                            "time_left": next_state.time_left,
                        },
                        visibility=("replay", "audience"),
                    )
                )
                next_state = replace(
                    next_state,
                    phase=CLUE_PROPOSAL,
                    current_clue=None,
                    guesses_this_turn=0,
                    pending_proposal=None,
                    guess_cursor=0,
                )
            else:
                next_state = self._end_turn(next_state)

        return StepTransition(next_state, tuple(events), tuple(rewards))

    def _end_turn(self, state: CodewordsState) -> CodewordsState:
        return replace(
            state,
            phase=CLUE_PROPOSAL,
            current_team=self._other_team(state.current_team),
            turn_number=state.turn_number + 1,
            current_clue=None,
            guesses_this_turn=0,
            time_left=state.config.turn_time_limit_words,
            final_guess_only=False,
            pending_proposal=None,
            guess_cursor=0,
            spoken_this_turn={},
            final_guess_seat=None,
        )

    def _is_bonus_guess(self, state: CodewordsState) -> bool:
        """The free '+1' try after clearing the clue's count."""
        return (
            state.phase == GUESSING
            and not state.final_guess_only
            and state.current_clue is not None
            and state.current_clue.count > 0
            and state.guesses_this_turn == state.current_clue.count
        )

    def _max_guesses_for_turn(self, state: CodewordsState) -> int:
        if state.current_clue is None:
            return 0
        if state.current_clue.count == 0:
            return state.config.width * state.config.height
        return state.current_clue.count + 1

    def _require_seat(self, state: CodewordsState, seat_id: str) -> Seat:
        seat = state.seat(seat_id)
        if seat is None:
            raise KeyError(f"unknown seat: {seat_id}")
        return seat

    def _other_team(self, team_id: str) -> str:
        return BLUE if team_id == RED else RED

    def _guess_reward(self, team_id: str, assignment: str) -> float:
        if assignment == team_id:
            return 1.0
        if assignment == ASSASSIN:
            return -10.0
        if assignment == NEUTRAL:
            return -0.25
        return -1.0

    def _generate_key(self, seed: str, config: CodewordsConfig) -> tuple[str, ...]:
        red_count, blue_count = (9, 8) if config.starting_team == RED else (8, 9)
        neutral_count = config.width * config.height - red_count - blue_count - 1
        assignments = [RED] * red_count
        assignments += [BLUE] * blue_count
        assignments += [NEUTRAL] * neutral_count
        assignments += [ASSASSIN]
        rng = random.Random(f"codewords-key:{seed}")
        rng.shuffle(assignments)
        return tuple(assignments)
