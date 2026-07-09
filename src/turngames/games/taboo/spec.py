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
from turngames.games.taboo.deck import DECKS
from turngames.games.taboo.types import (
    ARBITER,
    AWAIT_DESCRIBE,
    AWAIT_GUESS,
    BLUE,
    DESCRIBING,
    GAME_END,
    PLAYER,
    RED,
    TabooCard,
    TabooConfig,
    TabooState,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def utterance_says_target(text: str, target: str) -> bool:
    """Tabletop rule: the target crossing a guesser's lips counts as guessing
    it, formal action or not. Derivatives count ("astronauts" says
    ASTRONAUT); fragments don't."""
    if not text:
        return False
    normalized = target.casefold()
    return any(
        token == normalized or normalized in token
        for token in _TOKEN_RE.findall(text.casefold())
    )


# Tabletop taboo also bans clueing the word's FORM: spelling, letter counts,
# initials, rhymes. Patterns are deliberately narrow — "delivers letters" or
# "my initial thought" must not buzz.
_SPELLING_HINT_RES = (
    re.compile(r"\b(?:rhymes?\s+with|sounds?\s+like)\b", re.IGNORECASE),
    re.compile(r"\b(?:starts?|begins?|ends?)\s+with\b", re.IGNORECASE),
    re.compile(r"\b(?:first|last|initial|middle)\s+letters?\b", re.IGNORECASE),
    re.compile(r"\binitials\b", re.IGNORECASE),
    re.compile(
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
        r"[-\s]+letters?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bletter\s+['\"]?[a-z]['\"]?\b", re.IGNORECASE),
    re.compile(r"\bspell(?:s|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\babbreviat", re.IGNORECASE),
)


def find_spelling_hint(text: str) -> str | None:
    """Return the offending phrase if the description clues the word's form
    (spelling, length, initials, rhyme) instead of its meaning."""
    if not text:
        return None
    for pattern in _SPELLING_HINT_RES:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def find_taboo_word(text: str, card: TabooCard) -> str | None:
    """Return the banned word the text trips over, or None if it is clean.

    Mirrors the tabletop ruling: no form or part of the TARGET word in either
    direction (saying "snow" for SNOWMAN is a buzz), and no forbidden word or
    a derivative of it ("waves" trips "wave"). Forbidden words are not matched
    inside-out — saying "the" never trips "theater". Multi-word entries match
    as phrases.
    """
    if not text:
        return None
    tokens = _TOKEN_RE.findall(text.casefold())
    joined = " ".join(tokens)
    target = card.target.casefold()
    for token in tokens:
        if token == target or target in token or (len(token) >= 4 and token in target):
            return card.target
    for banned in card.forbidden:
        normalized = banned.casefold()
        if " " in normalized:
            if normalized in joined:
                return banned
            continue
        for token in tokens:
            if token == normalized or (len(normalized) >= 3 and normalized in token):
                return banned
    return None


class TabooSpec:
    """Taboo built on the generic state machine.

    Six players, no fixed game master: the card rotates through the
    describing team round by round. The description is the public
    deliberation itself — consume_public_deliberation stashes it on the
    state, and validate() rules it a taboo_violation if it trips the card.
    Every participant has a personal word clock per round; the round ends
    when the describer's runs dry.
    """

    id = "taboo"

    def initial_state(self, seed: str, config: TabooConfig) -> TabooState:
        if config.cards is None:
            deck = DECKS.get(config.deck)
            if deck is None:
                raise ValueError(f"unknown deck: {config.deck!r}")
            config = replace(config, cards=deck.cards)
        if not config.cards:
            raise ValueError("config.cards must not be empty")
        if config.words_per_participant <= 0:
            raise ValueError("taboo requires a positive words_per_participant clock")
        if config.players_per_team < 2:
            raise ValueError("config.players_per_team must be at least 2")
        if config.rounds_per_team < 1:
            raise ValueError("config.rounds_per_team must be at least 1")

        order = list(range(len(config.cards)))
        rng = random.Random(f"{seed}:taboo-deal")
        rng.shuffle(order)

        labels = config.seat_labels or {}
        seats: list[Seat] = []
        for team in (RED, BLUE):
            for index in range(config.players_per_team):
                player_id = f"{team}_player_{index + 1}"
                seats.append(
                    Seat(
                        id=player_id,
                        role=PLAYER,
                        team_id=team,
                        label=labels.get(player_id),
                    )
                )
        seats.append(Seat(id="canonical_arbiter", role=ARBITER, actor_kind="system"))
        state = TabooState(
            seed=seed,
            config=config,
            seats=tuple(seats),
            deal_order=tuple(order),
            current_team=config.starting_team,
        )
        return replace(state, speech_left=self._fresh_clocks(state))

    def _fresh_clocks(self, state: TabooState) -> dict[str, int]:
        return {
            player: state.config.words_per_participant for player in state.players()
        }

    def role_specs(self) -> dict[str, RoleSpec]:
        return {
            PLAYER: RoleSpec(
                id=PLAYER,
                description=(
                    "A taboo player. The card rotates: each round one player "
                    "on the describing team holds it and talks the team "
                    "toward the target — saying the target or any forbidden "
                    "word (or a form of them) is a buzz, and so are spelling "
                    "clues (letter counts, 'starts with', rhymes, initials). "
                    "A buzz is a turnover: point to the other team AND the "
                    "round ends on the spot, floor across. The card only "
                    "binds its holder: guessers may say anything, and the "
                    "target crossing a guesser's lips counts even "
                    "mid-sentence. Everyone has a personal word clock per "
                    "round; the round also ends when the describer's runs "
                    "dry."
                ),
                action_schemas=(
                    ActionSchema(
                        type="describe",
                        schema={},
                        description=(
                            "Holding the card: speak your description aloud "
                            "(in public_deliberation) and hand the floor to "
                            "a guesser."
                        ),
                    ),
                    ActionSchema(
                        type="skip",
                        schema={},
                        description="Holding the card: discard it unscored and draw the next.",
                    ),
                    ActionSchema(
                        type="guess",
                        schema={"word": "string"},
                        description="Guessing: name the target with exactly one word.",
                    ),
                    ActionSchema(
                        type="pass",
                        schema={},
                        description=(
                            "Guessing: think out loud (or stay quiet) and "
                            "give the floor back to the describer."
                        ),
                    ),
                ),
            ),
            ARBITER: RoleSpec(
                id=ARBITER,
                description="Buzzer: rules descriptions against the card automatically.",
                action_schemas=(),
                public_deliberation=False,
            ),
        }

    def seats(self, state: TabooState) -> tuple[Seat, ...]:
        return state.seats

    def acting_seat(self, state: TabooState) -> str | None:
        if self.is_terminal(state):
            return None
        if state.awaiting == AWAIT_DESCRIBE:
            return state.active_describer_id()
        return state.active_guesser_id()

    def observe(self, state: TabooState, seat_id: str) -> Observation:
        seat = state.seat(seat_id)
        if seat is None:
            raise KeyError(f"unknown seat: {seat_id}")
        data = self._shared_view(state)
        holds_card = seat.role == ARBITER or (
            not self.is_terminal(state) and seat_id == state.active_describer_id()
        )
        if holds_card and not self.is_terminal(state):
            data["current_card"] = state.current_card().to_json()
        return Observation(
            seat_id=seat_id,
            phase=state.phase,
            data=data,
            allowed_actions=self.legal_actions(state, seat_id),
        )

    def observe_viewer(self, state: TabooState, viewer: str) -> dict[str, JsonValue]:
        data = self._shared_view(state)
        if viewer in ("audience_omniscient", "replay") and not self.is_terminal(state):
            data["current_card"] = state.current_card().to_json()
        return data

    def _shared_view(self, state: TabooState) -> dict[str, JsonValue]:
        terminal = self.is_terminal(state)
        describer = None if terminal else state.active_describer_id()
        return {
            "game_id": self.id,
            "phase": state.phase,
            "current_team": state.current_team,
            "round_number": state.round_number,
            "total_rounds": state.total_rounds(),
            # Mirrors round_number so game-agnostic UI (turn flash, dial) works.
            "turn_number": state.round_number,
            "points": dict(state.points),
            "violations": dict(state.violations),
            "cards_played": state.cards_played,
            "skips_this_round": state.skips_this_round,
            "max_skips_per_round": state.config.max_skips_per_round,
            "current_hints": list(state.current_hints),
            "wrong_guesses_this_card": list(state.wrong_guesses),
            "awaiting": state.awaiting,
            "describer": describer,
            "winner": state.winner,
            "terminal_reason": state.terminal_reason,
            # The dial tracks the round's engine: the describer's own clock.
            "time_left": None if describer is None else state.speech_left.get(describer),
            "time_limit": state.config.words_per_participant,
            "speech_left": dict(state.speech_left),
            "acting_seat": self.acting_seat(state),
            "seats": [
                {
                    "id": seat.id,
                    "role": seat.role,
                    "team_id": seat.team_id,
                    "label": seat.label,
                    "actor_kind": seat.actor_kind,
                }
                for seat in state.seats
                if seat.role == PLAYER
            ],
        }

    def clip_public_deliberation(
        self,
        state: TabooState,
        seat_id: str,
        text: str,
    ) -> str:
        if not text or self.is_terminal(state):
            return text
        budget = state.speech_left.get(seat_id)
        if budget is None:
            return text
        return clip_to_word_budget(text, budget)

    def consume_public_deliberation(
        self,
        state: TabooState,
        seat_id: str,
        public_deliberation: str,
    ) -> StepTransition:
        # Always stash the utterance: validate()/step() rule on what was said.
        state = replace(state, last_utterance=public_deliberation)
        if self.is_terminal(state) or seat_id not in state.speech_left:
            return StepTransition(state)

        spent = count_time_words(public_deliberation)
        before = state.speech_left[seat_id]
        after = max(0, before - spent)
        clocks = dict(state.speech_left)
        clocks[seat_id] = after
        next_state = replace(state, speech_left=clocks)
        events = [
            EventRecord(
                "time_used",
                {
                    "seat_id": seat_id,
                    "team_id": state.current_team,
                    "before": before,
                    "spent": spent,
                    "after": after,
                },
                visibility=("replay", "audience"),
            )
        ]
        # Only the describer's clock ends the round; a spent guesser just
        # goes quiet (their speech clips to nothing from here on).
        if seat_id == state.active_describer_id() and before > 0 and after == 0:
            events.append(
                EventRecord(
                    "timer_expired",
                    {
                        "seat_id": seat_id,
                        "team_id": state.current_team,
                        "round_number": state.round_number,
                    },
                    visibility=("replay", "audience"),
                )
            )
            return self._round_end_transition(next_state, events)
        return StepTransition(next_state, tuple(events))

    def legal_actions(
        self,
        state: TabooState,
        seat_id: str,
    ) -> tuple[dict[str, JsonValue], ...]:
        if self.is_terminal(state) or seat_id != self.acting_seat(state):
            return ()
        if state.awaiting == AWAIT_DESCRIBE:
            actions: list[dict[str, JsonValue]] = [{"type": "describe", "schema": {}}]
            if state.skips_this_round < state.config.max_skips_per_round:
                actions.append({"type": "skip", "schema": {}})
            return tuple(actions)
        return (
            {"type": "guess", "schema": {"word": "single word"}},
            {"type": "pass", "schema": {}},
        )

    def validate(self, state: TabooState, seat_id: str, action: Action) -> Ruling:
        expected = self.acting_seat(state)
        if expected is None:
            return Ruling.illegal("terminal_game", "The game is already terminal.")
        if seat_id != expected:
            return Ruling.illegal(
                "wrong_acting_seat",
                f"Expected {expected} to act, but received {seat_id}.",
            )
        if state.awaiting == AWAIT_DESCRIBE:
            if action.type == "describe":
                tripped = find_taboo_word(state.last_utterance, state.current_card())
                # A buzz is severity "foul": legal-shaped play the rules punish,
                # not a malformed response — runners must not treat it as a
                # sign of a stuck model (see the illegal-streak fallbacks).
                if tripped is not None:
                    return Ruling.illegal(
                        "taboo_violation",
                        f"BUZZ — said a form of '{tripped}'.",
                        severity="foul",
                    )
                hint = find_spelling_hint(state.last_utterance)
                if hint is not None:
                    return Ruling.illegal(
                        "spelling_hint",
                        f"BUZZ — no spelling clues ('{hint}').",
                        severity="foul",
                    )
                return Ruling.legal("legal_describe", "Description is clean.")
            if action.type == "skip":
                if state.skips_this_round >= state.config.max_skips_per_round:
                    return Ruling.illegal(
                        "skip_limit_reached",
                        f"Only {state.config.max_skips_per_round} skips per round.",
                    )
                return Ruling.legal("legal_skip", "Card skipped.")
            return Ruling.illegal(
                "wrong_action_type", "The card holder must describe or skip."
            )
        if action.type == "guess":
            word = action.payload.get("word")
            if not isinstance(word, str) or not word.strip():
                return Ruling.illegal("missing_guess_word", "Guess must include a word.")
            if count_time_words(word) != 1:
                return Ruling.illegal(
                    "guess_must_be_one_word", "A guess is exactly one word."
                )
            return Ruling.legal("legal_guess", "Guess is legal.")
        if action.type == "pass":
            return Ruling.legal("legal_pass", "Guesser passes the floor back.")
        return Ruling.illegal("wrong_action_type", "Guesser must guess or pass.")

    def step(self, state: TabooState, seat_id: str, action: Action) -> StepTransition:
        if state.awaiting == AWAIT_DESCRIBE:
            if action.type == "describe":
                hints = state.current_hints
                if state.last_utterance.strip():
                    hints = (*hints, state.last_utterance.strip())
                return StepTransition(
                    replace(state, current_hints=hints, awaiting=AWAIT_GUESS),
                    (
                        EventRecord(
                            "hint_given",
                            {"seat_id": seat_id, "team_id": state.current_team},
                            visibility=("replay", "audience"),
                        ),
                    ),
                )
            if action.type == "skip":
                card = state.current_card()
                return StepTransition(
                    replace(
                        self._discard_card(state),
                        skips_this_round=state.skips_this_round + 1,
                    ),
                    (
                        EventRecord(
                            "card_skipped",
                            {
                                "seat_id": seat_id,
                                "team_id": state.current_team,
                                "card": card.to_json(),
                                "skips_left": state.config.max_skips_per_round
                                - state.skips_this_round
                                - 1,
                            },
                            visibility=("replay", "audience"),
                        ),
                    ),
                )
            raise RuntimeError(f"cannot step describe action {action.type}")

        guessers = state.guesser_ids()
        next_cursor = (state.guess_cursor + 1) % len(guessers)
        card = state.current_card()
        # Tabletop rule: saying the word IS guessing it — anything the guesser
        # spoke aloud this turn is checked against the target, formal guess or
        # not. (Hints can never contain the target: the describer would have
        # been buzzed, and clock-clipped speech only counts what was said.)
        spoken_hit = utterance_says_target(state.last_utterance, card.target)
        if action.type == "pass":
            if spoken_hit:
                return self._card_won(state, seat_id, card, next_cursor, via="spoken")
            return StepTransition(
                replace(state, awaiting=AWAIT_DESCRIBE, guess_cursor=next_cursor),
                (
                    EventRecord(
                        "guesser_passed",
                        {"seat_id": seat_id, "team_id": state.current_team},
                        visibility=("replay", "audience"),
                    ),
                ),
            )
        if action.type == "guess":
            word = str(action.payload["word"]).strip()
            if word.casefold() == card.target.casefold():
                return self._card_won(state, seat_id, card, next_cursor, via="guess")
            if spoken_hit:
                return self._card_won(state, seat_id, card, next_cursor, via="spoken")
            return StepTransition(
                replace(
                    state,
                    awaiting=AWAIT_DESCRIBE,
                    guess_cursor=next_cursor,
                    wrong_guesses=(*state.wrong_guesses, word),
                ),
                (
                    EventRecord(
                        "guess_incorrect",
                        {
                            "seat_id": seat_id,
                            "team_id": state.current_team,
                            "word": word,
                        },
                        visibility=("replay", "audience"),
                    ),
                ),
            )
        raise RuntimeError(f"cannot step guess action {action.type}")

    def _card_won(
        self,
        state: TabooState,
        seat_id: str,
        card: TabooCard,
        next_cursor: int,
        via: str,
    ) -> StepTransition:
        points = dict(state.points)
        points[state.current_team] += 1
        next_state = replace(
            self._discard_card(state),
            points=points,
            guess_cursor=next_cursor,
        )
        events = (
            EventRecord(
                "card_guessed",
                {
                    "seat_id": seat_id,
                    "team_id": state.current_team,
                    "card": card.to_json(),
                    "points": dict(points),
                    "via": via,
                },
                visibility=("replay", "audience"),
            ),
        )
        rewards = (
            Reward(seat_id=seat_id, name="card_guessed", value=1.0),
            Reward(
                seat_id=state.active_describer_id(),
                name="card_described",
                value=1.0,
            ),
        )
        return StepTransition(next_state, events, rewards)

    def handle_invalid(
        self,
        state: TabooState,
        seat_id: str,
        action: Action,
        ruling: Ruling,
    ) -> StepTransition:
        if ruling.rule_id == "terminal_game":
            # The clock died (ending the game) inside the same submit that
            # carried this action; nothing to punish.
            return StepTransition(state)
        if ruling.rule_id in ("taboo_violation", "spelling_hint"):
            # Both are a buzz: card lost, point across, foul on the record —
            # and a TURNOVER: the fouling team's round ends on the spot and
            # the floor crosses to the other team.
            card = state.current_card()
            tripped = (
                find_taboo_word(state.last_utterance, card)
                if ruling.rule_id == "taboo_violation"
                else find_spelling_hint(state.last_utterance)
            )
            points = dict(state.points)
            other = self._other_team(state.current_team)
            if state.config.violation_point_to_opponent:
                points[other] += 1
            violations = dict(state.violations)
            violations[state.current_team] += 1
            state = replace(state, points=points, violations=violations)
            events = [
                EventRecord(
                    "taboo_violation",
                    {
                        "seat_id": seat_id,
                        "team_id": state.current_team,
                        "said": tripped,
                        "card": card.to_json(),
                        "points": dict(points),
                    },
                    visibility=("replay", "audience"),
                ),
            ]
            rewards = (
                Reward(
                    seat_id=seat_id,
                    name="taboo_violation",
                    value=-1.0,
                    metadata={"said": tripped},
                ),
            )
            transition = self._round_end_transition(state, events)
            return StepTransition(transition.state, transition.events, rewards)
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
        return StepTransition(state, events, rewards)

    def is_terminal(self, state: TabooState) -> bool:
        return state.phase == GAME_END

    def score(self, log: EventLog, state: TabooState) -> dict[str, JsonValue]:
        return {
            "winner": state.winner,
            "terminal_reason": state.terminal_reason,
            "red_points": state.points[RED],
            "blue_points": state.points[BLUE],
            "violations": dict(state.violations),
            "rounds_played": min(state.round_number, state.total_rounds()),
            "cards_played": state.cards_played,
            "event_count": len(log.events),
        }

    # -- internals -----------------------------------------------------------

    def _discard_card(self, state: TabooState) -> TabooState:
        """The card in play leaves play: next card up, fresh hints, floor to
        the describer."""
        return replace(
            state,
            card_cursor=state.card_cursor + 1,
            cards_played=state.cards_played + 1,
            current_hints=(),
            wrong_guesses=(),
            awaiting=AWAIT_DESCRIBE,
        )

    def _round_end_transition(
        self,
        state: TabooState,
        lead_events: list[EventRecord],
    ) -> StepTransition:
        rounds_done = dict(state.rounds_done)
        rounds_done[state.current_team] += 1
        state = replace(state, rounds_done=rounds_done)
        events = [
            *lead_events,
            EventRecord(
                "round_ended",
                {
                    "round_number": state.round_number,
                    "team_id": state.current_team,
                    "points": dict(state.points),
                    "violations": dict(state.violations),
                },
                visibility=("replay", "audience"),
            ),
        ]
        if state.round_number >= state.total_rounds():
            # No game_ended event here: the submit that carried this expiry
            # still runs validate (terminal_game, absorbed by handle_invalid),
            # after which the state machine appends game_ended itself.
            winner, reason = self._final_verdict(state)
            final_state = replace(
                state,
                phase=GAME_END,
                winner=winner,
                terminal_reason=reason,
            )
            return StepTransition(final_state, tuple(events))
        next_state = replace(
            self._discard_card(state),
            current_team=self._other_team(state.current_team),
            round_number=state.round_number + 1,
            guess_cursor=0,
            skips_this_round=0,
        )
        next_state = replace(next_state, speech_left=self._fresh_clocks(next_state))
        return StepTransition(next_state, tuple(events))

    def _final_verdict(self, state: TabooState) -> tuple[str | None, str]:
        red, blue = state.points[RED], state.points[BLUE]
        if red != blue:
            return (RED if red > blue else BLUE), "most_points"
        red_v, blue_v = state.violations[RED], state.violations[BLUE]
        if red_v != blue_v:
            return (RED if red_v < blue_v else BLUE), "fewest_violations"
        return None, "draw"

    def _other_team(self, team_id: str) -> str:
        return BLUE if team_id == RED else RED
