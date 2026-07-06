from __future__ import annotations

from dataclasses import dataclass, field

from turngames.core.types import JsonValue, Seat

RED = "red"
BLUE = "blue"

DESCRIBING = "describing"
GAME_END = "game_end"

# Every competitor is a PLAYER: the describer role rotates through the team
# round by round, tabletop style, rather than being pinned to a seat.
PLAYER = "player"
ARBITER = "arbiter"

# Whose voice the floor is waiting on inside a round.
AWAIT_DESCRIBE = "describe"
AWAIT_GUESS = "guess"


@dataclass(frozen=True)
class TabooCard:
    """One card: the word to convey plus the words the describer may not say.

    `category` is coarse metadata ("food", "place", …) used by the scripted
    baseline describer, which needs a legal hint now that spelling hints
    ("9 letters, starts with a") are a buzz."""

    target: str
    forbidden: tuple[str, ...]
    category: str = ""

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "target": self.target,
            "forbidden": list(self.forbidden),
            "category": self.category,
        }


@dataclass(frozen=True)
class TabooConfig:
    # When cards is None, the deal comes from the named deck using the game
    # seed (see turngames.games.taboo.deck).
    cards: tuple[TabooCard, ...] | None = None
    deck: str = "party-en"
    starting_team: str = RED
    rounds_per_team: int = 3
    players_per_team: int = 3
    # Each participant's personal word clock per round. The round ends when
    # the DESCRIBER runs dry; a guesser who spends theirs just goes quiet.
    words_per_participant: int = 100
    max_skips_per_round: int = 2
    # Classic house rule: a buzzed card scores for the other team.
    violation_point_to_opponent: bool = True
    # Maps seat id -> actor/model label, presentation metadata only.
    seat_labels: dict[str, str] | None = None


@dataclass(frozen=True)
class TabooState:
    seed: str
    config: TabooConfig
    seats: tuple[Seat, ...]
    # Seeded permutation over config.cards; the cursor walks it (wrapping, so
    # a long game repeats cards rather than running dry).
    deal_order: tuple[int, ...]
    phase: str = DESCRIBING
    current_team: str = RED
    round_number: int = 1
    card_cursor: int = 0
    cards_played: int = 0
    awaiting: str = AWAIT_DESCRIBE
    guess_cursor: int = 0
    points: dict[str, int] = field(default_factory=lambda: {RED: 0, BLUE: 0})
    violations: dict[str, int] = field(default_factory=lambda: {RED: 0, BLUE: 0})
    skips_this_round: int = 0
    # Everything the describer has legally said about the current card, in
    # order — guessers (and the audience) work from these.
    current_hints: tuple[str, ...] = ()
    # Per-seat word clocks for the describing team's current round.
    speech_left: dict[str, int] = field(default_factory=dict)
    # How many rounds each team has completed — also the describer rotation
    # cursor: round N for a team is described by its (N mod size)-th player.
    rounds_done: dict[str, int] = field(default_factory=lambda: {RED: 0, BLUE: 0})
    # The public deliberation of the submit currently being processed; stashed
    # by consume_public_deliberation so validate()/step() can rule on what
    # was actually said aloud.
    last_utterance: str = ""
    winner: str | None = None
    terminal_reason: str | None = None

    def seat(self, seat_id: str) -> Seat | None:
        return next((seat for seat in self.seats if seat.id == seat_id), None)

    def players(self, team_id: str | None = None) -> tuple[str, ...]:
        team = team_id or self.current_team
        return tuple(
            seat.id for seat in self.seats if seat.role == PLAYER and seat.team_id == team
        )

    def active_describer_id(self) -> str:
        players = self.players()
        return players[self.rounds_done[self.current_team] % len(players)]

    def guesser_ids(self, team_id: str | None = None) -> tuple[str, ...]:
        describer = self.active_describer_id()
        return tuple(p for p in self.players(team_id) if p != describer)

    def active_guesser_id(self) -> str:
        guessers = self.guesser_ids()
        return guessers[self.guess_cursor % len(guessers)]

    def current_card(self) -> TabooCard:
        if self.config.cards is None:
            raise RuntimeError("state built without a dealt card set")
        return self.config.cards[self.deal_order[self.card_cursor % len(self.deal_order)]]

    def total_rounds(self) -> int:
        return self.config.rounds_per_team * 2
