from __future__ import annotations

from dataclasses import dataclass, field, replace

from turngames.core.board import BoardLayer, GridBoard
from turngames.core.types import JsonValue, Seat
from turngames.core.visibility import VisibilityContext, VisibilityRule

RED = "red"
BLUE = "blue"
NEUTRAL = "neutral"
ASSASSIN = "assassin"

CLUE_PROPOSAL = "clue_proposal"
GUESSING = "guessing"
GAME_END = "game_end"

CLUEGIVER = "cluegiver"
GUESSER = "guesser"
ARBITER = "arbiter"


@dataclass(frozen=True)
class CodewordsConfig:
    width: int = 5
    height: int = 5
    # When words is None, the board is dealt from the named deck using the
    # game seed (see turngames.games.codewords.deck).
    words: tuple[str, ...] | None = None
    deck: str = "classic-en"
    key: tuple[str, ...] | None = None
    starting_team: str = RED
    max_clue_count: int = 9
    illegal_clue_ends_turn: bool = True
    turn_time_limit_words: int | None = None
    guessers_per_team: int = 2
    # When a team completes all its guesses for a clue (more than one, all
    # correct) with clock time remaining, the cluegiver may give a bonus
    # clue instead of the turn ending. Off by default — standard play ends
    # the turn once the guesses (including the free +1) are spent.
    bonus_clue_when_time_left: bool = False
    # Maps seat id -> actor/model label (e.g. "claude-3.5-sonnet"), shown to
    # viewers and used by leaderboard aggregation. Seats stay ephemeral;
    # this is presentation/attribution metadata, not game state.
    seat_labels: dict[str, str] | None = None


@dataclass(frozen=True)
class Proposal:
    seat_id: str
    word: str

    def to_json(self) -> dict[str, JsonValue]:
        return {"seat_id": self.seat_id, "word": self.word}


@dataclass(frozen=True)
class Clue:
    team_id: str
    word: str
    count: int

    def to_json(self) -> dict[str, JsonValue]:
        return {"team_id": self.team_id, "word": self.word, "count": self.count}


@dataclass(frozen=True)
class CodewordsBoard:
    grid: GridBoard
    key: dict[str, str]
    revealed: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_words_and_key(
        cls,
        width: int,
        height: int,
        words: tuple[str, ...],
        key: tuple[str, ...],
    ) -> "CodewordsBoard":
        grid = GridBoard.from_labels(width, height, words, label_layer="word")
        assignments = {cell.id: key[index] for index, cell in enumerate(grid.cells)}
        return cls(grid=grid, key=assignments)

    def reveal(self, cell_id: str) -> "CodewordsBoard":
        return replace(self, revealed=self.revealed | {cell_id})

    def cell_id_for_word(self, word: str) -> str | None:
        normalized = word.casefold()
        word_layer = self.grid.layers[0]
        for cell_id, value in word_layer.values.items():
            if isinstance(value, str) and value.casefold() == normalized:
                return cell_id
        return None

    def word_for_cell(self, cell_id: str) -> str:
        value = self.grid.layers[0].values[cell_id]
        if not isinstance(value, str):
            raise TypeError("word layer contains a non-string value")
        return value

    def assignment_for_word(self, word: str) -> str | None:
        cell_id = self.cell_id_for_word(word)
        return None if cell_id is None else self.key[cell_id]

    def unrevealed_words(self) -> tuple[str, ...]:
        return tuple(
            self.word_for_cell(cell.id)
            for cell in self.grid.cells
            if cell.id not in self.revealed
        )

    def remaining_for_team(self, team_id: str) -> int:
        return sum(
            1
            for cell_id, assignment in self.key.items()
            if assignment == team_id and cell_id not in self.revealed
        )

    def view_for(self, context: VisibilityContext) -> dict[str, JsonValue]:
        key_rule = VisibilityRule.for_roles(CLUEGIVER, ARBITER).union(
            VisibilityRule.for_viewers("audience_omniscient", "replay")
        )
        revealed_values: dict[str, JsonValue] = {
            cell.id: self.key[cell.id] if cell.id in self.revealed else None
            for cell in self.grid.cells
        }
        key_values: dict[str, JsonValue] = {
            cell.id: self.key[cell.id] for cell in self.grid.cells
        }
        board = (
            self.grid.with_layer(
                BoardLayer(
                    id="revealed",
                    values=revealed_values,
                    visibility=VisibilityRule.public(),
                )
            )
            .with_layer(
                BoardLayer(
                    id="key",
                    values=key_values,
                    visibility=key_rule,
                )
            )
            .project(context)
        )
        return {
            "width": board.width,
            "height": board.height,
            "cells": list(board.cells),
        }


@dataclass(frozen=True)
class CodewordsState:
    seed: str
    config: CodewordsConfig
    board: CodewordsBoard
    seats: tuple[Seat, ...]
    phase: str = CLUE_PROPOSAL
    current_team: str = RED
    turn_number: int = 1
    current_clue: Clue | None = None
    guesses_this_turn: int = 0
    winner: str | None = None
    terminal_reason: str | None = None
    illegal_clues: dict[str, int] = field(default_factory=dict)
    time_left: int | None = None
    final_guess_only: bool = False
    pending_proposal: Proposal | None = None
    guess_cursor: int = 0
    spoken_this_turn: dict[str, int] = field(default_factory=dict)
    final_guess_seat: str | None = None

    def seat(self, seat_id: str) -> Seat | None:
        return next((seat for seat in self.seats if seat.id == seat_id), None)

    def active_cluegiver_id(self) -> str:
        return f"{self.current_team}_cluegiver"

    def guesser_ids(self, team_id: str | None = None) -> tuple[str, ...]:
        team = team_id or self.current_team
        return tuple(
            seat.id for seat in self.seats if seat.role == "guesser" and seat.team_id == team
        )

    def active_guesser_id(self) -> str:
        """The guesser expected to act right now.

        Final constrained guess goes to the designated quiet seat; while a
        proposal is pending, the next guesser in seat order must confirm or
        reject it; otherwise the round-robin cursor decides.
        """
        guessers = self.guesser_ids()
        if self.final_guess_seat is not None:
            return self.final_guess_seat
        if self.pending_proposal is not None:
            proposer_index = guessers.index(self.pending_proposal.seat_id)
            return guessers[(proposer_index + 1) % len(guessers)]
        return guessers[self.guess_cursor % len(guessers)]

    def quietest_guesser(self) -> str:
        """The guesser who spent the least clock time talking this turn."""
        guessers = self.guesser_ids()
        return min(guessers, key=lambda seat: (self.spoken_this_turn.get(seat, 0), guessers.index(seat)))
