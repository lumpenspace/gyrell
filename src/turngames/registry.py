"""Registry of playable games.

One place mapping a game id to everything a table runner (live channel,
replay recorder, eval script) needs to seat a match: the spec, a live-table
config, the seat layout, and per-game actor machinery. Adding a game means
adding an entry here — the server and scripts stay game-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from turngames.actors import (
    BaselineCodewordsActor,
    BaselineTabooActor,
    CodewordsPrompter,
    TabooPrompter,
)
from turngames.games.codewords import CodewordsConfig, CodewordsSpec
from turngames.games.taboo import TabooConfig, TabooSpec


@dataclass(frozen=True)
class GameEntry:
    id: str
    title: str
    spec_factory: Callable[[], Any]
    # Live-table defaults: seat labels in, ready-to-play config out.
    config_factory: Callable[[dict[str, str] | None], Any]
    seat_ids: tuple[str, ...]
    baseline_factory: Callable[[str], Any]  # seed -> scripted actor
    prompter_factory: Callable[[], Any]  # -> OpenRouterActor prompter


GAMES: dict[str, GameEntry] = {
    "codewords": GameEntry(
        id="codewords",
        title="Codewords",
        spec_factory=CodewordsSpec,
        config_factory=lambda labels: CodewordsConfig(
            turn_time_limit_words=120,
            seat_labels=dict(labels) if labels else None,
        ),
        seat_ids=(
            "red_cluegiver",
            "red_guesser_1",
            "red_guesser_2",
            "blue_cluegiver",
            "blue_guesser_1",
            "blue_guesser_2",
        ),
        baseline_factory=lambda seed: BaselineCodewordsActor(seed=seed),
        prompter_factory=CodewordsPrompter,
    ),
    "taboo": GameEntry(
        id="taboo",
        title="Taboo",
        spec_factory=TabooSpec,
        config_factory=lambda labels: TabooConfig(
            seat_labels=dict(labels) if labels else None,
        ),
        seat_ids=(
            "red_player_1",
            "red_player_2",
            "red_player_3",
            "blue_player_1",
            "blue_player_2",
            "blue_player_3",
        ),
        baseline_factory=lambda seed: BaselineTabooActor(seed=seed),
        prompter_factory=TabooPrompter,
    ),
}
