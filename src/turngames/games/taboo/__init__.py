"""Taboo: describe the target word without saying the forbidden ones."""

from turngames.games.taboo.deck import DECKS, PARTY_DECK, TabooDeck
from turngames.games.taboo.spec import (
    TabooSpec,
    find_spelling_hint,
    find_taboo_word,
    utterance_says_target,
)
from turngames.games.taboo.types import TabooCard, TabooConfig, TabooState

__all__ = [
    "DECKS",
    "PARTY_DECK",
    "TabooCard",
    "TabooConfig",
    "TabooDeck",
    "TabooSpec",
    "TabooState",
    "find_spelling_hint",
    "find_taboo_word",
    "utterance_says_target",
]
