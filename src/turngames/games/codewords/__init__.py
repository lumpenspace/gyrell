"""Codenames-like game implementation."""

from turngames.games.codewords.deck import CLASSIC_DECK, DECKS, CardDeck
from turngames.games.codewords.spec import CodewordsSpec
from turngames.games.codewords.types import CodewordsConfig, CodewordsState

__all__ = [
    "CLASSIC_DECK",
    "CardDeck",
    "CodewordsConfig",
    "CodewordsSpec",
    "CodewordsState",
    "DECKS",
]

