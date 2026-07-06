"""Random-but-legal baseline actor for Taboo.

The describer gives the card's category ("It is a kind of object.") — legal
under the spelling-hint ban — plus checked-safe flavor; the guesser mines
the category from the hints and works through matching deck targets.
Deliberately beatable filler: enough hits to keep demo games lively, no
real word knowledge.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from turngames.core import Action, ActorInput, ActorOutput
from turngames.games.taboo import PARTY_DECK, TabooCard, find_taboo_word

PASS_LINES = (
    "Nothing yet, keep going.",
    "Give me another angle.",
    "I need one more hint.",
)

GUESS_LINES = (
    "Has to be",
    "I'll say",
    "Trying",
)

FLAVOR_LINES = (
    "You would recognize one instantly.",
    "Everyone has run into one of these.",
    "Very common, nothing exotic.",
)

_CATEGORY_RE = re.compile(r"kind of (\w+)", re.IGNORECASE)


@dataclass
class BaselineTabooActor:
    """One instance per seat; deterministic for a given seed."""

    seed: str
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, actor_input: ActorInput) -> ActorOutput:
        # The describer role rotates; the allowed actions say which hat this
        # seat wears right now.
        allowed = {a["type"] for a in actor_input.observation.allowed_actions}
        if "describe" in allowed:
            return self._describer(actor_input)
        if "guess" in allowed or "pass" in allowed:
            return self._guesser(actor_input)
        raise RuntimeError(
            f"baseline taboo actor has no legal actions for seat {actor_input.seat.id}"
        )

    # -- describer -----------------------------------------------------------

    def _describer(self, actor_input: ActorInput) -> ActorOutput:
        data = actor_input.observation.data
        card_json = data.get("current_card") or {}
        card = TabooCard(
            target=str(card_json.get("target", "")),
            forbidden=tuple(card_json.get("forbidden", ())),
            category=str(card_json.get("category", "")),
        )
        allowed = {a["type"] for a in actor_input.observation.allowed_actions}
        hints_given = len(data.get("current_hints") or ())
        candidates = []
        if card.category:
            candidates.append(f"It is a kind of {card.category}.")
        if hints_given:
            candidates.append(self.rng.choice(FLAVOR_LINES))
        for hint in candidates:
            if find_taboo_word(hint, card) is None:
                return ActorOutput(public_deliberation=hint, action=Action.of("describe"))
        if "skip" in allowed:
            return ActorOutput(public_deliberation="", action=Action.of("skip"))
        return ActorOutput(public_deliberation="", action=Action.of("describe"))

    # -- guesser ---------------------------------------------------------------

    def _guesser(self, actor_input: ActorInput) -> ActorOutput:
        data = actor_input.observation.data
        hints = " ".join(data.get("current_hints") or ())
        category_match = _CATEGORY_RE.search(hints)
        if category_match:
            category = category_match.group(1).casefold()
            candidates = [
                c.target for c in PARTY_DECK.cards if c.category == category
            ]
            if candidates:
                word = self.rng.choice(candidates)
                return ActorOutput(
                    public_deliberation=f"{self.rng.choice(GUESS_LINES)} {word}!",
                    action=Action.of("guess", word=word),
                )
        return ActorOutput(
            public_deliberation=self.rng.choice(PASS_LINES),
            action=Action.of("pass"),
        )
