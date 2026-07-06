"""Random-but-legal baseline actor for Codewords.

Plays any dealt board: the cluegiver picks own-team targets and a legal
vocab clue, guessers propose/confirm/stop with canned chatter. Useful for
live broadcast filler between real model integrations and as an RL baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from turngames.core import Action, ActorInput, ActorOutput

# Clue vocabulary: single alphabetic words unlikely to collide with board
# words; filtered against the live board before use anyway.
CLUE_VOCAB = (
    "melody", "voyage", "royalty", "cuisine", "wilderness", "machinery",
    "folklore", "geometry", "weather", "commerce", "athletics", "mythology",
    "chemistry", "festivity", "navigation", "archery", "monarchy", "orchestra",
    "astronomy", "carpentry", "warfare", "harvest", "medicine", "treasure",
    "villainy", "transport", "predator", "frontier", "ceremony", "currency",
)

STRATEGY_LINES = (
    "Let me scan the board for a tight cluster.",
    "I want two safe words before anything risky.",
    "Avoiding the assassin is the whole game here.",
    "There is a theme hiding in plain sight.",
)

CLUE_LINES = (
    "This should cover a couple of ours.",
    "A bit of a stretch, but worth it.",
    "Keeping it conservative this turn.",
)

PROPOSE_LINES = (
    "This one feels right.",
    "I can connect this to the clue.",
    "Low risk, decent upside.",
)

CONFIRM_LINES = ("Agreed.", "I see it too.", "Locking it in.")
REJECT_LINES = ("Too risky.", "I read that differently.",)
STOP_LINES = ("Let's bank what we have.", "Stopping here.")


@dataclass
class BaselineCodewordsActor:
    """One instance per seat; deterministic for a given seed."""

    seed: str
    chattiness: float = 0.5
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, actor_input: ActorInput) -> ActorOutput:
        role = actor_input.seat.role
        if role == "cluegiver":
            return self._cluegiver(actor_input)
        if role == "guesser":
            return self._guesser(actor_input)
        raise RuntimeError(f"baseline actor cannot play role: {role}")

    # -- cluegiver ---------------------------------------------------------

    def _cluegiver(self, actor_input: ActorInput) -> ActorOutput:
        data = actor_input.observation.data
        team = actor_input.seat.team_id
        cells = data["board"]["cells"]
        own_unrevealed = [
            c["word"] for c in cells if c.get("key") == team and c["revealed"] is None
        ]
        board_words = [c["word"].casefold() for c in cells if c["revealed"] is None]

        allowed_says = {a["type"] for a in actor_input.observation.allowed_actions}
        if "say" in allowed_says and self.rng.random() < self.chattiness:
            return ActorOutput(
                public_deliberation=self.rng.choice(STRATEGY_LINES),
                action=Action.of("say"),
            )

        legal_vocab = [
            w
            for w in CLUE_VOCAB
            if all(w not in b and b not in w for b in board_words)
        ]
        clue_word = self.rng.choice(legal_vocab or ["signal"])
        count = min(len(own_unrevealed), self.rng.choice((1, 2, 2, 3)))
        return ActorOutput(
            public_deliberation=self.rng.choice(CLUE_LINES),
            action=Action.of("give_clue", word=clue_word, count=max(count, 1)),
        )

    # -- guesser -----------------------------------------------------------

    def _guesser(self, actor_input: ActorInput) -> ActorOutput:
        data = actor_input.observation.data
        cells = data["board"]["cells"]
        unrevealed = [c["word"] for c in cells if c["revealed"] is None]
        allowed = {a["type"] for a in actor_input.observation.allowed_actions}

        if "choose" in allowed and "propose" not in allowed:
            # Solo guesser or constrained final word.
            word = self.rng.choice(unrevealed)
            constrained = bool(actor_input.constraints.get("final_guess_one_word"))
            return ActorOutput(
                public_deliberation="" if constrained else self.rng.choice(PROPOSE_LINES),
                action=Action.of("choose", word=word),
            )

        if data.get("pending_proposal") is not None and "confirm" in allowed:
            if self.rng.random() < 0.8:
                return ActorOutput(
                    public_deliberation=self.rng.choice(CONFIRM_LINES),
                    action=Action.of("confirm"),
                )
            return ActorOutput(
                public_deliberation=self.rng.choice(REJECT_LINES),
                action=Action.of("reject"),
            )

        clue = data.get("current_clue") or {}
        clue_count = int(clue.get("count", 1))
        if data.get("guesses_this_turn", 0) >= clue_count and "stop" in allowed:
            return ActorOutput(
                public_deliberation=self.rng.choice(STOP_LINES),
                action=Action.of("stop"),
            )

        word = self.rng.choice(unrevealed)
        return ActorOutput(
            public_deliberation=self.rng.choice(PROPOSE_LINES),
            action=Action.of("propose", word=word),
        )
