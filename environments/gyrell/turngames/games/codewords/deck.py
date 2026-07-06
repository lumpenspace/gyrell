"""Card decks for Codewords.

A deck is just a named word pack plus a seeded, reproducible deal. The classic
deck mirrors the original Codenames (2015) English word list.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CardDeck:
    """A named pack of card words that can deal reproducible boards."""

    name: str
    words: tuple[str, ...]

    def deal(self, seed: str, count: int = 25) -> tuple[str, ...]:
        """Deal `count` distinct cards, deterministically for a given seed."""
        if count > len(self.words):
            raise ValueError(
                f"deck {self.name!r} has {len(self.words)} cards, cannot deal {count}"
            )
        rng = random.Random(f"{seed}:deck:{self.name}")
        return tuple(rng.sample(self.words, count))


CLASSIC_WORDS = (
    "africa", "agent", "air", "alien", "alps", "amazon", "ambulance",
    "america", "angel", "antarctica", "apple", "arm", "atlantis",
    "australia", "aztec", "back", "ball", "band", "bank", "bar", "bark",
    "bat", "battery", "beach", "bear", "beat", "bed", "beijing", "bell",
    "belt", "berlin", "bermuda", "berry", "bill", "block", "board", "bolt",
    "bomb", "bond", "boom", "boot", "bottle", "bow", "box", "bridge",
    "brush", "buck", "buffalo", "bug", "bugle", "button", "calf", "canada",
    "cap", "capital", "car", "card", "carrot", "casino", "cast", "cat",
    "cell", "centaur", "center", "chair", "change", "charge", "check",
    "chest", "chick", "china", "chocolate", "church", "circle", "cliff",
    "cloak", "club", "code", "cold", "comic", "compound", "concert",
    "conductor", "contract", "cook", "copper", "cotton", "court", "cover",
    "crane", "crash", "cricket", "cross", "crown", "cycle", "czech",
    "dance", "date", "day", "death", "deck", "degree", "diamond", "dice",
    "dinosaur", "disease", "doctor", "dog", "draft", "dragon", "dress",
    "drill", "drop", "duck", "dwarf", "eagle", "egypt", "embassy",
    "engine", "england", "europe", "eye", "face", "fair", "fall", "fan",
    "fence", "field", "fighter", "figure", "file", "film", "fire", "fish",
    "flute", "fly", "foot", "force", "forest", "fork", "france", "game",
    "gas", "genius", "germany", "ghost", "giant", "glass", "glove", "gold",
    "grace", "grass", "greece", "green", "ground", "ham", "hand", "hawk",
    "head", "heart", "helicopter", "himalayas", "hole", "hollywood",
    "honey", "hood", "hook", "horn", "horse", "horseshoe", "hospital",
    "hotel", "ice", "ice cream", "india", "iron", "ivory", "jack", "jam",
    "jet", "jupiter", "kangaroo", "ketchup", "key", "kid", "king", "kiwi",
    "knife", "knight", "lab", "lap", "laser", "lawyer", "lead", "lemon",
    "leprechaun", "life", "light", "limousine", "line", "link", "lion",
    "litter", "loch ness", "lock", "log", "london", "luck", "mail",
    "mammoth", "maple", "marble", "march", "mass", "match", "mercury",
    "mexico", "microscope", "millionaire", "mine", "mint", "missile",
    "model", "mole", "moon", "moscow", "mount", "mouse", "mouth", "mug",
    "nail", "needle", "net", "new york", "night", "ninja", "note",
    "novel", "nurse", "nut", "octopus", "oil", "olive", "olympus",
    "opera", "orange", "organ", "palm", "pan", "pants", "paper",
    "parachute", "park", "part", "pass", "paste", "penguin", "phoenix",
    "piano", "pie", "pilot", "pin", "pipe", "pirate", "pistol", "pit",
    "pitch", "plane", "plastic", "plate", "platypus", "play", "plot",
    "point", "poison", "pole", "police", "pool", "port", "post", "pound",
    "press", "princess", "pumpkin", "pupil", "pyramid", "queen", "rabbit",
    "racket", "ray", "revolution", "ring", "robin", "robot", "rock",
    "rome", "root", "rose", "roulette", "round", "row", "ruler",
    "satellite", "saturn", "scale", "school", "scientist", "scorpion",
    "screen", "scuba diver", "seal", "server", "shadow", "shakespeare",
    "shark", "ship", "shoe", "shop", "shot", "sink", "skyscraper", "slip",
    "slug", "smuggler", "snow", "snowman", "sock", "soldier", "soul",
    "sound", "space", "spell", "spider", "spike", "spine", "spot",
    "spring", "spy", "square", "stadium", "staff", "star", "state",
    "stick", "stock", "straw", "stream", "strike", "string", "sub",
    "suit", "superhero", "swing", "switch", "table", "tablet", "tag",
    "tail", "tap", "teacher", "telescope", "temple", "theater", "thief",
    "thumb", "tick", "tie", "time", "tokyo", "tooth", "torch", "tower",
    "track", "train", "triangle", "trip", "trunk", "tube", "turkey",
    "undertaker", "unicorn", "vacuum", "van", "vet", "wake", "wall",
    "war", "washer", "washington", "watch", "water", "wave", "web",
    "well", "whale", "whip", "wind", "witch", "worm", "yard",
)

CLASSIC_DECK = CardDeck(name="classic-en", words=CLASSIC_WORDS)

DECKS: dict[str, CardDeck] = {
    CLASSIC_DECK.name: CLASSIC_DECK,
}
