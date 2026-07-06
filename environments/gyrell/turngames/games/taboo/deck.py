"""Card decks for Taboo.

A deck is a named pack of cards (target word + forbidden words) plus a
seeded, reproducible deal order. Targets are single alphabetic words so the
guess action can stay a one-token exact match.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from turngames.games.taboo.types import TabooCard


@dataclass(frozen=True)
class TabooDeck:
    """A named pack of taboo cards dealing reproducible orders."""

    name: str
    cards: tuple[TabooCard, ...]

    def deal_order(self, seed: str) -> tuple[int, ...]:
        """A deterministic permutation of card indices for a given seed."""
        order = list(range(len(self.cards)))
        rng = random.Random(f"{seed}:taboo-deck:{self.name}")
        rng.shuffle(order)
        return tuple(order)


def _card(target: str, category: str, *forbidden: str) -> TabooCard:
    return TabooCard(target=target, forbidden=tuple(forbidden), category=category)


PARTY_CARDS = (
    _card("pizza", "food", "cheese", "slice", "italy", "pepperoni", "dough"),
    _card("beach", "place", "sand", "ocean", "wave", "sun", "towel"),
    _card("guitar", "object", "string", "music", "strum", "instrument", "rock"),
    _card("winter", "nature", "cold", "snow", "season", "ice", "december"),
    _card("doctor", "person", "hospital", "patient", "medicine", "nurse", "sick"),
    _card("rainbow", "nature", "color", "rain", "sky", "arc", "pot of gold"),
    _card("camera", "object", "photo", "picture", "lens", "film", "snap"),
    _card("honey", "food", "bee", "sweet", "jar", "pooh", "sticky"),
    _card("pirate", "person", "ship", "treasure", "parrot", "eye patch", "sea"),
    _card("library", "place", "book", "quiet", "read", "shelf", "borrow"),
    _card("volcano", "nature", "lava", "erupt", "mountain", "magma", "ash"),
    _card("breakfast", "food", "morning", "eggs", "cereal", "meal", "toast"),
    _card("umbrella", "object", "rain", "wet", "open", "handle", "cover"),
    _card("dentist", "person", "teeth", "drill", "cavity", "mouth", "floss"),
    _card("circus", "place", "clown", "tent", "acrobat", "ring", "elephant"),
    _card("wedding", "activity", "bride", "groom", "marriage", "ring", "cake"),
    _card("robot", "object", "machine", "metal", "artificial", "beep", "android"),
    _card("desert", "place", "sand", "hot", "cactus", "dry", "camel"),
    _card("chess", "activity", "board", "king", "queen", "checkmate", "pawn"),
    _card("firefighter", "person", "hose", "truck", "flame", "rescue", "ladder"),
    _card("bicycle", "object", "pedal", "wheel", "ride", "helmet", "two"),
    _card("astronaut", "person", "space", "rocket", "moon", "suit", "nasa"),
    _card("chocolate", "food", "candy", "sweet", "cocoa", "brown", "bar"),
    _card("garden", "place", "plant", "flower", "grow", "soil", "vegetable"),
    _card("thunder", "nature", "lightning", "storm", "loud", "sound", "boom"),
    _card("penguin", "animal", "bird", "antarctica", "waddle", "black and white", "ice"),
    _card("birthday", "activity", "cake", "party", "candles", "gift", "year"),
    _card("shadow", "nature", "dark", "light", "follow", "ground", "silhouette"),
    _card("magnet", "object", "attract", "metal", "pole", "fridge", "pull"),
    _card("vampire", "person", "blood", "fangs", "dracula", "bat", "garlic"),
    _card("orchestra", "activity", "music", "conductor", "violin", "symphony", "instruments"),
    _card("submarine", "object", "underwater", "boat", "periscope", "navy", "yellow"),
    _card("mirror", "object", "reflection", "glass", "look", "wall", "image"),
    _card("passport", "object", "travel", "country", "photo", "airport", "document"),
    _card("skeleton", "object", "bones", "skull", "body", "halloween", "ribs"),
    _card("tornado", "nature", "wind", "spin", "storm", "funnel", "twister"),
    _card("waiter", "person", "restaurant", "serve", "tray", "tip", "order"),
    _card("igloo", "place", "ice", "eskimo", "dome", "snow", "house"),
    _card("juggler", "person", "balls", "throw", "catch", "circus", "air"),
    _card("compass", "object", "north", "direction", "needle", "navigate", "map"),
    _card("karaoke", "activity", "sing", "microphone", "song", "bar", "lyrics"),
    _card("avalanche", "nature", "snow", "mountain", "slide", "bury", "fall"),
    _card("scarecrow", "object", "field", "crow", "straw", "farm", "wizard of oz"),
    _card("marathon", "activity", "run", "race", "miles", "long", "finish line"),
    _card("telescope", "object", "stars", "look", "far", "astronomy", "lens"),
    _card("sandwich", "food", "bread", "lunch", "meat", "layers", "sub"),
    _card("elevator", "object", "up", "down", "floor", "button", "lift"),
    _card("fountain", "object", "water", "coin", "wish", "spray", "park"),
    _card("hammock", "object", "sleep", "hang", "trees", "swing", "relax"),
    _card("lighthouse", "place", "ship", "beam", "coast", "warning", "tower"),
    _card("parachute", "object", "jump", "plane", "fall", "open", "skydive"),
    _card("popcorn", "food", "movie", "kernel", "butter", "pop", "snack"),
    _card("snowman", "object", "carrot", "frosty", "winter", "build", "melt"),
    _card("trophy", "object", "win", "award", "gold", "champion", "cup"),
    _card("whistle", "object", "blow", "referee", "sound", "sport", "lips"),
    _card("zoo", "place", "animals", "cage", "visit", "keeper", "wild"),
    _card("bakery", "place", "bread", "cake", "oven", "pastry", "shop"),
    _card("glacier", "nature", "ice", "slow", "melt", "mountain", "cold"),
    _card("jungle", "place", "trees", "tarzan", "dense", "tropical", "vines"),
    _card("lullaby", "activity", "sleep", "baby", "song", "sing", "night"),
    _card("origami", "activity", "paper", "fold", "japan", "crane", "art"),
    _card("quicksand", "nature", "sink", "trap", "wet", "struggle", "stuck"),
    _card("stethoscope", "object", "doctor", "heart", "listen", "chest", "ears"),
    _card("ventriloquist", "person", "dummy", "voice", "mouth", "puppet", "throw"),
)

PARTY_DECK = TabooDeck(name="party-en", cards=PARTY_CARDS)

DECKS: dict[str, TabooDeck] = {
    PARTY_DECK.name: PARTY_DECK,
}
