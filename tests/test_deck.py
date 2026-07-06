import unittest

from turngames.core import StateMachine
from turngames.games.codewords import CLASSIC_DECK, CodewordsConfig, CodewordsSpec


class DeckTest(unittest.TestCase):
    def test_deal_is_deterministic_per_seed(self) -> None:
        first = CLASSIC_DECK.deal("seed-a")
        second = CLASSIC_DECK.deal("seed-a")
        other = CLASSIC_DECK.deal("seed-b")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 25)
        self.assertEqual(len(set(first)), 25)
        for word in first:
            self.assertIn(word, CLASSIC_DECK.words)

    def test_deal_rejects_oversized_request(self) -> None:
        with self.assertRaises(ValueError):
            CLASSIC_DECK.deal("seed", count=len(CLASSIC_DECK.words) + 1)

    def test_initial_state_deals_from_deck_when_words_omitted(self) -> None:
        machine = StateMachine.new(CodewordsSpec(), "deck-seed", CodewordsConfig())
        board_words = machine.state.board.unrevealed_words()

        self.assertEqual(len(board_words), 25)
        self.assertEqual(board_words, tuple(CLASSIC_DECK.deal("deck-seed")))

    def test_unknown_deck_raises(self) -> None:
        with self.assertRaises(ValueError):
            CodewordsSpec().initial_state("seed", CodewordsConfig(deck="no-such-deck"))


if __name__ == "__main__":
    unittest.main()
