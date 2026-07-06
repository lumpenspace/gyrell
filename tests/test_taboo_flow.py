import unittest

from turngames.core import Action, StateMachine
from turngames.games.taboo import (
    TabooCard,
    TabooConfig,
    TabooSpec,
    find_spelling_hint,
    find_taboo_word,
    utterance_says_target,
)
from turngames.games.taboo.types import BLUE, RED

CARD = TabooCard(target="pizza", forbidden=("cheese", "slice", "italy", "pepperoni", "dough"))
OTHER = TabooCard(target="beach", forbidden=("sand", "ocean", "wave", "sun", "towel"))


class FindTabooWordTest(unittest.TestCase):
    def test_clean_description_passes(self) -> None:
        self.assertIsNone(find_taboo_word("a round dinner you order by phone", CARD))

    def test_target_and_derivatives_trip(self) -> None:
        self.assertEqual(find_taboo_word("everyone loves Pizza night", CARD), "pizza")
        self.assertEqual(find_taboo_word("we ordered two pizzas", CARD), "pizza")

    def test_part_of_target_trips(self) -> None:
        card = TabooCard(target="snowman", forbidden=())
        self.assertEqual(find_taboo_word("it melts like snow", card), "snowman")

    def test_forbidden_word_and_derivative_trip(self) -> None:
        self.assertEqual(find_taboo_word("covered in melted cheese", CARD), "cheese")
        self.assertEqual(find_taboo_word("cut into slices", CARD), "slice")

    def test_short_stopwords_do_not_trip_inside_out(self) -> None:
        card = TabooCard(target="curtain", forbidden=("theater",))
        self.assertIsNone(find_taboo_word("the thing on the window", card))

    def test_multiword_forbidden_matches_as_phrase(self) -> None:
        card = TabooCard(target="rainbow", forbidden=("pot of gold",))
        self.assertEqual(find_taboo_word("look for a pot of gold", card), "pot of gold")
        self.assertIsNone(find_taboo_word("a gold pot", card))

    def test_utterance_says_target(self) -> None:
        self.assertTrue(utterance_says_target("maybe pizza?", "pizza"))
        self.assertTrue(utterance_says_target("two pizzas!", "pizza"))
        self.assertFalse(utterance_says_target("pizz... no idea", "pizza"))


class FindSpellingHintTest(unittest.TestCase):
    def test_letter_and_length_hints_trip(self) -> None:
        self.assertIsNotNone(find_spelling_hint("9 letters, first letter a"))
        self.assertIsNotNone(find_spelling_hint("nine letters long"))
        self.assertIsNotNone(find_spelling_hint("it starts with P"))
        self.assertIsNotNone(find_spelling_hint("rhymes with fizz"))
        self.assertIsNotNone(find_spelling_hint("the letter 'q' is in it"))
        self.assertIsNotNone(find_spelling_hint("spelled with a silent k"))
        self.assertIsNotNone(find_spelling_hint("its initials are N A"))

    def test_ordinary_speech_does_not_trip(self) -> None:
        self.assertIsNone(find_spelling_hint("you deliver letters with it"))
        self.assertIsNone(find_spelling_hint("my initial thought is a tool"))
        self.assertIsNone(find_spelling_hint("one of these hangs between trees"))


class TabooFlowTest(unittest.TestCase):
    def new_machine(self, **config_kwargs) -> StateMachine:
        config_kwargs.setdefault("cards", (CARD, OTHER))
        config_kwargs.setdefault("words_per_participant", 30)
        config_kwargs.setdefault("rounds_per_team", 1)
        config = TabooConfig(starting_team=RED, **config_kwargs)
        return StateMachine.new(TabooSpec(), "test-seed", config)

    def burn_describer(self, machine, seat_id: str) -> None:
        """Speak the describer's whole clock away, ending their round."""
        words = " ".join(["word"] * (machine.state.config.words_per_participant + 1))
        machine.submit(seat_id, Action.of("describe"), public_deliberation=words)

    def test_round_one_describer_sees_card_and_others_do_not(self) -> None:
        machine = self.new_machine()
        self.assertIn("current_card", machine.observe("red_player_1").data)
        self.assertNotIn("current_card", machine.observe("red_player_2").data)
        self.assertNotIn("current_card", machine.observe("blue_player_1").data)

    def test_viewer_projections(self) -> None:
        machine = self.new_machine()
        spec = machine.spec
        omniscient = spec.observe_viewer(machine.state, "audience_omniscient")
        safe = spec.observe_viewer(machine.state, "spoiler_safe")
        self.assertIn("current_card", omniscient)
        self.assertNotIn("current_card", safe)
        self.assertEqual(omniscient["describer"], "red_player_1")

    def test_describer_rotates_each_team_round(self) -> None:
        machine = self.new_machine(rounds_per_team=2, words_per_participant=5)
        self.assertEqual(machine.spec.acting_seat(machine.state), "red_player_1")
        self.burn_describer(machine, "red_player_1")
        self.assertEqual(machine.state.current_team, BLUE)
        self.assertEqual(machine.spec.acting_seat(machine.state), "blue_player_1")
        self.burn_describer(machine, "blue_player_1")
        # second red round: the card moves to the next red player
        self.assertEqual(machine.spec.acting_seat(machine.state), "red_player_2")
        self.assertIn("current_card", machine.observe("red_player_2").data)
        self.assertNotIn("current_card", machine.observe("red_player_1").data)

    def test_describe_then_correct_guess_scores_and_deals_next_card(self) -> None:
        machine = self.new_machine()
        target = machine.state.current_card().target
        ruling, _ = machine.submit(
            "red_player_1", Action.of("describe"), public_deliberation="round dinner"
        )
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.awaiting, "guess")
        self.assertEqual(machine.spec.acting_seat(machine.state), "red_player_2")

        ruling, rewards = machine.submit(
            "red_player_2", Action.of("guess", word=target), public_deliberation=""
        )
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.points[RED], 1)
        self.assertNotEqual(machine.state.current_card().target, target)
        self.assertEqual(machine.state.current_hints, ())
        self.assertEqual({r.seat_id for r in rewards}, {"red_player_2", "red_player_1"})

    def test_guessers_alternate(self) -> None:
        machine = self.new_machine()
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="hm")
        machine.submit("red_player_2", Action.of("guess", word="wrong"), public_deliberation="")
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="hm2")
        self.assertEqual(machine.spec.acting_seat(machine.state), "red_player_3")

    def test_saying_the_target_counts_without_a_formal_guess(self) -> None:
        machine = self.new_machine()
        target = machine.state.current_card().target
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="hm")
        ruling, _ = machine.submit(
            "red_player_2",
            Action.of("pass"),
            public_deliberation=f"wait — is it {target}? not sure, back to you",
        )
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.points[RED], 1)
        event = next(e for e in machine.log.events if e.type == "card_guessed")
        self.assertEqual(event.payload["via"], "spoken")

    def test_spoken_target_beats_a_wrong_formal_guess(self) -> None:
        machine = self.new_machine()
        target = machine.state.current_card().target
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="hm")
        machine.submit(
            "red_player_2",
            Action.of("guess", word="wrong"),
            public_deliberation=f"either wrong or {target}s, going wrong",
        )
        self.assertEqual(machine.state.points[RED], 1)

    def test_guesser_clock_clips_speech_but_does_not_end_round(self) -> None:
        machine = self.new_machine(words_per_participant=4, rounds_per_team=2)
        target = machine.state.current_card().target
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="a hint")
        # the target falls beyond the guesser's own 4-word clock: never spoken
        machine.submit(
            "red_player_2",
            Action.of("pass"),
            public_deliberation=f"one two three four {target}",
        )
        self.assertEqual(machine.state.points[RED], 0)
        self.assertEqual(machine.state.current_team, RED)  # round continues
        self.assertEqual(machine.state.speech_left["red_player_2"], 0)
        # a spent guesser's later speech clips to nothing
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="go")
        machine.submit(
            "red_player_3", Action.of("pass"), public_deliberation="thinking hard"
        )
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="go2")
        machine.submit(
            "red_player_2", Action.of("pass"), public_deliberation=target
        )
        self.assertEqual(machine.state.points[RED], 0)

    def test_violation_scores_opponent_and_discards_card(self) -> None:
        machine = self.new_machine()
        target = machine.state.current_card().target
        forbidden = machine.state.current_card().forbidden[0]
        ruling, rewards = machine.submit(
            "red_player_1",
            Action.of("describe"),
            public_deliberation=f"it is covered in {forbidden}",
        )
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "taboo_violation")
        self.assertEqual(machine.state.points[BLUE], 1)
        self.assertEqual(machine.state.violations[RED], 1)
        self.assertNotEqual(machine.state.current_card().target, target)
        self.assertEqual(rewards[-1].value, -1.0)

    def test_spelling_hint_is_a_buzz(self) -> None:
        machine = self.new_machine()
        target = machine.state.current_card().target
        ruling, _ = machine.submit(
            "red_player_1",
            Action.of("describe"),
            public_deliberation=f"{len(target)} letters, starts with {target[0]}",
        )
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "spelling_hint")
        self.assertEqual(machine.state.points[BLUE], 1)
        self.assertEqual(machine.state.violations[RED], 1)
        self.assertNotEqual(machine.state.current_card().target, target)

    def test_skip_limit(self) -> None:
        machine = self.new_machine(max_skips_per_round=1)
        ruling, _ = machine.submit("red_player_1", Action.of("skip"), public_deliberation="")
        self.assertTrue(ruling.is_legal)
        ruling, _ = machine.submit("red_player_1", Action.of("skip"), public_deliberation="")
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "skip_limit_reached")

    def test_describer_expiry_ends_round_and_voids_action(self) -> None:
        machine = self.new_machine(rounds_per_team=2, words_per_participant=5)
        ruling, _ = machine.submit(
            "red_player_1",
            Action.of("describe"),
            public_deliberation="one two three four five six seven",
        )
        self.assertEqual(ruling.rule_id, "time_expired_while_talking")
        self.assertEqual(machine.state.current_team, BLUE)
        self.assertEqual(machine.state.round_number, 2)
        self.assertEqual(machine.state.speech_left["blue_player_1"], 5)
        spoken = next(
            e.payload["public_deliberation"]
            for e in machine.log.events
            if e.type == "actor_responded"
        )
        self.assertEqual(spoken, "one two three four five…")

    def test_game_ends_after_all_rounds_with_points_winner(self) -> None:
        machine = self.new_machine(words_per_participant=6)
        target = machine.state.current_card().target
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="go")
        machine.submit("red_player_2", Action.of("guess", word=target), public_deliberation="")
        self.burn_describer(machine, "red_player_1")
        self.assertEqual(machine.state.current_team, BLUE)
        self.burn_describer(machine, "blue_player_1")
        self.assertTrue(machine.spec.is_terminal(machine.state))
        score = machine.score()
        self.assertEqual(score["winner"], RED)
        self.assertEqual(score["terminal_reason"], "most_points")
        self.assertIn("game_ended", [event.type for event in machine.log.events])

    def test_scoreless_tie_is_a_draw(self) -> None:
        machine = self.new_machine(words_per_participant=3)
        self.burn_describer(machine, "red_player_1")
        self.burn_describer(machine, "blue_player_1")
        self.assertTrue(machine.spec.is_terminal(machine.state))
        score = machine.score()
        self.assertIsNone(score["winner"])
        self.assertEqual(score["terminal_reason"], "draw")

    def test_multiword_guess_is_illegal(self) -> None:
        machine = self.new_machine()
        machine.submit("red_player_1", Action.of("describe"), public_deliberation="go")
        ruling, _ = machine.submit(
            "red_player_2", Action.of("guess", word="two words"), public_deliberation=""
        )
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "guess_must_be_one_word")


if __name__ == "__main__":
    unittest.main()
