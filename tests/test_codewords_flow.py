import unittest

from turngames.core import Action, StateMachine
from turngames.games.codewords import CodewordsConfig, CodewordsSpec
from turngames.games.codewords.types import BLUE, CLUEGIVER, GUESSER, GUESSING, RED


WORDS = tuple(f"word{i}" for i in range(25))
KEY = (
    RED,
    RED,
    BLUE,
    "neutral",
    "assassin",
    RED,
    BLUE,
    "neutral",
    RED,
    BLUE,
    "neutral",
    RED,
    BLUE,
    "neutral",
    RED,
    BLUE,
    "neutral",
    RED,
    BLUE,
    "neutral",
    RED,
    BLUE,
    "neutral",
    RED,
    BLUE,
)


class CodewordsFlowTest(unittest.TestCase):
    def new_machine(self, **config_kwargs) -> StateMachine:
        config = CodewordsConfig(words=WORDS, key=KEY, starting_team=RED, **config_kwargs)
        return StateMachine.new(CodewordsSpec(), "test-seed", config)

    def test_cluegiver_sees_key_and_guesser_does_not(self) -> None:
        machine = self.new_machine()

        cluegiver_obs = machine.observe("red_cluegiver")
        guesser_obs = machine.observe("red_guesser_1")

        cluegiver_cell = cluegiver_obs.data["board"]["cells"][0]
        guesser_cell = guesser_obs.data["board"]["cells"][0]
        self.assertEqual(cluegiver_cell["word"], "word0")
        self.assertEqual(cluegiver_cell["key"], RED)
        self.assertNotIn("key", guesser_cell)
        self.assertIsNone(guesser_cell["revealed"])

    def test_each_team_has_configured_guessers(self) -> None:
        machine = self.new_machine(guessers_per_team=3)
        guesser_ids = [seat.id for seat in machine.state.seats if seat.role == GUESSER]
        self.assertEqual(
            guesser_ids,
            [
                "red_guesser_1",
                "red_guesser_2",
                "red_guesser_3",
                "blue_guesser_1",
                "blue_guesser_2",
                "blue_guesser_3",
            ],
        )

    def test_role_protocols_define_expected_actions(self) -> None:
        specs = CodewordsSpec().role_specs()

        clue_actions = [schema.type for schema in specs[CLUEGIVER].action_schemas]
        guess_actions = [schema.type for schema in specs[GUESSER].action_schemas]

        self.assertEqual(clue_actions, ["say", "give_clue"])
        self.assertEqual(
            guess_actions,
            ["say", "propose", "confirm", "reject", "choose", "stop"],
        )

    def test_omniscient_audience_sees_key(self) -> None:
        spec = CodewordsSpec()
        state = spec.initial_state("test-seed", CodewordsConfig(words=WORDS, key=KEY))

        safe = spec.observe_viewer(state, "audience_spoiler_safe")
        omni = spec.observe_viewer(state, "audience_omniscient")

        self.assertNotIn("key", safe["board"]["cells"][0])
        self.assertEqual(omni["board"]["cells"][0]["key"], RED)

    def test_reveal_requires_propose_then_confirm_by_other_guesser(self) -> None:
        machine = self.new_machine()

        ruling, _ = machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=2),
        )
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.phase, GUESSING)
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_1")

        ruling, _ = machine.submit("red_guesser_1", Action.of("propose", word="word0"))
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.pending_proposal.word, "word0")
        # Nothing revealed yet; the other guesser must act now.
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_2")
        self.assertEqual(len(machine.state.board.revealed), 0)

        ruling, rewards = machine.submit("red_guesser_2", Action.of("confirm"))
        self.assertTrue(ruling.is_legal)
        self.assertEqual(len(machine.state.board.revealed), 1)
        self.assertIsNone(machine.state.pending_proposal)
        # Reward credit goes to the proposer.
        self.assertEqual(rewards[0].seat_id, "red_guesser_1")
        self.assertEqual(rewards[0].value, 1.0)
        # The confirmer keeps the floor and may propose the next guess.
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_2")
        ruling, _ = machine.submit("red_guesser_2", Action.of("propose", word="word1"))
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.pending_proposal.word, "word1")
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_1")

    def test_proposer_cannot_confirm_own_proposal(self) -> None:
        machine = self.new_machine()
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=2))
        machine.submit("red_guesser_1", Action.of("propose", word="word0"))

        ruling, _ = machine.submit("red_guesser_1", Action.of("confirm"))

        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "wrong_acting_seat")
        self.assertEqual(len(machine.state.board.revealed), 0)

    def test_reject_clears_proposal_and_rejecter_acts_next(self) -> None:
        machine = self.new_machine()
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=2))
        machine.submit("red_guesser_1", Action.of("propose", word="word3"))

        ruling, _ = machine.submit("red_guesser_2", Action.of("reject"))

        self.assertTrue(ruling.is_legal)
        self.assertIsNone(machine.state.pending_proposal)
        self.assertEqual(len(machine.state.board.revealed), 0)
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_2")

    def test_say_rotates_speaker(self) -> None:
        machine = self.new_machine()
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=2))

        machine.submit("red_guesser_1", Action.of("say"), public_deliberation="hmm")
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_2")
        machine.submit("red_guesser_2", Action.of("say"), public_deliberation="agreed")
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_1")

    def test_illegal_clue_ends_turn_by_policy(self) -> None:
        machine = self.new_machine()

        ruling, rewards = machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="word0", count=1),
        )

        self.assertFalse(ruling.is_legal)
        self.assertEqual(rewards[0].name, "illegal_action")
        self.assertEqual(machine.state.current_team, BLUE)
        self.assertEqual(machine.state.illegal_clues[RED], 1)

    def test_public_deliberation_spends_turn_time(self) -> None:
        machine = self.new_machine(turn_time_limit_words=5)

        machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=1),
            public_deliberation="one two three",
        )

        self.assertEqual(machine.state.time_left, 2)

    def test_timeout_final_word_goes_to_quietest_guesser(self) -> None:
        machine = self.new_machine(turn_time_limit_words=6)
        machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=2),
            public_deliberation="one",
        )

        # Guesser 1 burns the rest of the clock talking.
        ruling, _ = machine.submit(
            "red_guesser_1",
            Action.of("say"),
            public_deliberation="two three four five six seven",
        )
        self.assertTrue(machine.state.final_guess_only)
        # The quiet teammate gets the final one-word choose.
        self.assertEqual(machine.state.final_guess_seat, "red_guesser_2")
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_2")
        self.assertEqual(
            machine.observe("red_guesser_2").allowed_actions[0]["type"], "choose"
        )

        # The talker cannot take the final guess.
        ruling, _ = machine.submit("red_guesser_1", Action.of("choose", word="word0"))
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "wrong_acting_seat")

        ruling, _ = machine.submit(
            "red_guesser_2",
            Action.of("choose", word="word0"),
            public_deliberation="",
        )
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.current_team, BLUE)

    def test_cluegiver_can_think_out_loud_before_clue(self) -> None:
        machine = self.new_machine(turn_time_limit_words=10)

        ruling, _ = machine.submit(
            "red_cluegiver",
            Action.of("say"),
            public_deliberation="going for the fruit cluster",
        )

        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.phase, "clue_proposal")
        self.assertEqual(machine.state.time_left, 5)

        ruling, _ = machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=2),
        )
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.phase, GUESSING)

    def test_solo_guesser_chooses_directly(self) -> None:
        machine = self.new_machine(guessers_per_team=1)
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=2))

        allowed = [a["type"] for a in machine.observe("red_guesser_1").allowed_actions]
        self.assertEqual(allowed, ["say", "choose", "stop"])

        ruling, _ = machine.submit("red_guesser_1", Action.of("propose", word="word0"))
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "solo_guesser_must_choose")

        ruling, rewards = machine.submit("red_guesser_1", Action.of("choose", word="word0"))
        self.assertTrue(ruling.is_legal)
        self.assertEqual(len(machine.state.board.revealed), 1)
        self.assertEqual(rewards[0].seat_id, "red_guesser_1")
        # Correct guess with count left: the same guesser keeps choosing.
        self.assertEqual(machine.state.current_team, RED)
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_1")

        ruling, _ = machine.submit("red_guesser_1", Action.of("choose", word="word3"))
        self.assertTrue(ruling.is_legal)
        # Neutral tile ends the turn.
        self.assertEqual(machine.state.current_team, BLUE)

    def test_multi_guesser_team_cannot_choose_before_timeout(self) -> None:
        machine = self.new_machine()
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=2))

        ruling, _ = machine.submit("red_guesser_1", Action.of("choose", word="word0"))

        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "choose_requires_solo_guesser")
        self.assertEqual(len(machine.state.board.revealed), 0)

    def test_bonus_guess_reached_announced_on_extra_guess(self) -> None:
        # A clue of 1 pays for one card; the standard "+1" lets the team try
        # once more. The moment that extra try becomes available is announced.
        machine = self.new_machine(turn_time_limit_words=40)
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=1))

        machine.submit("red_guesser_1", Action.of("propose", word="word0"))  # RED
        machine.submit("red_guesser_2", Action.of("confirm"))

        # Turn continues onto the bonus guess, and it is flagged in the log.
        self.assertEqual(machine.state.current_team, RED)
        self.assertEqual(machine.state.phase, GUESSING)
        self.assertEqual(machine.state.guesses_this_turn, 1)
        self.assertIn("bonus_guess_reached", [e.type for e in machine.log.events])

    def test_bonus_guess_is_a_snap_decision(self) -> None:
        # No propose/reject cycles to stall in: the bonus guess is choose or
        # stop, and anything spoken is clipped to two words.
        machine = self.new_machine(turn_time_limit_words=40)
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=1))
        machine.submit("red_guesser_1", Action.of("propose", word="word0"))
        machine.submit("red_guesser_2", Action.of("confirm"))

        allowed = {a["type"] for a in machine.spec.legal_actions(machine.state, "red_guesser_2")}
        self.assertEqual(allowed, {"choose", "stop"})

        ruling, _ = machine.submit("red_guesser_2", Action.of("propose", word="word1"))
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "bonus_guess_must_choose_or_stop")

        ruling, _ = machine.submit(
            "red_guesser_2",
            Action.of("choose", word="word1"),
            public_deliberation="going for one more because I am confident",
        )
        self.assertTrue(ruling.is_legal)
        spoken = [
            e.payload["public_deliberation"]
            for e in machine.log.events
            if e.type == "actor_responded" and e.payload.get("public_deliberation")
        ]
        self.assertEqual(spoken[-1], "going for…")

    def test_bonus_guess_stop_banks_the_clue(self) -> None:
        machine = self.new_machine(turn_time_limit_words=40)
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=1))
        machine.submit("red_guesser_1", Action.of("propose", word="word0"))
        machine.submit("red_guesser_2", Action.of("confirm"))
        ruling, _ = machine.submit("red_guesser_2", Action.of("stop"))
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.current_team, BLUE)

    def test_no_bonus_clue_by_default_even_with_clock(self) -> None:
        # Bonus *clue* is off by default: after the guesses (including the free
        # +1) are spent, the turn passes to the other team even with clock left.
        machine = self.new_machine(turn_time_limit_words=40)
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=1))

        machine.submit("red_guesser_1", Action.of("propose", word="word0"))  # RED
        machine.submit("red_guesser_2", Action.of("confirm"))
        machine.submit("red_guesser_2", Action.of("choose", word="word1"))  # RED (bonus)

        self.assertEqual(machine.state.current_team, BLUE)
        self.assertNotIn("bonus_clue_granted", [e.type for e in machine.log.events])

    def test_bonus_clue_after_clean_multi_guess_when_enabled(self) -> None:
        # The bonus-clue mechanic still works when explicitly enabled.
        machine = self.new_machine(
            turn_time_limit_words=40, bonus_clue_when_time_left=True
        )
        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=1))

        # Two correct guesses (count + 1) with plenty of clock left; the
        # bonus guess is a direct choose by whoever holds the floor.
        machine.submit("red_guesser_1", Action.of("propose", word="word0"))
        machine.submit("red_guesser_2", Action.of("confirm"))
        machine.submit("red_guesser_2", Action.of("choose", word="word1"))

        # Turn does not end: same team, back to clue proposal, clock intact.
        self.assertEqual(machine.state.current_team, RED)
        self.assertEqual(machine.state.phase, "clue_proposal")
        self.assertIsNone(machine.state.current_clue)
        self.assertIn(
            "bonus_clue_granted", [e.type for e in machine.log.events]
        )

        ruling, _ = machine.submit(
            "red_cluegiver", Action.of("give_clue", word="encore", count=1)
        )
        self.assertTrue(ruling.is_legal)
        self.assertEqual(machine.state.phase, GUESSING)

    def test_no_bonus_clue_without_clock(self) -> None:
        machine = self.new_machine(bonus_clue_when_time_left=True)  # but no clock

        machine.submit("red_cluegiver", Action.of("give_clue", word="start", count=1))
        machine.submit("red_guesser_1", Action.of("propose", word="word0"))
        machine.submit("red_guesser_2", Action.of("confirm"))
        machine.submit("red_guesser_2", Action.of("choose", word="word1"))  # bonus

        self.assertEqual(machine.state.current_team, BLUE)

    def test_action_voided_when_own_speech_expires_clock(self) -> None:
        machine = self.new_machine(turn_time_limit_words=6)
        machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=2),
            public_deliberation="one",
        )

        # Guesser 1 talks through the rest of the clock while proposing; the
        # final word falls to the quiet teammate, so the proposal is voided
        # without an infraction.
        ruling, _ = machine.submit(
            "red_guesser_1",
            Action.of("propose", word="word0"),
            public_deliberation="two three four five six seven",
        )

        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "time_expired_while_talking")
        self.assertEqual(ruling.severity, "info")
        self.assertIsNone(machine.state.pending_proposal)
        self.assertEqual(machine.state.active_guesser_id(), "red_guesser_2")
        voided = [e for e in machine.log.events if e.type == "action_voided"]
        self.assertEqual(len(voided), 1)
        self.assertNotIn(
            "illegal_action_applied", [e.type for e in machine.log.events]
        )

    def test_overlong_speech_is_clipped_and_void_hidden_from_audience(self) -> None:
        machine = self.new_machine(turn_time_limit_words=6)
        machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=2),
            public_deliberation="one",
        )

        # Five words of clock remain; the guesser rambles for eight.
        machine.submit(
            "red_guesser_1",
            Action.of("propose", word="word0"),
            public_deliberation="alpha beta gamma delta epsilon zeta eta theta",
        )

        responded = [
            e
            for e in machine.log.events
            if e.type == "actor_responded"
            and e.payload["seat_id"] == "red_guesser_1"
        ]
        self.assertEqual(len(responded), 1)
        # Cut to the five remaining words, trailing off with an ellipsis.
        self.assertEqual(
            responded[0].payload["public_deliberation"],
            "alpha beta gamma delta epsilon…",
        )
        # The clock still expires and the quiet teammate gets the final word.
        self.assertTrue(machine.state.final_guess_only)
        self.assertEqual(machine.state.final_guess_seat, "red_guesser_2")
        # The void is recorded for replay but kept out of the audience feed.
        voided = [e for e in machine.log.events if e.type == "action_voided"]
        self.assertEqual(len(voided), 1)
        self.assertNotIn("audience", voided[0].visibility)

    def test_speech_within_clock_is_left_untouched(self) -> None:
        machine = self.new_machine(turn_time_limit_words=10)
        machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=2),
            public_deliberation="one",
        )
        machine.submit(
            "red_guesser_1",
            Action.of("say"),
            public_deliberation="let us try the fruit",
        )
        responded = [
            e
            for e in machine.log.events
            if e.type == "actor_responded"
            and e.payload["seat_id"] == "red_guesser_1"
        ]
        self.assertEqual(
            responded[0].payload["public_deliberation"], "let us try the fruit"
        )
        self.assertFalse(machine.state.final_guess_only)

    def test_timeout_cannot_stop_or_propose(self) -> None:
        machine = self.new_machine(turn_time_limit_words=4)
        machine.submit(
            "red_cluegiver",
            Action.of("give_clue", word="start", count=2),
            public_deliberation="one",
        )
        machine.submit(
            "red_guesser_1",
            Action.of("say"),
            public_deliberation="two three four",
        )
        self.assertTrue(machine.state.final_guess_only)

        seat = machine.state.final_guess_seat
        ruling, _ = machine.submit(seat, Action.of("stop"))
        self.assertFalse(ruling.is_legal)
        self.assertEqual(ruling.rule_id, "final_guess_must_choose")

        ruling, _ = machine.submit(seat, Action.of("choose", word="word0"))
        self.assertTrue(ruling.is_legal)


if __name__ == "__main__":
    unittest.main()
