import unittest

from turngames.core import Action, StateMachine
from turngames.games.codewords import CodewordsConfig, CodewordsSpec


class StateMachineTest(unittest.TestCase):
    def test_state_machine_logs_actor_ruling_and_transition_events(self) -> None:
        machine = StateMachine.new(
            CodewordsSpec(),
            "log-seed",
            CodewordsConfig(),
        )

        machine.submit("red_cluegiver", Action.of("give_clue", word="space", count=2))
        event_types = [event.type for event in machine.log.events]

        self.assertIn("game_initialized", event_types)
        self.assertIn("actor_responded", event_types)
        self.assertIn("ruling_issued", event_types)
        self.assertIn("clue_given", event_types)


if __name__ == "__main__":
    unittest.main()

