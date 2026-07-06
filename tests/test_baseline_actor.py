import unittest

from turngames.actors import BaselineCodewordsActor
from turngames.core import ActorInput, StateMachine
from turngames.games.codewords import CodewordsConfig, CodewordsSpec


def run_match(seed: str, **config_kwargs) -> StateMachine:
    spec = CodewordsSpec()
    machine = StateMachine.new(spec, seed, CodewordsConfig(**config_kwargs))
    actors = {
        seat.id: BaselineCodewordsActor(seed=f"{seed}:{seat.id}")
        for seat in machine.state.seats
        if seat.role in ("cluegiver", "guesser")
    }
    role_specs = spec.role_specs()
    for _ in range(2000):
        if spec.is_terminal(machine.state):
            break
        seat_id = spec.acting_seat(machine.state)
        seat = machine.state.seat(seat_id)
        observation = machine.observe(seat_id)
        constraints = {}
        if observation.data.get("final_guess_only"):
            constraints["final_guess_one_word"] = True
        output = actors[seat_id].act(
            ActorInput(
                seat=seat,
                role=role_specs[seat.role],
                observation=observation,
                time_left=observation.data.get("time_left"),
                constraints=constraints,
            )
        )
        ruling, _ = machine.submit(
            seat_id,
            output.action,
            public_deliberation=output.public_deliberation,
        )
        assert ruling.is_legal, f"{seat_id} played illegal {output.action}: {ruling.message}"
    return machine


class BaselineActorTest(unittest.TestCase):
    def test_baseline_actors_finish_a_dealt_game_legally(self) -> None:
        machine = run_match("baseline-seed", turn_time_limit_words=40)
        self.assertTrue(CodewordsSpec().is_terminal(machine.state))
        self.assertIn(machine.state.winner, ("red", "blue"))

    def test_baseline_actors_are_deterministic_per_seed(self) -> None:
        first = run_match("det-seed")
        second = run_match("det-seed")
        self.assertEqual(
            [e.type for e in first.log.events],
            [e.type for e in second.log.events],
        )
        self.assertEqual(first.state.winner, second.state.winner)

    def test_baseline_actors_handle_solo_guessers(self) -> None:
        machine = run_match("solo-seed", guessers_per_team=1)
        self.assertTrue(CodewordsSpec().is_terminal(machine.state))


if __name__ == "__main__":
    unittest.main()
