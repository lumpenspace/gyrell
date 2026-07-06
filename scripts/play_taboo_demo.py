"""Run a full baseline-vs-baseline Taboo match and print the play-by-play.

Usage:
    PYTHONPATH=src python scripts/play_taboo_demo.py [seed]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turngames.core import ActorInput, StateMachine
from turngames.registry import GAMES


def main() -> None:
    seed = sys.argv[1] if len(sys.argv) > 1 else "taboo-demo"
    entry = GAMES["taboo"]
    spec = entry.spec_factory()
    machine = StateMachine.new(spec, seed, entry.config_factory(None))
    role_specs = spec.role_specs()
    actors = {seat_id: entry.baseline_factory(f"{seed}:{seat_id}") for seat_id in entry.seat_ids}

    while not spec.is_terminal(machine.state):
        seat_id = spec.acting_seat(machine.state)
        if seat_id is None:
            break
        seat = machine.state.seat(seat_id)
        observation = machine.observe(seat_id)
        output = actors[seat_id].act(
            ActorInput(
                seat=seat,
                role=role_specs[seat.role],
                observation=observation,
                time_left=observation.data.get("time_left"),
            )
        )
        event_start = len(machine.log.events)
        machine.submit(seat_id, output.action, public_deliberation=output.public_deliberation)
        for event in machine.log.events[event_start:]:
            if "audience" not in event.visibility:
                continue
            p = event.payload
            if event.type == "actor_responded" and p.get("public_deliberation"):
                print(f"  {p['seat_id']}: {p['public_deliberation']}")
            elif event.type == "card_guessed":
                print(
                    f"* {p['seat_id']} got {p['card']['target'].upper()} — "
                    f"red {p['points']['red']} · blue {p['points']['blue']}"
                )
            elif event.type == "taboo_violation":
                print(f"! BUZZ: {p['seat_id']} said {p['said']!r} — {p['card']['target'].upper()} lost")
            elif event.type == "card_skipped":
                print(f"- {p['seat_id']} skips {p['card']['target'].upper()}")
            elif event.type == "round_ended":
                print(
                    f"== round {p['round_number']} over — "
                    f"red {p['points']['red']} · blue {p['points']['blue']} =="
                )

    print()
    print("final:", machine.score())


if __name__ == "__main__":
    main()
