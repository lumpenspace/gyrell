"""Record a scripted Codewords match to a JSONL replay file.

This is milestone 1 of the spectator app plan (see docs/spectator-app.md):
produce the fixture the replay viewer builds against, without duplicating any
game logic — it just drives the existing StateMachine/CodewordsSpec and dumps
what observe_viewer() and the EventLog already compute.

Usage:
    PYTHONPATH=src python3 scripts/record_codewords_replay.py [output_path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from play_codewords_demo import KEY, TEAM_PLANS, WORDS, ScriptedCodewordsActors, make_actor_input  # noqa: E402

from turngames.core import StateMachine  # noqa: E402
from turngames.games.codewords import CodewordsConfig, CodewordsSpec  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "replays" / "demo-codewords.jsonl"

# Viewer profiles snapshotted after every step. "spoiler_safe" is what a
# public spectator client should default to; "audience_omniscient" is kept
# alongside it for the archive/postgame-reveal case (see spec.md's Codewords
# visibility profiles and Broadcast Boundaries in flows.md).
SNAPSHOT_VIEWERS = ("spoiler_safe", "audience_omniscient")


def event_to_json(event) -> dict:
    return {
        "seq": event.seq,
        "type": event.type,
        "payload": event.payload,
        "visibility": list(event.visibility),
    }


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spec = CodewordsSpec()
    machine = StateMachine.new(
        spec,
        "demo-seed",
        CodewordsConfig(
            words=WORDS,
            key=KEY,
            turn_time_limit_words=24,
            seat_labels={
                "red_cluegiver": "scripted-v1",
                "red_guesser_1": "scripted-v1",
                "red_guesser_2": "scripted-v1",
                "blue_cluegiver": "scripted-v1",
                "blue_guesser_1": "scripted-v1",
                "blue_guesser_2": "scripted-v1",
            },
        ),
    )
    actors = ScriptedCodewordsActors(
        plans={team: list(plans) for team, plans in TEAM_PLANS.items()}
    )

    records: list[dict] = [
        {
            "kind": "meta",
            "game_id": spec.id,
            "seed": machine.state.seed,
        }
    ]

    def snapshot_records() -> list[dict]:
        return [
            {"kind": "snapshot", "viewer": viewer, "view": spec.observe_viewer(machine.state, viewer)}
            for viewer in SNAPSHOT_VIEWERS
        ]

    records.extend(snapshot_records())

    while not spec.is_terminal(machine.state):
        seat_id = spec.acting_seat(machine.state)
        if seat_id is None:
            break
        actor_input = make_actor_input(spec, machine, seat_id)
        output = actors.act(actor_input)

        event_start = len(machine.log.events)
        machine.submit(
            seat_id,
            output.action,
            public_deliberation=output.public_deliberation,
            raw_response=output.raw_response,
        )

        for event in machine.log.events[event_start:]:
            if "audience" in event.visibility:
                records.append({"kind": "event", "event": event_to_json(event)})
        records.extend(snapshot_records())

    records.append({"kind": "final_score", "score": machine.score()})

    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
