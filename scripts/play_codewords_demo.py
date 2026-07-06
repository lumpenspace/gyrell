from __future__ import annotations

from dataclasses import dataclass, field

from turngames.core import Action, ActorInput, ActorOutput, StateMachine
from turngames.games.codewords import CodewordsConfig, CodewordsSpec
from turngames.games.codewords.types import ASSASSIN, BLUE, CLUEGIVER, GUESSER, NEUTRAL, RED

WORDS = (
    "apple",
    "cat",
    "car",
    "river",
    "poison",
    "banana",
    "dog",
    "train",
    "mountain",
    "queen",
    "grape",
    "horse",
    "plane",
    "desert",
    "crown",
    "orange",
    "wolf",
    "ship",
    "forest",
    "castle",
    "peach",
    "lion",
    "rocket",
    "island",
    "knight",
)

KEY = (
    RED,
    BLUE,
    RED,
    NEUTRAL,
    ASSASSIN,
    RED,
    BLUE,
    RED,
    NEUTRAL,
    BLUE,
    RED,
    BLUE,
    RED,
    NEUTRAL,
    BLUE,
    RED,
    BLUE,
    RED,
    NEUTRAL,
    BLUE,
    RED,
    BLUE,
    NEUTRAL,
    NEUTRAL,
    NEUTRAL,
)

TEAM_PLANS = {
    RED: [
        ("fruit", ("apple", "banana", "grape", "orange", "peach")),
        ("transit", ("car", "train", "plane", "ship")),
    ],
    BLUE: [
        ("animal", ("cat", "dog", "horse", "wolf", "lion")),
        ("royal", ("queen", "crown", "castle")),
    ],
}


@dataclass
class ScriptedCodewordsActors:
    """Scripted seats exercising the propose/confirm guessing protocol.

    Red plays cleanly: one guesser proposes each target with a one-word
    remark, the other confirms with a one-word agreement. Blue's first
    guesser rambles long enough to run out the word clock, so the final
    constrained choose falls to the quiet teammate.
    """

    plans: dict[str, list[tuple[str, tuple[str, ...]]]]
    pending_targets: dict[str, list[str]] = field(default_factory=dict)
    blue_rambled: bool = False
    strategized_turns: set = field(default_factory=set)

    def act(self, actor_input: ActorInput) -> ActorOutput:
        seat = actor_input.seat
        team_id = seat.team_id
        if team_id is None:
            raise RuntimeError("scripted actor requires a team seat")
        if seat.role == CLUEGIVER:
            return self._cluegiver_output(team_id, actor_input)
        if seat.role == GUESSER:
            return self._guesser_output(team_id, actor_input)
        raise RuntimeError(f"unsupported role: {seat.role}")

    def _cluegiver_output(self, team_id: str, actor_input: ActorInput) -> ActorOutput:
        # Open each turn thinking out loud about strategy before committing.
        turn = actor_input.observation.data.get("turn_number")
        if (team_id, turn) not in self.strategized_turns:
            self.strategized_turns.add((team_id, turn))
            remaining = len(self.plans[team_id])
            strategy = {
                (RED, True): "Plan: sweep the fruit first, transit later.",
                (RED, False): "One cluster left, keep it tight.",
                (BLUE, True): "Plan: animals first, then the royals.",
                (BLUE, False): "Royals now, watch the assassin.",
            }[(team_id, remaining > 1)]
            return ActorOutput(
                public_deliberation=strategy,
                action=Action.of("say"),
            )
        clue, targets = self.plans[team_id].pop(0)
        self.pending_targets[team_id] = list(targets)
        deliberation = {
            "fruit": "Fruit cluster.",
            "transit": "Transit route.",
            "animal": "Animal path.",
            "royal": "Royal trio.",
        }[clue]
        return ActorOutput(
            public_deliberation=deliberation,
            action=Action.of("give_clue", word=clue, count=len(targets)),
        )

    def _guesser_output(self, team_id: str, actor_input: ActorInput) -> ActorOutput:
        targets = self.pending_targets.get(team_id, [])
        data = actor_input.observation.data

        if actor_input.constraints.get("final_guess_one_word"):
            word = targets.pop(0) if targets else "castle"
            return ActorOutput(
                public_deliberation="",
                action=Action.of("choose", word=word),
            )

        if data.get("pending_proposal"):
            return ActorOutput(
                public_deliberation="agreed",
                action=Action.of("confirm"),
            )

        if not targets:
            return ActorOutput(
                public_deliberation="done",
                action=Action.of("stop"),
            )

        if team_id == BLUE and not self.blue_rambled:
            self.blue_rambled = True
            return ActorOutput(
                public_deliberation=(
                    "I compare every animal clue against the whole board slowly "
                    "before daring to commit"
                ),
                action=Action.of("say"),
            )

        word = targets.pop(0)
        current_clue = data.get("current_clue")
        clue_word = (
            current_clue.get("word", "clue") if isinstance(current_clue, dict) else "clue"
        )
        return ActorOutput(
            public_deliberation=str(clue_word),
            action=Action.of("propose", word=word),
        )


def make_actor_input(
    spec: CodewordsSpec,
    machine: StateMachine,
    seat_id: str,
) -> ActorInput:
    seat = machine.state.seat(seat_id)
    if seat is None:
        raise RuntimeError(f"unknown seat: {seat_id}")
    observation = machine.observe(seat_id)
    role_spec = spec.role_specs()[seat.role]
    constraints = {}
    if observation.data.get("final_guess_only"):
        constraints["final_guess_one_word"] = True
    return ActorInput(
        seat=seat,
        role=role_spec,
        observation=observation,
        time_left=observation.data.get("time_left"),
        constraints=constraints,
    )


def render_board(view: dict) -> str:
    cells = view["board"]["cells"]
    width = view["board"]["width"]
    lines = []
    for row_start in range(0, len(cells), width):
        row = []
        for cell in cells[row_start : row_start + width]:
            key = str(cell.get("key", "?"))[:1].upper()
            revealed = cell.get("revealed")
            marker = "*" if revealed else " "
            row.append(f"{cell['word']:<9} {key}{marker}")
        lines.append(" | ".join(row))
    return "\n".join(lines)


def print_events(events: list) -> None:
    for event in events:
        if event.type == "time_used":
            print(
                "  clock:"
                f" {event.payload['seat_id']} spent {event.payload['spent']}"
                f" ({event.payload['before']} -> {event.payload['after']})"
            )
        elif event.type == "timer_expired":
            print(
                "  clock: timer expired;"
                f" final word goes to {event.payload.get('final_guess_seat')}"
            )
        elif event.type == "clue_given":
            clue = event.payload["clue"]
            print(f"  clue: {clue['team_id']} says {clue['word']} {clue['count']}")
        elif event.type == "guess_proposed":
            print(f"  propose: {event.payload['seat_id']} suggests {event.payload['word']}")
        elif event.type == "guess_confirmed":
            print(f"  confirm: {event.payload['seat_id']} backs {event.payload['word']}")
        elif event.type == "guess_rejected":
            print(f"  reject: {event.payload['seat_id']} vetoes {event.payload['word']}")
        elif event.type == "word_revealed":
            print(
                "  reveal:"
                f" {event.payload['word']} -> {event.payload['assignment']}"
            )
        elif event.type == "guesser_stopped":
            print(f"  stop: {event.payload['team_id']} ends guessing")
        elif event.type == "game_ended":
            print(f"  game ended: {event.payload}")


def main() -> None:
    spec = CodewordsSpec()
    machine = StateMachine.new(
        spec,
        "demo-seed",
        CodewordsConfig(words=WORDS, key=KEY, turn_time_limit_words=14),
    )
    actors = ScriptedCodewordsActors(
        plans={team: list(plans) for team, plans in TEAM_PLANS.items()}
    )

    print("LUDICA Codewords demo")
    print("Each public-deliberation word spends one time unit.")
    print("Omniscient audience board: R red, B blue, N neutral, A assassin.")
    print(render_board(spec.observe_viewer(machine.state, "audience_omniscient")))
    print()

    while not spec.is_terminal(machine.state):
        seat_id = spec.acting_seat(machine.state)
        if seat_id is None:
            break
        actor_input = make_actor_input(spec, machine, seat_id)
        print(
            f"TURN {machine.state.turn_number}"
            f" | {machine.state.current_team}"
            f" | {seat_id}"
            f" | inject time_left={actor_input.time_left}"
        )
        if actor_input.constraints:
            print(f"  constraints: {actor_input.constraints}")

        output = actors.act(actor_input)
        if output.public_deliberation:
            print(f"  public deliberation: {output.public_deliberation}")
        else:
            print("  public deliberation: <silent final word>")
        print(f"  action: {output.action.type} {output.action.payload}")

        event_start = len(machine.log.events)
        ruling, _ = machine.submit(
            seat_id,
            output.action,
            public_deliberation=output.public_deliberation,
            raw_response=output.raw_response,
        )
        print(f"  ruling: {ruling.rule_id} legal={ruling.is_legal}")
        print_events(machine.log.events[event_start:])
        print(render_board(spec.observe_viewer(machine.state, "audience_omniscient")))
        print()

    print("Final score")
    print(machine.score())


if __name__ == "__main__":
    main()
