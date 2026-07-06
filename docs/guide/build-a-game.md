# Build a game

A game is a **spec**: one object implementing the `GameSpec` protocol from
`turngames.core`. The kernel's `StateMachine` owns event sequencing, logging
and replay; the spec owns legality, transitions, scoring, and visibility.

```python
class GameSpec(Protocol[StateT, ConfigT]):
    id: str

    def initial_state(self, seed: str, config: ConfigT) -> StateT: ...
    def seats(self, state: StateT) -> tuple[Seat, ...]: ...
    def acting_seat(self, state: StateT) -> str | None: ...
    def observe(self, state: StateT, seat_id: str) -> Observation: ...
    def legal_actions(self, state: StateT, seat_id: str) -> tuple[dict, ...]: ...
    def validate(self, state: StateT, seat_id: str, action: Action) -> Ruling: ...
    def step(self, state: StateT, seat_id: str, action: Action) -> StepTransition: ...
    def handle_invalid(self, ...) -> StepTransition: ...
    def is_terminal(self, state: StateT) -> bool: ...
    def score(self, log: EventLog, state: StateT) -> dict: ...
```

The load-bearing three: `observe` (per-seat visibility — hidden-information
games are projections), `validate`/`handle_invalid` (illegal moves are
*game events with costs*, not crashes — that's what lets flawed models play
unattended), and `score` (the verifiable reward, computed from the full
event log).

State is immutable; `step` returns the next state plus events to append.
The event log **is** the match — replays, the broadcast feed, and RL
rollouts all derive from it.

## Register and run

One entry in `turngames/registry.py` is the whole integration surface:

```python
GAMES["yourgame"] = GameEntry(
    id="yourgame",
    title="Your Game",
    spec_factory=YourGameSpec,
    config_factory=lambda labels: YourGameConfig(seat_labels=labels),
    seat_ids=("red_1", "red_2", "blue_1", "blue_2"),
    baseline_factory=lambda seed: BaselineYourGameActor(seed=seed),
    prompter_factory=YourGamePrompter,
)
```

`scripts/play_codewords_demo.py` runs a whole match in the terminal — the
fastest way to watch a new spec breathe.
