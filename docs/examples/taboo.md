# Example: taboo

Source:
[`src/turngames/games/taboo/`](https://github.com/lumpenspace/gyrell/tree/main/src/turngames/games/taboo)

[Taboo](https://en.wikipedia.org/wiki/Taboo_(game)): a rotating describer
talks teammates toward a target word without saying it or any forbidden
word; a violation is a **buzz** — a scored game event, not an error. Each
speaker spends a personal word clock; the round ends when the describer's
runs dry.

Kernel features it demonstrates: rotating roles (`acting_seat` moves every
round), legality-as-gameplay via `handle_invalid`, per-seat clocks, and a
points-per-card reward shape. Registering it touched one game package, one
baseline actor, one prompter, and one `GameEntry` — server, replays,
leaderboard and client picked it up from there.
