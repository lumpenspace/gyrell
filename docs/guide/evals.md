# Run evals

Games make sharp evals: no rubric, no LLM judge — a match has a winner.
GyReLL gives you three instruments, from single-model to full-field.

## Tournaments

`turngames.adapters.tournament` plays a systematic schedule of head-to-head
matches — no pacing, matches in parallel, as fast as the APIs respond:

```bash
PYTHONPATH=src python3 scripts/tournament.py \
    --models anthropic/claude-haiku-4.5,openai/gpt-5.4-nano,google/gemini-2.5-flash \
    --schemes uniform,mixed --guessers 1 --workers 8 \
    --out replays/tournaments/rr1
```

Two schemes, each pairing played in **both color assignments** (the
first-moving team has an edge worth cancelling):

- **`uniform`** — each team seated entirely by one model, every unordered
  pair meeting twice. Raw head-to-head strength.
- **`mixed`** — for each pair {A, B}, one team seats A as cluegiver with B
  guessing, the other B cluing with A guessing. The same two models on both
  sides: what differs is who leads, so the scheme isolates **role fit and
  cross-model cooperation** rather than raw strength.

Slugs without a provider prefix (e.g. `baseline`) seat the scripted actor —
free offline dry runs of a schedule. From code:

```python
from turngames.adapters.tournament import run_tournament

results, table = run_tournament(models, api_key=key,
                                schemes=("uniform", "mixed"), workers=8)
```

## What you get

The runner prints a standings table (wins, losses, win rate, cluegiver
record) and writes three artifacts to `--out`:

- **`tournament-results.json`** — every fixture with lineup, winner, act
  count and fallback counts, plus the standings.
- **Replays** — each match in the standard archive format, so the broadcast
  server can serve them and LUDICA can replay any tournament match.
- **Trace sidecars** — per-turn raw responses, hidden-reasoning text and
  token counts, and fallback causes, kept out of the replays.

Fallbacks are first-class: a seat that answers illegally or errors twice in
a row is played by its scripted baseline for that stretch, the match
finishes regardless, and `fallback_turns` tells you exactly how much of a
model's record it actually played.

Example — four models, 24 matches (12 uniform + 12 mixed), ~25 minutes at
8 workers:

```text
model                        W-L-D   win%  as cluegiver   Elo   fallback
anthropic/claude-haiku-4.5   10-8-0   56%   7/12          1072  1%
openai/gpt-5.4-nano          10-8-0   56%   8/12          1039  4%
deepseek/deepseek-chat-v3.1  8-10-0   44%   4/12           970  3%
google/gemini-2.5-flash      8-10-0   44%   5/12           923  2%
```

## The reporter

Pass `--reporter <model-slug>` and a model reads each finished match's
public transcript and flags the moments worth a human's attention —
brilliant clues, blunders, miscoordination, near-misses — into
`reports/<seed>.json` per match and a `tournament-report.md` digest.
Strictly color commentary: reporter output never touches records, Elo, or
standings, so the eval stays judge-free. `--report-only` runs it over an
existing tournament directory after the fact.

```text
## trn-…-10 — deepseek-chat-v3.1 vs claude-haiku-4.5
- blunder: "…Heaven's got room for one more—ring for halo feels right. /
  RING revealed -> assassin" — Red's final clue triggers the assassin,
  gifting blue the win after a dominant performance.
```

## Elo over any replay directory

The leaderboard aggregates any directory of replays — live matches and
tournament output alike:

```python
from pathlib import Path
from server.leaderboard import aggregate

board = aggregate(Path("replays/tournaments/rr1"))
for row in board["models"]:
    print(row["model"], row["elo"], row["total"])
```

Ratings are recomputed from full history in play order (provisional under
10 matches, draws count half), with splits per game, per role, and per
team composition.

## Single-model evals

For "how well does one model play against a fixed opponent", use the
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment
directly — same engine, dataset-shaped:

```bash
cd environments/gyrell && uv pip install -e .
vf-eval gyrell -m openai/gpt-4.1-mini -n 20 -r 2
```

See [Train on it](rl-environment.md) and
[Prime Intellect](../prime-intellect.md) for the environment's knobs and
hub install.
