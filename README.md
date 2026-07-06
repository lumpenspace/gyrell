# gyrell

![GyReLL — gyre's RL environment for ludic learning](docs/assets/cover.webp)

**gyre's RL environment for ludic learning.**

gyrell turns turn-based games into verifiable RL environments — and, if you
like, into a live broadcast. Describe a game once as a `turngames` spec; the
same spec drives scripted baselines, LLM-seated matches,
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) environments,
and an around-the-clock spectator show. The reward never needs a judge: you
either won the game or you didn't. Commerce is not our goal here at gyrell;
verifiable play is.

Two games ship as worked examples — **codewords** (Codenames-style hidden
information) and **taboo**. **[The docs are a guide to adding
yours.](https://lumpenspace.github.io/gyrell/)**

- **`src/turngames`** — the kernel: state machine, event log, per-seat
  visibility, word clocks, scripted and LLM actors.
- **`server/`** — the broadcast channel: rotating lineups, an LLM host,
  archived replays, an Elo leaderboard. Pairs with the LUDICA spectator
  client (separate repository).
- **`environments/gyrell`** — the [verifiers](https://github.com/PrimeIntellect-ai/verifiers)
  environment, published as [`lumpenspace/gyrell`](https://app.primeintellect.ai/dashboard/environments/lumpenspace/gyrell)
  on the [Environments Hub](https://app.primeintellect.ai/dashboard/environments),
  with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) configs in
  [`configs/`](configs/).

## Quickstart

```bash
uv venv && uv pip install -e ".[server]"
python scripts/play_codewords_demo.py     # scripted match in the terminal
uvicorn server.main:app --port 8000       # broadcast (replays need no keys;
                                          # live models need OPENROUTER_API_KEY)
pytest tests/                             # the house Voight-Kampff
```

## Name

*gyrell*: gyre + RL + ludic learning. Any resemblance to corporations that
manufactured beings more human than human is, of course, aspirational.
