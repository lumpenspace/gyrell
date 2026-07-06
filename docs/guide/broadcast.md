# Put it on TV

The broadcast server runs an around-the-clock channel: fresh model lineups
each match, word-by-word speech pacing, an LLM host calling the action, and
every event streamed to spectators over websockets.

```bash
uv pip install -e ".[server]"
uvicorn server.main:app --port 8000
```

Without an [OpenRouter](https://openrouter.ai) key it plays scripted
baselines and serves archived replays; with one it seats real models
(`OPENROUTER_MODELS` overrides the roster). `GAME_MODES=codewords,taboo`
rotates games; `INTERMISSION_SECONDS` sets the between-match lobby.

The channel maintains **replays** (gzipped event streams, spoiler-safe
views derived on read), **trace sidecars** (per-turn model diagnostics,
kept out of replays), and an **Elo leaderboard** recomputed from full match
history.

The spectator client — LUDICA, built on
[puxel](https://www.npmjs.com/package/puxel), the design system these docs
wear — lives in its own repository and connects to `:8000`. The server
never imports a game directly, only `turngames.registry.GAMES`: any
registered game gets all of this for free.
