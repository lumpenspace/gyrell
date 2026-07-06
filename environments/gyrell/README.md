# gyrell (env)

### Overview
- **Environment ID**: `gyrell` (hub: `lumpenspace/gyrell`)
- **Short description**: Codenames-style hidden-information word game. The policy plays the red team — cluegiver and guesser seats, each with its own private conversation — against a scripted opponent team.
- **Tags**: game, multi-turn, hidden-information, train, eval

### Datasets
- **Primary dataset(s)**: procedurally generated boards from the `turngames` Codewords engine (`classic-en` deck), seeded and fully deterministic. Each example is one game: a distinct board, key, and starting team (alternating first/second move).
- **Source links**: [lumpenspace/codenames-llms](https://github.com/lumpenspace/codenames-llms) — the same engine drives the live broadcast arena, replays, and this environment.
- **Split sizes**: configurable; defaults to 2000 train / 200 eval.

### Task
- **Type**: multi-turn (non-linear: per-seat private message histories)
- **Output format expectations**: one JSON object per turn, no fences:
  `{"public_deliberation": "...", "action": {"type": "give_clue", "word": "ocean", "count": 2}}`
- **Rubric overview**: trajectory-level rewards from the final game outcome — earned win (1.0), earned own-word progress (0.3), penalty-only legality (0.2) and format (0.1) terms — plus zero-weight metrics (raw wins, assassin losses, game completion, turns played). See Metrics below for the rationale.

### How it works
The trained policy always plays the **red** team. With `role="team"` (default) it
occupies both the cluegiver and guesser seats; because the cluegiver sees the
hidden key and the guesser must not, **each seat keeps a private conversation**
(implemented via the `get_prompt_messages` override — trajectory steps
interleave seats, and the guesser's context never contains the key).

Every seat the policy does not control is played by a configurable driver:

- `"scripted"` (default): the deterministic random-but-legal baseline from
  `turngames` — cheap and stationary, good for early training;
- `"policy"`: the same model/client being trained (frozen for those turns —
  they are environment transitions, not trajectory steps), i.e. self-play;
- an **endpoint spec**: any OpenAI-compatible endpoint, e.g.
  `{"model": "openai/gpt-4.1-mini", "base_url": "https://openrouter.ai/api/v1",
  "api_key_var": "OPENROUTER_API_KEY", "temperature": 0.7, "max_tokens": 400}`.

`opponent` sets the driver for the blue team; `partner` sets it for red seats
the policy does not control (relevant for `role="cluegiver"`/`"guesser"` —
e.g. train the cluegiver with a fixed strong guesser). Both also accept:

- a **role-split dict** assigning different models per role —
  `{"cluegiver": {"model": "anthropic/claude-haiku-4.5"}, "guesser": {"model": "openai/gpt-4.1-mini"}}`;
- a **pool** (list) of drivers/role-splits — one matchup is chosen per
  example, deterministically from the board seed, so all GRPO rollouts of an
  example face the same opponents. The chosen matchup is recorded in
  `game_result["seat_models"]` for per-matchup analysis.

LLM-driven seats keep their own private conversations (a blue guesser never
sees any key), and any unusable reply or endpoint failure falls back to the
scripted baseline for that turn (counted in the `other_seat_fallbacks`
metric), so a misbehaving opponent can never stall a rollout.

Illegal actions (e.g. a clue matching a board word) are not silently repaired:
the canonical arbiter rules on them, applies the in-game penalty (an illegal
clue ends the turn), and the ruling is reported back to the policy — so
legality is learned, not enforced by the harness. Unparseable replies get a
format reminder and count against `format_compliance`.

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval gyrell
```

Configure model and sampling (OpenRouter example, as used in this repo):

```bash
uv run vf-eval gyrell \
  --provider openrouter -m "openai/gpt-4.1-mini" \
  -n 5 -r 2 --max-tokens 400 -T 0.7 \
  -a '{"role": "team", "num_eval_examples": 20, "max_turns": 40}'
```

Play against a real LLM opponent instead of the scripted baseline:

```bash
uv run vf-eval gyrell \
  --provider openrouter -m "openai/gpt-4.1-mini" \
  -n 5 -r 2 --max-tokens 400 -T 0.7 \
  -a '{"opponent": {"model": "anthropic/claude-haiku-4.5", "base_url": "https://openrouter.ai/api/v1", "api_key_var": "OPENROUTER_API_KEY"}}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `role` | str | `"team"` | Seats the policy controls: `team` (cluegiver + guessers), `cluegiver`, or `guesser`. |
| `opponent` | str \| dict \| list | `"scripted"` | Driver(s) for the blue team: `"scripted"`, `"policy"` (self-play with the trained model's own client), an endpoint spec, a role-split dict, or a pool (list) of these — one matchup per example (see above). |
| `partner` | str \| dict \| list | `"scripted"` | Driver(s) for red seats the policy does not control (only used with `role="cluegiver"`/`"guesser"`); same forms as `opponent`. |
| `guessers_per_team` | int | `1` | `1` uses the simple choose/stop action space; `2+` enables the propose/confirm protocol between guesser seats. |
| `deck` | str | `"classic-en"` | Word deck from `turngames`. |
| `num_train_examples` | int | `2000` | Number of seeded boards in the train split. |
| `num_eval_examples` | int | `200` | Number of seeded boards in the eval split. |
| `seed` | int | `0` | Base seed for boards and scripted opponents. |
| `max_turns` | int | `50` | Cap on policy turns per rollout (a full game usually takes 20–40). |
| max_consecutive_says | int | 2 | Cap on consecutive non-passing/deliberating actions (e.g. "say") to prevent stalling. |
| `endpoint_seat_memory` | str | `"digest"` | How endpoint seats see the game: `"digest"` — stateless per call, a compact clue/reveal record rebuilt from the event log (the guesser role is nearly Markovian; ~1/20th the tokens, and cache keys become path-independent — measured hit rate 44% vs 24%); `"conversation"` — full growing per-seat history, for stateful partners (e.g. an endpoint cluegiver that should remember its intent). |
| `cache_endpoint_seats` | bool | `true` | Memoize endpoint-seat responses on (endpoint, exact seat history) with in-flight dedup. Seat conversations are deterministic in the transcript prefix, so rollouts of the same board share adversary turns until the policy diverges — GRPO groups pay for one opening, not eight. Never applied to `"policy"` seats (their weights change every step). Cache hits are counted in `game_result["endpoint_cache_hits"]`. |

### Metrics

Two rules, each learned from a real reward hack observed in training:
**only outcomes the policy can influence pay** (rewarding raw wins taught a
policy to forfeit every turn and let a weak opponent self-destruct — 100%
wins, zero legal moves), and **protocol quality is a penalty, never income**
(flat compliance bonuses taught a self-play policy to filibuster with legal
`say` actions forever). A consecutive-`say` quota (`max_consecutive_says`)
additionally caps stalling structurally.

| Metric | Weight | Meaning |
| ------ | ------ | ------- |
| `reward` | — | Weighted sum of the criteria below |
| `win_earned` | 1.0 | 1 if red won by revealing its own last word |
| `earned_progress` | 0.3 | Fraction of red's words revealed by red's own play |
| `illegal_action_penalty` | 0.2 | 0 if all actions legal, down to −1 |
| `format_penalty` | 0.1 | 0 if all turns parsed, down to −1 |
| `action_legality` | 0.0 | Share of submitted actions ruled legal by the arbiter |
| `format_compliance` | 0.0 | Share of turns with a parseable JSON action |
| `team_win` | 0.0 | 1 if the red team won (any reason, incl. opponent mistakes) |
| `word_progress` | 0.0 | Fraction of red's words revealed by anyone |
| `assassin_loss` | 0.0 | 1 if red lost by revealing the assassin |
| `game_finished` | 0.0 | 1 if the game reached a natural end within `max_turns` |
| `turns_played` | 0.0 | Game turns (team alternations) played |
| `other_seat_fallbacks` | 0.0 | LLM-driven non-policy turns that fell back to the scripted baseline |
