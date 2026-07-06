# Train on it

`environments/gyrell/` packages the engine as a
[verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment.
The policy plays **whole teams**, each seat with its own private
conversation, against configurable opposition. No LLM judge anywhere:
the reward comes from `spec.score`.

```python
import gyrell

env = gyrell.load_environment(
    role="team",            # play every red seat
    opponent="scripted",    # or "selfplay", or any OpenAI-compatible endpoint
    partner="scripted",
)
```

```bash
cd environments/gyrell && uv pip install -e .
vf-eval gyrell -m openai/gpt-4.1-mini -n 5
```

To train on a new game, follow [Build a game](build-a-game.md), then mirror
the environment package: vendor the engine, expose a `load_environment`
that seats your policy, and let `spec.score` be the reward. Per-seat
observations map directly onto per-seat conversation threads.

Hub installs, training configs, and publishing:
**[Prime Intellect](../prime-intellect.md)**.
