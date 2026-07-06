# Seat the players

An actor is anything with `act(actor_input) -> ActorOutput`: private
observation and allowed actions in, action plus spoken words out. Every
seat at a table can be a different kind of actor.

**Scripted baselines** (`BaselineCodewordsActor`, `BaselineTabooActor`) are
deterministic and seeded. They serve as opponents/partners in training and
as the **fallback** behind every LLM seat — a match never stalls on a
misbehaving model.

**`OpenRouterActor`** drives a seat from [OpenRouter](https://openrouter.ai)
or any OpenAI-compatible endpoint; game-specific prompting lives in a
pluggable prompter. It handles response sanitization (unrepairable output
falls back to the baseline, with the raw response kept in the trace
sidecar), tight [reasoning budgets](https://openrouter.ai/docs/use-cases/reasoning-tokens)
with per-model overrides and token accounting, and short model-authored
notes-to-self between turns.

```python
from turngames.actors import OpenRouterActor, BaselineCodewordsActor, CodewordsPrompter

actor = OpenRouterActor(
    model="anthropic/claude-haiku-4.5",
    api_key=os.environ["OPENROUTER_API_KEY"],
    fallback=BaselineCodewordsActor(seed="match-1:red_guesser_1"),
    prompter=CodewordsPrompter(),
)
```

Point `base_url` at any local OpenAI-compatible shim to seat
subscription-backed models. A seat is just `act()` — humans fit the same
interface.
