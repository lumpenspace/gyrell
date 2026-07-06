# Prime Intellect

The GyReLL environment is published on the
[Environments Hub](https://app.primeintellect.ai/dashboard/environments) as
[**`lumpenspace/gyrell`**](https://app.primeintellect.ai/dashboard/environments/lumpenspace/gyrell).

## Use it

```bash
# with the prime CLI (https://docs.primeintellect.ai/cli)
prime env install lumpenspace/gyrell

# evaluate any OpenAI-compatible model on it
uv run vf-eval gyrell -m openai/gpt-4.1-mini -n 5
```

## Train on it

Ready-made [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) configs
live in [`configs/`](https://github.com/lumpenspace/gyrell/tree/main/configs)
— `rl-smoke` (sanity run), `rl-selfplay`, `eval-checkpoint`, and
`dryrun-free` (free-tier models end to end):

```bash
prime train configs/rl-smoke.toml -e PRIME_API_KEY -y
```

## Publish changes

The environment package is [`environments/gyrell/`](https://github.com/lumpenspace/gyrell/tree/main/environments/gyrell);
it vendors a snapshot of `src/turngames` so the wheel is self-contained.

```bash
environments/gyrell/sync_engine.sh    # refresh the vendored engine
cd environments/gyrell
prime env push                        # bump version in pyproject.toml first
```

!!! note "Old hub id"
    The env previously shipped as `lumpenspace/codewords`; that id stays
    published so existing runs keep resolving. New work targets
    `lumpenspace/gyrell`.
