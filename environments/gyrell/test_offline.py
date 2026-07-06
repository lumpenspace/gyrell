"""Offline harness test: drive CodewordsEnv rollouts with a scripted policy.

This does NOT test model quality — it verifies the environment plumbing
(turn-taking, per-seat hidden information, termination, rewards, endpoint
fallback) without any API calls, by having a scripted actor stand in for the
trained policy. Real rollouts (vf-eval / prime-rl) always have the actual
model playing full games.
"""

import asyncio
import json
import os

from codewords import (
    SYSTEM_PROMPT,
    load_environment,
    team_win,
    win_earned,
    word_progress,
    earned_progress,
    action_legality,
    format_compliance,
    illegal_action_penalty,
    format_penalty,
    assassin_loss,
    game_finished,
    turns_played,
    other_seat_fallbacks,
)
from turngames.actors.baseline import BaselineCodewordsActor
from turngames.core import ActorInput

REWARD_FUNCS = [
    team_win,
    win_earned,
    word_progress,
    earned_progress,
    action_legality,
    format_compliance,
    illegal_action_penalty,
    format_penalty,
    assassin_loss,
    game_finished,
    turns_played,
    other_seat_fallbacks,
]


def scripted_policy_reply(env, state, garbage=False):
    if garbage:
        return "I think we should definitely guess something!!"
    machine = state["machine"]
    seat_id = state["acting_seat"]
    seat = machine.state.seat(seat_id)
    actor = state.setdefault("_policy_actors", {}).setdefault(
        seat_id, BaselineCodewordsActor(seed=f"policy:{seat_id}")
    )
    out = actor.act(
        ActorInput(
            seat=seat,
            role=env.role_specs[seat.role],
            observation=env.spec.observe(machine.state, seat_id),
            constraints={},
        )
    )
    return json.dumps(
        {
            "public_deliberation": out.public_deliberation,
            "action": {"type": out.action.type, **out.action.payload},
        }
    )


async def run_rollout(env, row, inject_garbage_turn=None, max_steps=200):
    state = {"info": row["info"], "trajectory": []}
    await env.setup_state(state)
    steps = 0
    while steps < max_steps and state.get("final_env_response") is None:
        msgs = await env.get_prompt_messages(state)
        if state.get("final_env_response") is not None:
            break
        garbage = inject_garbage_turn is not None and steps == inject_garbage_turn
        reply = scripted_policy_reply(env, state, garbage=garbage)
        state["trajectory"].append(
            {
                "prompt": msgs,
                "completion": [{"role": "assistant", "content": reply}],
            }
        )
        steps += 1
    env._snapshot_result(state)
    rewards = {}
    for func in REWARD_FUNCS:
        rewards[func.__name__] = await func(state=state)
    return state, steps, rewards


class _StubCompletions:
    """OpenAI-shaped stub that always returns an unusable reply, so every
    LLM-driven seat exercises the baseline-fallback path. Counts calls so
    the endpoint-cache test can assert deduplication."""

    calls = 0

    class _Message:
        content = "sorry, I refuse to answer in JSON today"

    class _Choice:
        message = None

    async def create(self, **kwargs):
        _StubCompletions.calls += 1
        choice = self._Choice()
        choice.message = self._Message()
        result = type("R", (), {})()
        result.choices = [choice]
        return result


class _StubClient:
    def __init__(self):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _StubCompletions()


async def main():
    for role, gpt in [("team", 1), ("team", 2), ("cluegiver", 1), ("guesser", 1)]:
        env = load_environment(
            role=role,
            guessers_per_team=gpt,
            num_train_examples=3,
            num_eval_examples=1,
            max_turns=100,
        )
        ds = env.get_dataset()
        assert len(ds) == 3
        for i, row in enumerate(ds):
            state, steps, rewards = await run_rollout(
                env, row, inject_garbage_turn=2 if i == 0 else None
            )
            result = state["game_result"]
            print(
                f"role={role} gpt={gpt} ex={i} steps={steps:3d} "
                f"winner={result['winner']} reason={result['terminal_reason']} "
                f"fmt_err={result['format_errors']} illegal={result['illegal_actions']} "
                f"rewards={{{', '.join(f'{k}={v:.2f}' for k, v in rewards.items())}}}"
            )
            assert result["winner"] is not None or steps == 200, "game should finish"
            if i == 0:
                assert result["format_errors"] >= 1, "garbage turn should count"

    # determinism: same info must produce the same initial prompt
    env1 = load_environment(num_train_examples=2, num_eval_examples=0)
    env2 = load_environment(num_train_examples=2, num_eval_examples=0)
    row = env1.get_dataset()[0]
    s1 = {"info": row["info"], "trajectory": []}
    s2 = {"info": row["info"], "trajectory": []}
    await env1.setup_state(s1)
    await env2.setup_state(s2)
    assert s1["prompt"] == s2["prompt"]
    print("determinism ok")

    # hidden info: guesser seat history must never contain the key
    env = load_environment(role="team", num_train_examples=1, num_eval_examples=0)
    row = env.get_dataset()[0]
    state, _, _ = await run_rollout(env, row)
    for seat_id, history in state["seat_histories"].items():
        text = json.dumps(history)
        if "guesser" in seat_id:
            assert ", key=" not in text, f"key leaked into {seat_id} history"
        else:
            assert ", key=" in text, "cluegiver should see the key"
    print("hidden-information check ok")

    # endpoint-driven opponent with a stub client: games must still complete
    # via the baseline fallback, and fallbacks must be counted
    os.environ.setdefault("CODEWORDS_TEST_KEY", "stub")
    env = load_environment(
        role="team",
        opponent={"model": "stub-model", "api_key_var": "CODEWORDS_TEST_KEY"},
        num_train_examples=1,
        num_eval_examples=0,
        max_turns=100,
    )
    spec = next(iter(env._endpoint_clients))
    env._endpoint_clients[spec] = _StubClient()
    row = env.get_dataset()[0]
    state, steps, rewards = await run_rollout(env, row)
    result = state["game_result"]
    assert result["other_seat_fallbacks"] > 0, "stub replies should trigger fallback"
    assert result["winner"] is not None, "game should still finish"
    # opponent guessers are conversational now and must not see any key
    for seat_id, history in state["seat_histories"].items():
        if "blue_guesser" in seat_id:
            assert ", key=" not in json.dumps(history), f"key leaked to {seat_id}"
    print(
        f"endpoint fallback ok (fallbacks={result['other_seat_fallbacks']}, "
        f"winner={result['winner']})"
    )

    # endpoint cache: rerunning the same seeded board with the same scripted
    # policy is an identical game, so the second rollout must be served
    # entirely from the response cache (zero new endpoint calls)
    calls_after_first = _StubCompletions.calls
    state2, _, _ = await run_rollout(env, row)
    assert _StubCompletions.calls == calls_after_first, (
        f"replayed rollout made {_StubCompletions.calls - calls_after_first} "
        "uncached endpoint calls"
    )
    assert state2["counters"]["endpoint_cache_hits"] > 0
    assert state2["game_result"]["winner"] == result["winner"], "replay must match"
    print(
        f"endpoint cache ok (hits={state2['counters']['endpoint_cache_hits']}, "
        f"0 new calls on replay)"
    )

    # driver pools with role-split combos: matchup must be chosen per example
    # (deterministic in the board seed) and recorded in game_result
    env = load_environment(
        role="team",
        opponent=[
            "scripted",
            {"model": "stub-a", "api_key_var": "CODEWORDS_TEST_KEY"},
            {
                "cluegiver": {"model": "stub-b", "api_key_var": "CODEWORDS_TEST_KEY"},
                "guesser": "scripted",
            },
        ],
        num_train_examples=8,
        num_eval_examples=0,
        max_turns=100,
    )
    for spec in list(env._endpoint_clients):
        env._endpoint_clients[spec] = _StubClient()
    ds = env.get_dataset()
    matchups = {}
    for row in ds:
        state, _, _ = await run_rollout(env, row)
        labels = state["game_result"]["seat_models"]
        matchups[row["info"]["game_seed"]] = json.dumps(labels, sort_keys=True)
        assert set(labels) == {"blue_cluegiver", "blue_guesser_1"}
    assert len(set(matchups.values())) > 1, "pool should yield varied matchups"
    # re-running the same seeds must reproduce the same matchups
    state, _, _ = await run_rollout(env, ds[0])
    assert (
        json.dumps(state["game_result"]["seat_models"], sort_keys=True)
        == matchups[ds[0]["info"]["game_seed"]]
    )
    print(f"driver pools ok ({len(set(matchups.values()))} distinct matchups over 8 games)")

    # say quota: after max_consecutive_says, 'say' leaves the allowed set and
    # a submitted say is rejected as a format error (anti-filibuster guard)
    env = load_environment(
        role="team", num_train_examples=1, num_eval_examples=0, max_consecutive_says=2
    )
    row = env.get_dataset()[0]
    state = {"info": row["info"], "trajectory": []}
    await env.setup_state(state)
    seat = state["acting_seat"]
    say = json.dumps({"public_deliberation": "hmm", "action": {"type": "say"}})
    before = state["counters"]["format_errors"]
    for _ in range(4):
        msgs = await env.get_prompt_messages(state)
        state["trajectory"].append(
            {"prompt": msgs, "completion": [{"role": "assistant", "content": say}]}
        )
    await env.get_prompt_messages(state)  # process 4th say
    allowed = {a["type"] for a in env._allowed_actions(state, seat)}
    assert "say" not in allowed, "say should be quota-limited"
    assert state["counters"]["format_errors"] > before, "over-quota say must be rejected"
    print("say quota ok")

    # process opponent: zero-token stochastic blue team — games complete,
    # blue reveals 1-3 own cards + terminal card, assassin never touched by blue
    env = load_environment(
        role="team",
        opponent="process",
        num_train_examples=4,
        num_eval_examples=0,
        max_turns=100,
    )
    for row in env.get_dataset():
        state, steps, rewards = await run_rollout(env, row)
        result = state["game_result"]
        assert result["winner"] is not None, "process-opponent game should finish"
        assert result["endpoint_calls"] == 0 and result["other_seat_fallbacks"] == 0
        assert result["seat_models"]["blue_cluegiver"] == "process(1-3)"
        # blue must never reveal the assassin (excluded from terminal pool)
        for event in state["machine"].log.events:
            if event.type == "word_revealed" and event.payload.get("team_id") == "blue":
                assert event.payload.get("assignment") != "assassin", (
                    "process opponent must never touch the assassin"
                )
    print("process opponent ok (4 games, 0 tokens, no blue assassin)")

    # partner dossiers: deterministic real/false assignment, injected into
    # policy prompts only, condition recorded in game_result
    env = load_environment(
        role="cluegiver",
        partner={"model": "stub-a", "api_key_var": "CODEWORDS_TEST_KEY"},
        opponent="process",
        num_train_examples=12,
        num_eval_examples=0,
        max_turns=100,
        partner_dossiers={
            "stub-a": "DOSSIER-REAL: chases pop-culture references.",
            "stub-b": "DOSSIER-DECOY: extremely literal decoder.",
        },
        dossier_rate=0.7,
        false_dossier_rate=0.5,
    )
    for spec in list(env._endpoint_clients):
        env._endpoint_clients[spec] = _StubClient()
    kinds = {"real": 0, "false": 0, None: 0}
    for row in env.get_dataset():
        state, _, _ = await run_rollout(env, row)
        kind = state["game_result"]["dossier"]
        kinds[kind] += 1
        policy_hist = json.dumps(state["seat_histories"][env.policy_seats[0]])
        if kind == "real":
            assert "DOSSIER-REAL" in policy_hist
        elif kind == "false":
            assert "DOSSIER-DECOY" in policy_hist
        else:
            assert "DOSSIER" not in policy_hist
    # same seed -> same condition (determinism, shared across a GRPO group)
    first_info = env.get_dataset()[0]["info"]
    s_a = {"info": first_info, "trajectory": []}
    s_b = {"info": first_info, "trajectory": []}
    await env.setup_state(s_a)
    await env.setup_state(s_b)
    assert (s_a.get("dossier") or {}).get("kind") == (s_b.get("dossier") or {}).get("kind")
    assert kinds["real"] > 0 and kinds["false"] > 0 and kinds[None] > 0, kinds
    print(f"partner dossiers ok (mix over 12 games: {kinds})")

    # misconfigured endpoint must fail fast
    try:
        load_environment(
            opponent={"model": "x", "api_key_var": "CODEWORDS_MISSING_KEY_XYZ"},
            num_train_examples=1,
            num_eval_examples=0,
        )
        raise AssertionError("expected ValueError for missing api key")
    except ValueError:
        print("missing-key fail-fast ok")

    # phase-0 hooks: think-tag stripping, persona, label exposure, rendered
    # deliberation
    from turngames.actors.llm import extract_json

    assert extract_json(
        '<think>{"decoy": 1} braces inside</think>{"action": {"type": "stop"}}'
    ) == {"action": {"type": "stop"}}
    assert extract_json(
        'dangling reasoning...</think>\n{"action": {"type": "stop"}}'
    ) == {"action": {"type": "stop"}}
    env = load_environment(
        role="team",
        num_train_examples=1,
        num_eval_examples=0,
        max_turns=100,
        render_deliberation=True,
        expose_partner_labels=True,
        opponent={
            "model": "stub-styled",
            "api_key_var": "CODEWORDS_TEST_KEY",
            "persona": "You speak in pirate slang at all times.",
        },
    )
    spec = next(iter(env._endpoint_clients))
    env._endpoint_clients[spec] = _StubClient()
    row = env.get_dataset()[0]
    state, _, _ = await run_rollout(env, row)
    assert "pirate slang" in env._seat_system_prompt(state, "blue_cluegiver")
    assert "blue_cluegiver" not in state["seat_histories"], (
        "digest-mode endpoint seats must be stateless"
    )
    digest = env._game_digest(state["machine"])
    assert "clue" in digest and "->" in digest, "digest should record clue outcomes"
    assert ", key=" not in digest, "digest must not leak the key"
    assert " said \"" in digest, "render_deliberation should reach the digest"
    policy_hist = json.dumps(state["seat_histories"][env.policy_seats[0]])
    assert "is played by stub-styled" in policy_hist, "labels should be exposed"
    assert " said: " in policy_hist, "deliberation should be rendered"
    print("phase-0 hooks ok (think-strip, persona, labels, deliberation)")

    # cache_control annotation: auto-on for anthropic via openrouter only
    from codewords import EndpointSpec, _annotate_cache_control, _wants_cache_control

    assert _wants_cache_control(EndpointSpec(model="anthropic/claude-haiku-4.5"))
    assert not _wants_cache_control(EndpointSpec(model="openai/gpt-4.1-mini"))
    assert not _wants_cache_control(
        EndpointSpec(model="anthropic/claude-haiku-4.5", base_url="https://api.pinference.ai/api/v1")
    )
    assert _wants_cache_control(
        EndpointSpec(model="deepseek/deepseek-v4-flash", prompt_cache="on")
    )
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    annotated = _annotate_cache_control(msgs)
    assert annotated[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert annotated[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert annotated[1]["content"] == "u1" and annotated[2]["content"] == "a1"
    assert msgs[0]["content"] == "sys", "original history must not be mutated"
    print("cache_control annotation ok")

    print("smoke test passed")


asyncio.run(main())
