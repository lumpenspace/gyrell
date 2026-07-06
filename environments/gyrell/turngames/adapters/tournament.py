"""Round-robin tournament runner: head-to-head LLM evals on any registered game.

The live channel answers "what does the show look like tonight"; this module
answers "which model is actually better, and with whom". It plays a systematic
schedule of matches — no pacing, no host, as fast as the APIs respond — and
archives each one in the standard replay format, so the existing leaderboard
(`server.leaderboard.aggregate`) and the LUDICA client consume tournament
matches exactly like live ones.

Schemes:

- ``uniform`` — every unordered pair of models meets twice (colors swapped),
  each team seated entirely by one model. Measures raw head-to-head strength.
- ``mixed`` — for every unordered pair {A, B}, one team seats A as cluegiver
  with B guessing while the other seats B cluing with A guessing (again both
  colors). Same two models on both sides — what differs is who leads, so the
  scheme isolates role fit and cross-model cooperation.

Use ``run_tournament`` from code or ``scripts/tournament.py`` from a shell.
"""

from __future__ import annotations

import gzip
import itertools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from turngames.actors import OpenRouterActor
from turngames.actors.llm import REASONING_OVERRIDES
from turngames.core import ActorInput, StateMachine
from turngames.games.codewords import CodewordsConfig
from turngames.registry import GAMES

SNAPSHOT_VIEWERS = ("spoiler_safe", "audience_omniscient")
# Hard ceiling on acts per match — a runaway match is a bug, not a game.
MAX_ACTS = 400


def transcript_line(event, name_of, game_id: str) -> str | None:
    """Human-readable line for an audience-visible event (the same voice the
    live channel feeds back to seated models between their turns)."""
    p = event.payload
    t = event.type
    if t == "clue_given":
        clue = p.get("clue", {}) or {}
        return f"{clue.get('team_id', '?')} clue: {str(clue.get('word', '')).upper()} for {clue.get('count', '?')}"
    if t == "actor_responded":
        said = p.get("public_deliberation")
        return f"{name_of(p.get('seat_id'))}: {said}" if said else None
    if t == "guess_proposed":
        return f"{name_of(p.get('seat_id'))} proposes {str(p.get('word', '')).upper()}"
    if t == "guess_confirmed":
        return f"{name_of(p.get('seat_id'))} confirms {str(p.get('word', '')).upper()}"
    if t == "guess_rejected":
        return f"{name_of(p.get('seat_id'))} vetoes {str(p.get('word', '')).upper()}"
    if t == "word_revealed":
        return f"{str(p.get('word', '')).upper()} revealed -> {p.get('assignment', '?')}"
    if t == "guesser_stopped":
        return f"{p.get('team_id', '?')} stops guessing"
    if t == "bonus_guess_reached":
        return f"{p.get('team_id', '?')} earns a bonus guess"
    if t == "bonus_clue_granted":
        return f"{p.get('team_id', '?')} earns a bonus clue"
    if t == "timer_expired":
        return "TIME! The round is over" if game_id == "taboo" else "time expired — one final guess"
    if t == "illegal_action_applied":
        return f"ILLEGAL move: {p.get('message', p.get('rule_id', ''))}"
    if t == "card_guessed":
        card = p.get("card", {}) or {}
        verb = "SAID" if p.get("via") == "spoken" else "got it:"
        return f"{name_of(p.get('seat_id'))} {verb} {str(card.get('target', '')).upper()} — point {p.get('team_id', '?')}"
    if t == "guess_incorrect":
        return f"{name_of(p.get('seat_id'))} guesses {str(p.get('word', '')).upper()} — no"
    if t == "guesser_passed":
        return f"{name_of(p.get('seat_id'))} passes the floor back"
    if t == "card_skipped":
        card = p.get("card", {}) or {}
        return f"{name_of(p.get('seat_id'))} skips {str(card.get('target', '')).upper()}"
    if t == "taboo_violation":
        card = p.get("card", {}) or {}
        return f"BUZZ! {name_of(p.get('seat_id'))} said '{p.get('said', '?')}' — {str(card.get('target', '')).upper()} lost"
    if t == "round_ended":
        points = p.get("points", {}) or {}
        return f"round {p.get('round_number', '?')} over — red {points.get('red', 0)} · blue {points.get('blue', 0)}"
    return None


@dataclass(frozen=True)
class Fixture:
    """One scheduled match: a seat->model lineup plus its scheme tag."""

    seed: str
    scheme: str  # "uniform" | "mixed"
    lineup: dict[str, str]  # seat_id -> model slug


@dataclass
class MatchResult:
    fixture: Fixture
    winner: str | None
    score: dict
    acts: int
    fallback_turns: dict[str, int] = field(default_factory=dict)
    error: str | None = None


def _codewords_seats(guessers: int) -> dict[str, list[str]]:
    return {
        team: [f"{team}_cluegiver"] + [f"{team}_guesser_{i + 1}" for i in range(guessers)]
        for team in ("red", "blue")
    }


def make_fixtures(
    models: list[str],
    schemes: tuple[str, ...] = ("uniform", "mixed"),
    guessers: int = 1,
    rounds: int = 1,
    stamp: str | None = None,
) -> list[Fixture]:
    """The schedule. Every pairing is played in both color assignments, since
    the first-moving team has an edge worth cancelling out."""
    seats = _codewords_seats(guessers)
    fixtures: list[Fixture] = []
    stamp = stamp or str(int(time.time()))

    def add(scheme: str, red_by_seat: dict[str, str], blue_by_seat: dict[str, str]) -> None:
        n = len(fixtures) + 1
        fixtures.append(
            Fixture(seed=f"trn-{stamp}-{n}", scheme=scheme, lineup={**red_by_seat, **blue_by_seat})
        )

    def team(team_id: str, cluegiver: str, guesser: str) -> dict[str, str]:
        ids = seats[team_id]
        return {ids[0]: cluegiver, **{sid: guesser for sid in ids[1:]}}

    for _ in range(rounds):
        for a, b in itertools.combinations(models, 2):
            if "uniform" in schemes:
                add("uniform", team("red", a, a), team("blue", b, b))
                add("uniform", team("red", b, b), team("blue", a, a))
            if "mixed" in schemes:
                # A leads B against B leading A — role fit, colors swapped.
                add("mixed", team("red", a, b), team("blue", b, a))
                add("mixed", team("red", b, a), team("blue", a, b))
    return fixtures


def play_match(
    fixture: Fixture,
    api_key: str,
    game_id: str = "codewords",
    guessers: int = 1,
    out_dir: Path | None = None,
    endpoints: dict[str, dict] | None = None,
) -> MatchResult:
    """One synchronous match, archived in the live channel's replay format."""
    entry = GAMES[game_id]
    spec = entry.spec_factory()
    if game_id == "codewords":
        config = CodewordsConfig(
            guessers_per_team=guessers,
            turn_time_limit_words=120,
            seat_labels=dict(fixture.lineup),
        )
    else:
        config = entry.config_factory(dict(fixture.lineup))
    machine = StateMachine.new(spec, fixture.seed, config)
    role_specs = spec.role_specs()

    # Three ways to seat a label: an `endpoints` entry (any OpenAI-compatible
    # base_url — local shims, self-hosted, whatever), a provider-prefixed
    # OpenRouter slug, or anything else = the scripted baseline (dry runs).
    endpoints = endpoints or {}

    def seat_actor(seat_id: str, slug: str):
        if slug in endpoints:
            spec = endpoints[slug]
            return OpenRouterActor(
                model=str(spec.get("model", slug)),
                api_key=str(spec.get("api_key", "local")),
                base_url=str(spec["base_url"]),
                timeout=float(spec.get("timeout", 120.0)),
                fallback=entry.baseline_factory(f"{fixture.seed}:{slug}"),
                prompter=entry.prompter_factory(),
                reasoning_override=spec.get("reasoning_override"),
            )
        if "/" in slug:
            return OpenRouterActor(
                model=slug,
                api_key=api_key,
                fallback=entry.baseline_factory(f"{fixture.seed}:{slug}"),
                prompter=entry.prompter_factory(),
                reasoning_override=REASONING_OVERRIDES.get(slug),
            )
        return entry.baseline_factory(f"{fixture.seed}:{seat_id}:{slug}")

    actors = {seat_id: seat_actor(seat_id, slug) for seat_id, slug in fixture.lineup.items()}

    def name_of(seat_id: str | None) -> str:
        return str(fixture.lineup.get(seat_id, seat_id)) if seat_id else "someone"

    transcript: list[str] = []
    illegal_streak: dict[str, int] = {}
    fallback_turns: dict[str, int] = {}
    turns: dict[str, int] = {}
    traces: list[dict] = []
    archive: list[dict] = [
        {"kind": "meta", "game_id": spec.id, "seed": fixture.seed, "live": False,
         "tournament": fixture.scheme}
    ]

    def snapshots() -> list[dict]:
        return [
            {"kind": "snapshot", "viewer": viewer, "view": spec.observe_viewer(machine.state, viewer)}
            for viewer in SNAPSHOT_VIEWERS
        ]

    archive.extend(snapshots())

    acts = 0
    error: str | None = None
    try:
        while not spec.is_terminal(machine.state) and acts < MAX_ACTS:
            seat_id = spec.acting_seat(machine.state)
            if seat_id is None:
                break
            acts += 1
            seat = machine.state.seat(seat_id)
            observation = machine.observe(seat_id)
            constraints = {}
            if observation.data.get("final_guess_only"):
                constraints["final_guess_one_word"] = True
            if observation.data.get("bonus_guess"):
                constraints["bonus_guess_two_words"] = True
            actor = actors[seat_id]
            forced_fallback = illegal_streak.get(seat_id, 0) >= 2 and hasattr(actor, "fallback")
            if forced_fallback:
                actor = actor.fallback
            actor_input = ActorInput(
                seat=seat,
                role=role_specs[seat.role],
                observation=observation,
                time_left=observation.data.get("time_left"),
                constraints=constraints,
                transcript=tuple(transcript[-40:]),
            )
            output = actor.act(actor_input)
            event_start = len(machine.log.events)
            ruling, _ = machine.submit(
                seat_id, output.action, public_deliberation=output.public_deliberation
            )
            seated = actors[seat_id]
            fell_back = forced_fallback or getattr(seated, "last_error", None) is not None
            turns[seat_id] = turns.get(seat_id, 0) + 1
            if fell_back:
                fallback_turns[seat_id] = fallback_turns.get(seat_id, 0) + 1
            traces.append(
                {
                    "seq": event_start,
                    "seat_id": seat_id,
                    "model": fixture.lineup.get(seat_id),
                    "action_type": output.action.type,
                    "public_deliberation": output.public_deliberation,
                    "reasoning": output.reasoning,
                    "reasoning_tokens": getattr(seated, "last_reasoning_tokens", None),
                    "raw_response": output.raw_response,
                    "fallback": fell_back,
                    "error": (
                        "forced_baseline_after_illegal_streak"
                        if forced_fallback
                        else getattr(seated, "last_error", None)
                    ),
                }
            )
            illegal_streak[seat_id] = 0 if ruling.is_legal else illegal_streak.get(seat_id, 0) + 1
            for event in machine.log.events[event_start:]:
                if "audience" not in event.visibility:
                    continue
                archive.append(
                    {
                        "kind": "event",
                        "event": {
                            "seq": event.seq,
                            "type": event.type,
                            "payload": event.payload,
                            "visibility": list(event.visibility),
                        },
                    }
                )
                line = transcript_line(event, name_of, game_id)
                if line:
                    transcript.append(line)
            archive.extend(snapshots())
    except Exception as exc:  # keep the bracket alive; one match may die
        error = repr(exc)

    score = machine.score()
    archive.append(
        {"kind": "final_score", "score": score, "fallback_turns": fallback_turns, "turns": turns}
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(out_dir / f"{fixture.seed}.jsonl.gz", "wt") as fh:
            for record in archive:
                fh.write(json.dumps(record) + "\n")
        trace_dir = out_dir / "traces"
        trace_dir.mkdir(exist_ok=True)
        with gzip.open(trace_dir / f"{fixture.seed}.jsonl.gz", "wt") as fh:
            for trace in traces:
                fh.write(json.dumps(trace) + "\n")

    return MatchResult(
        fixture=fixture,
        winner=score.get("winner"),
        score=score,
        acts=acts,
        fallback_turns=fallback_turns,
        error=error,
    )


def standings(results: list[MatchResult]) -> dict:
    """Per-model records across every seat they held, plus scheme splits."""
    table: dict[str, dict] = {}

    def bucket(model: str) -> dict:
        return table.setdefault(
            model,
            {"matches": 0, "wins": 0, "losses": 0, "draws": 0,
             "as_cluegiver_wins": 0, "as_cluegiver_matches": 0},
        )

    for r in results:
        if r.error:
            continue
        teams: dict[str, set[str]] = {"red": set(), "blue": set()}
        cluegivers: dict[str, str] = {}
        for seat_id, slug in r.fixture.lineup.items():
            team = seat_id.split("_", 1)[0]
            teams[team].add(slug)
            if "cluegiver" in seat_id or seat_id.endswith("_player_1"):
                cluegivers[team] = slug
        for team, members in teams.items():
            for slug in members:
                b = bucket(slug)
                b["matches"] += 1
                if r.winner is None:
                    b["draws"] += 1
                elif r.winner == team:
                    b["wins"] += 1
                else:
                    b["losses"] += 1
        for team, slug in cluegivers.items():
            b = bucket(slug)
            b["as_cluegiver_matches"] += 1
            if r.winner == team:
                b["as_cluegiver_wins"] += 1

    for model, b in table.items():
        b["win_rate"] = round(b["wins"] / b["matches"], 3) if b["matches"] else 0.0
    return dict(sorted(table.items(), key=lambda kv: -kv[1]["win_rate"]))


def run_tournament(
    models: list[str],
    api_key: str,
    game_id: str = "codewords",
    schemes: tuple[str, ...] = ("uniform", "mixed"),
    guessers: int = 1,
    rounds: int = 1,
    workers: int = 6,
    out_dir: Path | str = "replays",
    endpoints: dict[str, dict] | None = None,
    progress=print,
) -> tuple[list[MatchResult], dict]:
    out_dir = Path(out_dir)
    fixtures = make_fixtures(models, schemes=schemes, guessers=guessers, rounds=rounds)
    progress(f"{len(fixtures)} matches scheduled ({', '.join(schemes)}; {len(models)} models)")
    results: list[MatchResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(play_match, f, api_key, game_id, guessers, out_dir, endpoints): f
            for f in fixtures
        }
        for done in as_completed(futures):
            r = done.result()
            results.append(r)
            tag = r.winner or ("ERROR " + (r.error or "")[:40] if r.error else "draw")
            red = {s for i, s in r.fixture.lineup.items() if i.startswith("red")}
            blue = {s for i, s in r.fixture.lineup.items() if i.startswith("blue")}
            progress(
                f"[{len(results)}/{len(fixtures)}] {r.fixture.seed} ({r.fixture.scheme}) "
                f"{'+'.join(sorted(red))} vs {'+'.join(sorted(blue))} -> {tag} ({r.acts} acts)"
            )
    results.sort(key=lambda r: r.fixture.seed)
    final = standings(results)
    summary = {
        "models": models,
        "schemes": list(schemes),
        "guessers_per_team": guessers,
        "matches": [
            {
                "seed": r.fixture.seed,
                "scheme": r.fixture.scheme,
                "lineup": r.fixture.lineup,
                "winner": r.winner,
                "acts": r.acts,
                "fallback_turns": r.fallback_turns,
                "error": r.error,
            }
            for r in results
        ],
        "standings": final,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tournament-results.json").write_text(json.dumps(summary, indent=2))
    return results, final


# -- Reporter: color commentary, never scoring -------------------------------
#
# An optional model reads each finished match's public transcript and flags
# the moments worth a human's attention — brilliant clues, disasters,
# miscoordination, near-misses. Strictly qualitative: reporter output never
# touches records, Elo, or standings, so the eval stays judge-free.

REPORTER_SYSTEM = (
    "You are a sharp, economical match reporter for a word-game broadcast "
    "between language models. Given a match transcript, flag ONLY genuinely "
    "interesting moments: brilliant or catastrophic clues, miscoordination "
    "between teammates, near-misses (e.g. almost picking the assassin), "
    "clever recoveries, or telling misunderstandings. 0 to 3 highlights; an "
    "uneventful match gets none. Respond with ONLY a JSON object: "
    '{"highlights": [{"tag": "<brilliant|blunder|miscoordination|near_miss|recovery|other>", '
    '"quote": "<the transcript line(s), verbatim>", '
    '"note": "<one dry sentence on why it matters>"}]}'
)


def _chat_once(model: str, api_key: str, system: str, user: str, timeout: float = 60.0) -> str:
    import urllib.request

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1200,
        }
    ).encode()
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload["choices"][0]["message"].get("content") or ""


def transcript_from_replay(records: list[dict], game_id: str) -> tuple[list[str], dict]:
    """Rebuild the public transcript (and seat->model lineup) from an archive."""
    from types import SimpleNamespace

    lineup: dict[str, str] = {}
    for record in records:
        if record.get("kind") == "snapshot":
            for seat in record.get("view", {}).get("seats", []):
                if seat.get("label"):
                    lineup[seat["id"]] = seat["label"]
            break

    def name_of(seat_id):
        return str(lineup.get(seat_id, seat_id)) if seat_id else "someone"

    lines: list[str] = []
    for record in records:
        if record.get("kind") != "event":
            continue
        e = record["event"]
        line = transcript_line(
            SimpleNamespace(type=e["type"], payload=e["payload"]), name_of, game_id
        )
        if line:
            lines.append(line)
    return lines, lineup


def report_match(
    transcript: list[str],
    lineup: dict[str, str],
    winner: str | None,
    reporter_model: str,
    api_key: str,
) -> dict | None:
    """One reporter pass; None when the reporter errors or sees nothing."""
    from turngames.actors.llm import extract_json

    teams: dict[str, list[str]] = {}
    for seat_id, slug in sorted(lineup.items()):
        teams.setdefault(seat_id.split("_", 1)[0], []).append(f"{seat_id}={slug}")
    user = (
        "Lineup: " + "; ".join(f"{t}: {', '.join(m)}" for t, m in teams.items())
        + f"\nWinner: {winner or 'draw'}\n\nTranscript:\n" + "\n".join(transcript)
    )
    try:
        raw = _chat_once(reporter_model, api_key, REPORTER_SYSTEM, user)
        parsed = extract_json(raw)
    except Exception:
        return None
    if not parsed or not parsed.get("highlights"):
        return None
    return {"highlights": parsed["highlights"]}


def report_tournament(
    out_dir: Path | str, reporter_model: str, api_key: str, progress=print
) -> Path:
    """Report every archived match in a tournament directory, then write
    reports/<seed>.json per match and a digest at tournament-report.md."""
    out_dir = Path(out_dir)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    digest: list[str] = ["# Tournament report", ""]
    for replay in sorted(out_dir.glob("*.jsonl.gz")):
        records = [json.loads(line) for line in gzip.open(replay, "rt")]
        meta = records[0] if records and records[0].get("kind") == "meta" else {}
        final = next((r for r in records if r.get("kind") == "final_score"), {})
        transcript, lineup = transcript_from_replay(records, str(meta.get("game_id", "codewords")))
        report = report_match(
            transcript, lineup, final.get("score", {}).get("winner"), reporter_model, api_key
        )
        seed = str(meta.get("seed", replay.stem))
        if report:
            (report_dir / f"{seed}.json").write_text(json.dumps(report, indent=2))
            reds = sorted({s.split("/")[-1] for i, s in lineup.items() if i.startswith("red")})
            blues = sorted({s.split("/")[-1] for i, s in lineup.items() if i.startswith("blue")})
            digest.append(f"## {seed} — {'+'.join(reds)} vs {'+'.join(blues)}")
            for h in report["highlights"]:
                digest.append(f"- **{h.get('tag', 'moment')}**: “{h.get('quote', '')}” — {h.get('note', '')}")
            digest.append("")
            progress(f"{seed}: {len(report['highlights'])} highlight(s)")
        else:
            progress(f"{seed}: nothing notable")
    path = out_dir / "tournament-report.md"
    path.write_text("\n".join(digest))
    return path
