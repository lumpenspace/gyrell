"""Live broadcast channel: game -> intermission -> next game, forever.

One match is live at a time (spec.md's one-match constraint). Each match
runs one of the registered games (GAME_MODES rotates through them), deals a
fresh board/deck, and seats a rotating combination of roster personas
(baseline actors as fallback; LLM adapters behind the same Actor.act
interface). Finished matches are archived as replay JSONL so the existing
/ws/replay path can re-serve them.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from turngames.actors import OpenRouterActor
from turngames.actors.llm import REASONING_OVERRIDES
from turngames.core import ActorInput, StateMachine
from turngames.registry import GAMES

from server.host import LLMHost
from server.leaderboard import matches_played

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"
SNAPSHOT_VIEWERS = ("spoiler_safe", "audience_omniscient")

INTERMISSION_SECONDS = int(os.environ.get("INTERMISSION_SECONDS", "300"))
SECONDS_PER_DELIBERATION_WORD = 0.22
SECONDS_PER_EVENT = 0.9

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Which games the channel rotates through, one per match, in order. Override
# with GAME_MODES=codewords (say) to pin a single game.
GAME_MODES = tuple(
    mode.strip()
    for mode in os.environ.get("GAME_MODES", "codewords,taboo").split(",")
    if mode.strip() in GAMES
) or ("codewords",)

# Model roster (OpenRouter slugs) used when an API key is available; one seat
# per entry, rotated into fresh combinations each match. Override with
# OPENROUTER_MODELS=slug1,slug2,... Labels shown on stage (and later
# aggregated by the leaderboard) are the slug tails.
DEFAULT_MODEL_ROSTER = (
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4o-mini",
    "openai/gpt-5.4-nano",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-small-3.2-24b-instruct",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen-2.5-72b-instruct",
    "z-ai/glm-4.6",
    # Benched for now: laguna thinks 4-5k tokens per turn and ignores every
    # reasoning cap, so under show-pace budgets it mostly plays as its
    # fallback. Re-add (or use OPENROUTER_MODELS) when it behaves or when a
    # per-model budget override exists.
    # "poolside/laguna-xs-2.1",
)

# Scripted personas used when no API key is configured.
BASELINE_ROSTER = (
    "baseline-alpha",
    "baseline-bravo",
    "baseline-carol",
    "baseline-delta",
    "baseline-echo",
    "baseline-fox",
    "baseline-golf",
    "baseline-hotel",
)


def active_roster() -> tuple[str, ...]:
    if not OPENROUTER_API_KEY:
        return BASELINE_ROSTER
    env_models = os.environ.get("OPENROUTER_MODELS")
    if env_models:
        return tuple(slug.strip() for slug in env_models.split(",") if slug.strip())
    return DEFAULT_MODEL_ROSTER


ROSTER = active_roster()

# Per-slug reasoning overrides live with the actor so the tournament runner
# shares them: see turngames.actors.llm.REASONING_OVERRIDES.

# Canned host interjections: the fallback whenever the LLM host (server.host)
# is unavailable or errors. The host is not a game seat — see spec.md's
# broadcast roles; these never enter player context.
HOST_END_LINES = (
    "{winner} takes it! What a match.",
    "And that's the game — {winner} wins!",
    "It's all over: {winner} on top tonight.",
)

# Opening the show — the LLM host introduces the actual lineup; the canned
# fallback just welcomes the crowd.
HOST_START_LINES = (
    "Lights up — welcome to LUDICA! Let's meet tonight's models.",
    "We're live! Six models, two teams, one grid — here's the lineup.",
    "Showtime at LUDICA. Please welcome tonight's competitors.",
)

# Top of a round — recap and hand the floor to the team on the clock.
HOST_ROUND_LINES = (
    "Top of the round — {team}, the board's waiting.",
    "New round: {team} steps to the table.",
    "Here we go again — {team} on the clock.",
)

HOST_FOUL_LINES = (
    "Flag on the play — that one's ruled illegal.",
    "Ooh, a foul! The refs weren't having it.",
    "Illegal move — that gets waved off.",
)

def game_for_match(match_number: int) -> str:
    """Which registered game a given match plays: GAME_MODES, round-robin."""
    return GAME_MODES[(match_number - 1) % len(GAME_MODES)]


def lineup_for_match(
    match_number: int,
    seat_ids: tuple[str, ...],
    played: dict[str, int] | None = None,
) -> dict[str, str]:
    """Pick the seats for a match. Selection is random, but biased for fair
    playtime: the models seated in the fewest matches so far get priority, so
    nobody gets starved by chance. `played` maps model label -> matches played
    (see server.leaderboard.matches_played); once everyone is level it's just a
    random draw. Deterministic given (match_number, seat_ids, played)."""
    rng = random.Random(f"lineup-{match_number}")
    counts = played or {}
    roster = list(ROSTER)
    rng.shuffle(roster)  # random base order → random tiebreak among equal counts
    roster.sort(key=lambda slug: counts.get(slug.split("/")[-1], 0))
    picks = roster[: len(seat_ids)]
    # Config guard: if the roster is smaller than the table, top up (with repeats).
    while len(picks) < len(seat_ids):
        picks.append(rng.choice(ROSTER))
    rng.shuffle(picks)  # randomise which seat each model takes
    return dict(zip(seat_ids, picks))


@dataclass(eq=False)  # identity hash so subscribers can live in a set
class Subscriber:
    queue: asyncio.Queue
    viewer: str = "spoiler_safe"


@dataclass
class LiveChannel:
    intermission_seconds: int = INTERMISSION_SECONDS
    subscribers: set[Subscriber] = field(default_factory=set)
    match_number: int = 0
    status: str = "starting"
    game_id: str = GAME_MODES[0]
    lineup: dict[str, str] = field(default_factory=dict)
    _slugs: dict[str, str] = field(default_factory=dict)
    # Catch-up state for late joiners: meta + latest snapshot per viewer.
    _meta: dict | None = None
    _latest_snapshots: dict[str, dict] = field(default_factory=dict)
    _channel_msg: dict = field(default_factory=dict)

    def subscribe(self, viewer: str) -> Subscriber:
        sub = Subscriber(queue=asyncio.Queue(maxsize=500), viewer=viewer)
        self.subscribers.add(sub)
        # Catch the new client up to the current picture.
        if self._meta is not None:
            sub.queue.put_nowait(self._meta)
        snapshot = self._latest_snapshots.get(viewer)
        if snapshot is not None:
            sub.queue.put_nowait(snapshot)
        if self._channel_msg:
            sub.queue.put_nowait(self._channel_msg)
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        self.subscribers.discard(sub)

    def _broadcast(self, record: dict) -> None:
        if record["kind"] == "meta":
            self._meta = record
            self._latest_snapshots = {}
        elif record["kind"] == "snapshot":
            self._latest_snapshots[record["viewer"]] = record
        elif record["kind"] == "channel":
            self._channel_msg = record

        for sub in tuple(self.subscribers):
            if record["kind"] == "snapshot" and record["viewer"] != sub.viewer:
                continue
            try:
                sub.queue.put_nowait(record)
            except asyncio.QueueFull:
                self.subscribers.discard(sub)

    def _announce(self, status: str, seconds_left: int | None = None) -> None:
        self.status = status
        self._broadcast(
            {
                "kind": "channel",
                "status": status,
                "match_number": self.match_number,
                "game": self.game_id,
                "lineup": self.lineup,
                "seconds_left": seconds_left,
            }
        )

    def _set_lineup(self, match_number: int) -> None:
        # Full slugs drive the actors; stage/leaderboard labels are the tails.
        # Bias seating toward the least-played models (fair rotation).
        self.game_id = game_for_match(match_number)
        seat_ids = GAMES[self.game_id].seat_ids
        self._slugs = lineup_for_match(match_number, seat_ids, matches_played(REPLAY_DIR))
        self.lineup = {
            seat: slug.split("/")[-1] for seat, slug in self._slugs.items()
        }

    async def run(self) -> None:
        while True:
            self.match_number += 1
            self._set_lineup(self.match_number)
            self._announce("live")
            try:
                await self._play_match()
            except Exception as error:  # keep the channel alive across bad matches
                self._broadcast(
                    {"kind": "error", "message": f"match crashed: {error!r}"}
                )
            # Announce the *next* lineup during the break.
            self._set_lineup(self.match_number + 1)
            for remaining in range(self.intermission_seconds, 0, -1):
                self._announce("intermission", seconds_left=remaining)
                await asyncio.sleep(1)

    async def _play_match(self) -> None:
        entry = GAMES[self.game_id]
        spec = entry.spec_factory()
        seed = f"live-{int(time.time())}-{self.match_number}"
        machine = StateMachine.new(spec, seed, entry.config_factory(self.lineup))
        actors: dict[str, object] = {}
        for seat_id, slug in self._slugs.items():
            fallback = entry.baseline_factory(f"{seed}:{slug}")
            if OPENROUTER_API_KEY and "/" in slug:
                actors[seat_id] = OpenRouterActor(
                    model=slug,
                    api_key=OPENROUTER_API_KEY,
                    fallback=fallback,
                    prompter=entry.prompter_factory(),
                    reasoning_override=REASONING_OVERRIDES.get(slug),
                )
            else:
                actors[seat_id] = fallback
        illegal_streak: dict[str, int] = {}
        host_rng = random.Random(seed + ":host")
        game_blurbs = {
            "codewords": "a Codenames-style word game",
            "taboo": (
                "Taboo — describers talk their team toward a secret word "
                "without saying it or its forbidden words, on a word clock"
            ),
        }
        llm_host = (
            LLMHost(
                api_key=OPENROUTER_API_KEY,
                game_blurb=game_blurbs.get(self.game_id, entry.title),
            )
            if OPENROUTER_API_KEY
            else None
        )

        # Running audience-visible transcript, so the host reads the whole
        # conversation (clues, reasoning, moves, reveals, its own lines) — the
        # only way it can catch a clue or a guess that leans too close to a
        # board word.
        transcript: list[str] = []

        def speaker(seat_id: str | None) -> str:
            return str(self.lineup.get(seat_id, seat_id)) if seat_id else "someone"

        def transcript_line(event) -> str | None:
            p = event.payload
            t = event.type
            if t == "clue_given":
                clue = p.get("clue", {}) or {}
                return f"{clue.get('team_id', '?')} clue: {str(clue.get('word', '')).upper()} for {clue.get('count', '?')}"
            if t == "actor_responded":
                said = p.get("public_deliberation")
                return f"{speaker(p.get('seat_id'))}: {said}" if said else None
            if t == "guess_proposed":
                return f"{speaker(p.get('seat_id'))} proposes {str(p.get('word', '')).upper()}"
            if t == "guess_confirmed":
                return f"{speaker(p.get('seat_id'))} confirms {str(p.get('word', '')).upper()}"
            if t == "guess_rejected":
                return f"{speaker(p.get('seat_id'))} vetoes {str(p.get('word', '')).upper()}"
            if t == "word_revealed":
                return f"{str(p.get('word', '')).upper()} revealed -> {p.get('assignment', '?')}"
            if t == "guesser_stopped":
                return f"{p.get('team_id', '?')} stops guessing"
            if t == "bonus_guess_reached":
                return f"{p.get('team_id', '?')} earns a bonus guess"
            if t == "bonus_clue_granted":
                return f"{p.get('team_id', '?')} earns a bonus clue"
            if t == "timer_expired":
                if self.game_id == "taboo":
                    return "TIME! The round is over"
                return "time expired — one final guess"
            if t == "illegal_action_applied":
                return f"ILLEGAL move: {p.get('message', p.get('rule_id', ''))}"
            # -- taboo ---------------------------------------------------------
            if t == "card_guessed":
                card = p.get("card", {}) or {}
                if p.get("via") == "spoken":
                    return (
                        f"{speaker(p.get('seat_id'))} SAID "
                        f"{str(card.get('target', '')).upper()} out loud mid-sentence — "
                        f"it counts! point {p.get('team_id', '?')}"
                    )
                return (
                    f"{speaker(p.get('seat_id'))} got it: "
                    f"{str(card.get('target', '')).upper()} — point {p.get('team_id', '?')}"
                )
            if t == "guess_incorrect":
                return f"{speaker(p.get('seat_id'))} guesses {str(p.get('word', '')).upper()} — no"
            if t == "guesser_passed":
                return f"{speaker(p.get('seat_id'))} passes the floor back"
            if t == "card_skipped":
                card = p.get("card", {}) or {}
                return f"{speaker(p.get('seat_id'))} skips {str(card.get('target', '')).upper()}"
            if t == "taboo_violation":
                card = p.get("card", {}) or {}
                return (
                    f"BUZZ! {speaker(p.get('seat_id'))} said '{p.get('said', '?')}' — "
                    f"{str(card.get('target', '')).upper()} lost, point to the other team"
                )
            if t == "round_ended":
                points = p.get("points", {}) or {}
                return (
                    f"round {p.get('round_number', '?')} over — "
                    f"red {points.get('red', 0)} · blue {points.get('blue', 0)}"
                )
            return None

        async def host_says(text: str) -> None:
            transcript.append(f"HOST: {text}")
            record = {"kind": "host", "text": text}
            archive.append(record)
            self._broadcast(record)
            await asyncio.sleep(
                max(SECONDS_PER_EVENT, len(text.split()) * SECONDS_PER_DELIBERATION_WORD)
            )

        def board_public() -> list[dict]:
            """Every board word plus the colour of the ones already revealed — all
            audience-visible; the host never sees an unrevealed card's key."""
            board = machine.state.board
            return [
                {
                    "word": board.word_for_cell(cell.id),
                    "revealed_as": board.key[cell.id] if cell.id in board.revealed else None,
                }
                for cell in board.grid.cells
            ]

        def host_context() -> dict:
            """Everything the audience can see: the running conversation plus
            the game's public score — never a hidden key or an unresolved card."""
            state = machine.state
            base = {
                "game": entry.title,
                "lineup": dict(self.lineup),
                "conversation": transcript[-60:],
            }
            if self.game_id == "taboo":
                return base | {
                    "round_number": state.round_number,
                    "total_rounds": state.total_rounds(),
                    "describing_team": state.current_team,
                    "points": dict(state.points),
                    "violations": dict(state.violations),
                    "hints_this_card": list(state.current_hints),
                }
            revealed = {"red": 0, "blue": 0, "neutral": 0}
            for cell_id in state.board.revealed:
                assignment = state.board.key[cell_id]
                if assignment in revealed:
                    revealed[assignment] += 1
            return base | {
                "turn_number": state.turn_number,
                "up_next_team": state.current_team,
                "score_revealed": revealed,
                "board_words": board_public(),
            }

        async def host_comments(moment: str, fallback: str, **extra: object) -> None:
            text = fallback
            if llm_host is not None:
                context = {**host_context(), **extra}
                text = await asyncio.to_thread(llm_host.line, moment, context, fallback)
            await host_says(text)

        async def host_maybe_comments(moment: str, **extra: object) -> None:
            # Optional beat: the host speaks only when it has something worth
            # saying. No LLM host (or a PASS) means silence — the card flip stands
            # on its own.
            if llm_host is None:
                return
            context = {**host_context(), **extra}
            text = await asyncio.to_thread(llm_host.maybe_line, moment, context)
            if text:
                await host_says(text)
        role_specs = spec.role_specs()
        archive: list[dict] = [
            {"kind": "meta", "game_id": spec.id, "seed": seed, "live": True}
        ]
        self._broadcast(archive[0])

        # Per-turn diagnostics (reasoning traces, raw responses, fallback
        # cause). Written to a sidecar file, never the replay: replays feed
        # the broadcast and RL rollouts; thinking traces must not.
        traces: list[dict] = []
        turns_by_seat: dict[str, int] = {}
        fallback_by_seat: dict[str, int] = {}

        def snapshot_records() -> list[dict]:
            return [
                {
                    "kind": "snapshot",
                    "viewer": viewer,
                    "view": spec.observe_viewer(machine.state, viewer),
                }
                for viewer in SNAPSHOT_VIEWERS
            ]

        for record in snapshot_records():
            archive.append(record)
            self._broadcast(record)

        # Open the show: the host welcomes the crowd and introduces the lineup.
        await host_comments(
            "match open — welcome the crowd and introduce tonight's models by team and role",
            host_rng.choice(HOST_START_LINES),
        )

        clues_seen = 0
        while not spec.is_terminal(machine.state):
            seat_id = spec.acting_seat(machine.state)
            if seat_id is None:
                break
            seat = machine.state.seat(seat_id)
            observation = machine.observe(seat_id)
            constraints = {}
            if observation.data.get("final_guess_only"):
                constraints["final_guess_one_word"] = True
            if observation.data.get("bonus_guess"):
                constraints["bonus_guess_two_words"] = True
            actor = actors[seat_id]
            # After two consecutive illegal moves, hand the seat to its
            # baseline fallback so the match can't stall on a stuck model.
            forced_fallback = (
                illegal_streak.get(seat_id, 0) >= 2 and isinstance(actor, OpenRouterActor)
            )
            if forced_fallback:
                actor = actor.fallback
            actor_input = ActorInput(
                seat=seat,
                role=role_specs[seat.role],
                observation=observation,
                time_left=observation.data.get("time_left"),
                constraints=constraints,
                # Seats hear the show: recent public table talk gives models
                # continuity between their otherwise stateless turns.
                transcript=tuple(transcript[-40:]),
            )
            # LLM calls block on network I/O — keep the event loop free.
            output = await asyncio.to_thread(actor.act, actor_input)

            team_before = machine.state.current_team
            event_start = len(machine.log.events)
            ruling, _ = machine.submit(
                seat_id,
                output.action,
                public_deliberation=output.public_deliberation,
            )

            seated = actors[seat_id]
            fell_back = forced_fallback or (
                isinstance(seated, OpenRouterActor) and seated.last_error is not None
            )
            turns_by_seat[seat_id] = turns_by_seat.get(seat_id, 0) + 1
            if fell_back:
                fallback_by_seat[seat_id] = fallback_by_seat.get(seat_id, 0) + 1
            traces.append(
                {
                    "seq": event_start,
                    "seat_id": seat_id,
                    "model": self._slugs.get(seat_id),
                    "action_type": output.action.type,
                    "public_deliberation": output.public_deliberation,
                    "reasoning": output.reasoning,
                    "reasoning_tokens": (
                        seated.last_reasoning_tokens
                        if isinstance(seated, OpenRouterActor)
                        else None
                    ),
                    "raw_response": output.raw_response,
                    "notes": output.notes,
                    "fallback": fell_back,
                    "error": (
                        "forced_baseline_after_illegal_streak"
                        if forced_fallback
                        else (seated.last_error if isinstance(seated, OpenRouterActor) else None)
                    ),
                }
            )
            if ruling.is_legal:
                illegal_streak[seat_id] = 0
            else:
                illegal_streak[seat_id] = illegal_streak.get(seat_id, 0) + 1

            delay = SECONDS_PER_EVENT
            clue_given = False
            revealed: tuple[str, str] | None = None
            fouled = False
            card_won: dict | None = None
            round_over = False
            for event in machine.log.events[event_start:]:
                if "audience" not in event.visibility:
                    continue
                record = {
                    "kind": "event",
                    "event": {
                        "seq": event.seq,
                        "type": event.type,
                        "payload": event.payload,
                        "visibility": list(event.visibility),
                    },
                }
                archive.append(record)
                self._broadcast(record)
                line = transcript_line(event)
                if line:
                    transcript.append(line)
                if event.type == "actor_responded":
                    # Pace on the text the audience actually sees, which the
                    # spec may have clipped to the remaining word clock.
                    shown = event.payload.get("public_deliberation") or ""
                    words = len(shown.split())
                    delay = max(delay, words * SECONDS_PER_DELIBERATION_WORD)
                elif event.type == "clue_given":
                    clue_given = True
                elif event.type == "word_revealed":
                    revealed = (
                        str(event.payload.get("word", "")),
                        str(event.payload.get("assignment", "")),
                    )
                elif event.type in ("illegal_action_applied", "taboo_violation"):
                    fouled = True
                elif event.type == "card_guessed":
                    card_won = dict(event.payload)
                elif event.type == "round_ended":
                    round_over = True
            for record in snapshot_records():
                archive.append(record)
                self._broadcast(record)
            await asyncio.sleep(delay)

            # The host reacts to what just happened. The match-over sign-off
            # below owns the final beat, so stay quiet once the game is decided.
            if clue_given:
                clues_seen += 1
            if spec.is_terminal(machine.state):
                pass
            elif fouled:
                await host_comments(
                    "foul — a model just broke a rule (in Taboo: said a forbidden "
                    "word and got buzzed); call it out with a light touch",
                    host_rng.choice(HOST_FOUL_LINES),
                )
            elif round_over:
                await host_comments(
                    "between rounds — recap the score and set up the team now "
                    "taking the floor",
                    host_rng.choice(HOST_ROUND_LINES).format(
                        team=machine.state.current_team.upper()
                    ),
                )
            elif card_won is not None and card_won.get("via") == "spoken":
                # The best moment in taboo: a guesser blurted the target
                # mid-sentence without realizing it counts. Always call it.
                card = card_won.get("card", {}) or {}
                await host_comments(
                    "a guesser just SAID the target word out loud mid-sentence "
                    "— maybe without even noticing — and by rule that counts. "
                    "Declare the accidental win with delight.",
                    "They said it! That counts — point scored!",
                    guessed_target=card.get("target"),
                    points=card_won.get("points"),
                )
            elif card_won is not None:
                card = card_won.get("card", {}) or {}
                await host_maybe_comments(
                    "a card was just guessed. Speak up ONLY if there's something "
                    "worth it: a hint that skated absurdly close to a forbidden "
                    "word, a wild guess that landed, or a dramatic score swing. "
                    "If it's routine, reply PASS.",
                    guessed_target=card.get("target"),
                    forbidden_words=card.get("forbidden"),
                    points=card_won.get("points"),
                )
            elif revealed is not None:
                # After a confirmed guess the host *may* chime in — but only if
                # it spots an irregularity or something genuinely interesting.
                # Routine guesses get no line; the card flip speaks for itself.
                word, color = revealed
                turn_over = machine.state.current_team != team_before
                await host_maybe_comments(
                    "a guess just resolved. Speak up ONLY if there's something worth it: "
                    "an irregularity (a clue or a guesser's reasoning that leaned too close "
                    "to a word actually on the board, or a borderline-illegal move) or a "
                    "genuinely surprising/dramatic swing. If it's a routine guess, reply PASS "
                    "and we'll just let the card flip.",
                    revealed_word=word,
                    revealed_color=color,
                    turn_over=turn_over,
                )
            elif clue_given and clues_seen > 1:
                # Top of a fresh round (skip the very first clue — the opening
                # announcement already set the stage).
                await host_comments(
                    "top of the round — recap the score and set the stage for this team's guesses",
                    host_rng.choice(HOST_ROUND_LINES).format(
                        team=machine.state.current_team.upper()
                    ),
                )

        if spec.is_terminal(machine.state):
            winner = machine.state.winner
            fallback = (
                host_rng.choice(HOST_END_LINES).format(winner=winner.upper())
                if winner
                else "All square — it ends in a draw!"
            )
            await host_comments(
                "match over — sign off with a verdict on the result",
                fallback,
                winner=winner,
                terminal_reason=machine.state.terminal_reason,
            )

        final = {
            "kind": "final_score",
            "score": machine.score(),
            # Exact per-seat fallback record so stats can discount turns the
            # model never actually played (see server.leaderboard).
            "fallback_turns": fallback_by_seat,
            "turns": turns_by_seat,
        }
        archive.append(final)
        self._broadcast(final)
        self._archive(archive, seed, traces)

    def _archive(self, records: list[dict], seed: str, traces: list[dict] | None = None) -> None:
        # Seed-based name: match numbers restart with the process, and stats
        # are aggregated from these files — never overwrite an old archive.
        # Gzipped (this stream is highly repetitive) and stored omniscient-only:
        # the spoiler-safe view is just the omniscient one minus each cell's
        # key, so it's re-derived on read (see server.main). Snapshots are ~86%
        # of the bytes, so dropping the redundant viewer roughly halves them
        # before gzip does the rest.
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        path = REPLAY_DIR / f"{seed}.jsonl.gz"
        with gzip.open(path, "wt") as f:
            for record in records:
                if (
                    record.get("kind") == "snapshot"
                    and record.get("viewer") != "audience_omniscient"
                ):
                    continue
                f.write(json.dumps(record) + "\n")
        # Thinking traces live beside the replay, not in it: they are
        # diagnostics for humans, and RL rollouts must never see them.
        if traces:
            trace_dir = REPLAY_DIR / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace_dir / f"{seed}.jsonl.gz", "wt") as f:
                for trace in traces:
                    f.write(json.dumps(trace) + "\n")
