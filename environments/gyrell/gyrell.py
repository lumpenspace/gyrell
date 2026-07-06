"""Verifiers RL environment for Codewords (the Codenames-like turngames game).

The trained policy always plays the RED team. Depending on `role`, it controls
the whole team (cluegiver + guessers, the flagship self-play-team task), only
the cluegiver, or only the guessers. Every seat the policy does not control is
played by a configurable driver:

- "scripted": the deterministic random-but-legal baseline from turngames;
- "policy": the same model/client being trained (frozen for those turns —
  they are environment transitions, not trajectory steps);
- an endpoint spec: any OpenAI-compatible endpoint, e.g.
  {"model": "openai/gpt-4.1-mini", "base_url": "https://openrouter.ai/api/v1",
   "api_key_var": "OPENROUTER_API_KEY", "temperature": 0.7}.

`opponent` configures the blue team, `partner` configures red seats the policy
does not control (only relevant for role="cluegiver"/"guesser").

Hidden information is preserved even when one policy plays several seats: each
seat keeps its own private message history (the guesser conversation never
contains the cluegiver's view of the key), which maps onto verifiers'
non-linear rollout support via the `get_prompt_messages` override. LLM-driven
non-policy seats keep private histories the same way.

Rewards are computed at the trajectory level from the final game outcome plus
protocol-compliance counters; the canonical game engine (`turngames`) remains
the single source of truth for rules, legality, and scoring. The policy's
illegal actions are ruled and penalized in-game by the arbiter; non-policy LLM
seats are sanitized and fall back to the scripted baseline instead, so a
misbehaving opponent model cannot stall or distort the game.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import verifiers as vf
from datasets import Dataset

from turngames.actors.baseline import CLUE_VOCAB as BASELINE_CLUE_VOCAB
from turngames.actors.baseline import BaselineCodewordsActor
from turngames.actors.llm import extract_json, render_board_for_prompt
from turngames.core import ActorInput, ActorOutput
from turngames.core.state_machine import StateMachine
from turngames.core.types import Action, Event
from turngames.games.codewords.spec import CodewordsSpec
from turngames.games.codewords.types import (
    ASSASSIN,
    BLUE,
    CLUEGIVER,
    CodewordsConfig,
    GUESSER,
    RED,
)

POLICY_TEAM = RED
OPPONENT_TEAM = BLUE

# Hard cap on non-policy actions between two policy turns, purely as a
# runaway-loop backstop; a legal game segment is far shorter.
MAX_OTHER_STEPS = 500

SYSTEM_PROMPT_TEMPLATE = """\
You are playing Codewords, a Codenames-style word game, on the {team} team.

Rules: the board is a 5x5 grid of words. Each word is secretly assigned to
red, blue, neutral, or the assassin. Cluegivers see the full key; guessers do
not. Teams alternate turns: the acting team's cluegiver gives a one-word clue
plus a count, then its guessers reveal words one at a time. Revealing your own
team's word lets you keep guessing (up to count + 1 words); a neutral or
opposing word ends your turn; the assassin loses the game instantly. A clue
must be a single alphabetic word and may not match or contain an unrevealed
board word. The first team to reveal all of its words wins.

Each message tells you which seat you are acting as (cluegiver or guesser),
what you can see, and which actions are allowed. Respond with ONLY a JSON
object, no markdown fences, of the form:
{{"public_deliberation": "<one short optional sentence of table talk>", \
"action": {{"type": "<one of the allowed action types>", ...action fields}}}}

Examples:
{{"public_deliberation": "Two ocean words here.", "action": {{"type": "give_clue", "word": "ocean", "count": 2}}}}
{{"public_deliberation": "Going with the safest link.", "action": {{"type": "choose", "word": "WHALE"}}}}
{{"public_deliberation": "Too risky to continue.", "action": {{"type": "stop"}}}}
"""


def system_prompt_for(team: str) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(team=team.upper())


SYSTEM_PROMPT = system_prompt_for(POLICY_TEAM)

FORMAT_REMINDER = (
    'Your previous reply could not be used. Respond with ONLY a JSON object of '
    'the form {"public_deliberation": "...", "action": {"type": "...", ...}} '
    "using one of the allowed action types."
)


# ---------------------------------------------------------------------------
# seat drivers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EndpointSpec:
    """OpenAI-compatible endpoint driving a non-policy seat.

    `extra_body` carries provider-specific request fields as a canonical JSON
    string (pass a dict in the spec; it is normalized). The main use is
    disabling reasoning on thinking models, which otherwise burn the whole
    token budget before emitting the JSON action, e.g.:
    {"model": "deepseek/deepseek-v4-flash",
     "base_url": "https://api.pinference.ai/api/v1",
     "api_key_var": "PRIME_API_KEY",
     "extra_body": {"reasoning": {"enabled": false}}}
    """

    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_var: str = "OPENROUTER_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 400
    timeout: float = 120.0
    extra_body: str = ""
    # Optional style/behavior persona appended to this seat's system prompt
    # (e.g. "You speak in formal academic register at all times."). Used to
    # build partner pools with deliberately divergent, stable registers.
    persona: str = ""
    # Anthropic prompt-cache breakpoints ("auto" | "on" | "off"). Seat
    # histories are growing prefixes, so ephemeral cache_control on the
    # system + tail message cuts cached input to ~1/10 price on providers
    # that honor it. "auto" enables it for anthropic/* models on OpenRouter
    # (OpenAI/DeepSeek/Gemini cache implicitly; the PI gateway ignores it).
    prompt_cache: str = "auto"


@dataclass(frozen=True)
class ProcessSpec:
    """Opponent team as a pure stochastic process — no model at all.

    Each opposing turn reveals `min_reveals..max_reveals` of that team's own
    cards, then ends on a random non-assassin card (theirs, ours, or
    neutral). Justified because the policy's team never interacts with the
    opponent's reasoning — only with which cards leave the board. Zero
    tokens, deterministic per seed, and the reveal range is a difficulty
    dial for curriculum bottom rungs. Excluding the assassin removes the
    opponent-suicide free-win channel entirely."""

    min_reveals: int = 1
    max_reveals: int = 3


def _normalize_driver(value: Any, arg_name: str) -> str | EndpointSpec | ProcessSpec:
    if value == "process":
        return ProcessSpec()
    if value in ("scripted", "policy"):
        return value
    if isinstance(value, (EndpointSpec, ProcessSpec)):
        return value
    if isinstance(value, dict) and value.get("type") == "process":
        fields = {k: v for k, v in value.items() if k != "type"}
        try:
            return ProcessSpec(**fields)
        except TypeError as error:
            raise ValueError(f"invalid process spec for {arg_name}: {error}") from error
    if isinstance(value, dict):
        fields = dict(value)
        if isinstance(fields.get("extra_body"), dict):
            fields["extra_body"] = json.dumps(fields["extra_body"], sort_keys=True)
        try:
            return EndpointSpec(**fields)
        except TypeError as error:
            raise ValueError(f"invalid endpoint spec for {arg_name}: {error}") from error
    raise ValueError(
        f"{arg_name} must be 'scripted', 'policy', or an endpoint spec dict "
        f"with at least a 'model' key; got {value!r}"
    )


# A combo assigns one driver per role on a team; a pool is a list of combos
# from which one is chosen per example (deterministic in the board seed, so
# all GRPO rollouts of the same example face the same matchup).
Combo = dict[str, "str | EndpointSpec"]


def _normalize_combo(value: Any, arg_name: str) -> Combo:
    if isinstance(value, dict) and "model" not in value:
        extra = set(value) - {CLUEGIVER, GUESSER}
        if extra or not value:
            raise ValueError(
                f"{arg_name} role-split dict may only have 'cluegiver'/'guesser' "
                f"keys; got {sorted(value)}"
            )
        return {
            role: _normalize_driver(value.get(role, "scripted"), f"{arg_name}.{role}")
            for role in (CLUEGIVER, GUESSER)
        }
    driver = _normalize_driver(value, arg_name)
    return {CLUEGIVER: driver, GUESSER: driver}


def _normalize_pool(value: Any, arg_name: str) -> list[Combo]:
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{arg_name} pool must not be empty")
        return [
            _normalize_combo(entry, f"{arg_name}[{index}]")
            for index, entry in enumerate(value)
        ]
    return [_normalize_combo(value, arg_name)]


def _driver_label(driver: str | EndpointSpec | ProcessSpec) -> str:
    if isinstance(driver, ProcessSpec):
        return f"process({driver.min_reveals}-{driver.max_reveals})"
    return driver if isinstance(driver, str) else driver.model


def _wants_cache_control(driver: EndpointSpec) -> bool:
    if driver.prompt_cache == "on":
        return True
    if driver.prompt_cache == "off":
        return False
    return driver.model.startswith("anthropic/") and "openrouter" in driver.base_url


def _annotate_cache_control(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the system message and the conversation tail as ephemeral cache
    breakpoints (Anthropic-style). Lookup walks backward from breakpoints, so
    each turn hits the prefix cached by the previous turn's tail breakpoint."""
    out = []
    last = len(messages) - 1
    for index, message in enumerate(messages):
        content = message.get("content")
        if (message.get("role") == "system" or index == last) and isinstance(
            content, str
        ):
            message = {
                **message,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        out.append(message)
    return out


# ---------------------------------------------------------------------------
# message helpers
# ---------------------------------------------------------------------------


def _message_dict(message: Any) -> dict[str, Any]:
    """Normalize a verifiers message (pydantic model or dict) to a plain dict."""
    if hasattr(message, "model_dump"):
        message = message.model_dump()
    if isinstance(message, dict):
        return message
    return {
        "role": getattr(message, "role", "user"),
        "content": getattr(message, "content", ""),
    }


def _message_text(message: Any) -> str:
    content = _message_dict(message).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(getattr(part, "text", "")))
        return "".join(parts)
    return ""


def _last_assistant_text(messages: Any) -> str:
    for message in reversed(list(messages)):
        if _message_dict(message).get("role") == "assistant":
            return _message_text(message)
    return ""


# ---------------------------------------------------------------------------
# reward functions (trajectory-level, computed from the final game snapshot)
# ---------------------------------------------------------------------------


def _result(state: vf.State) -> dict[str, Any]:
    return state.get("game_result") or {}


async def team_win(state: vf.State, **kwargs) -> float:
    """1.0 when the policy team won the game, else 0.0."""
    return 1.0 if _result(state).get("winner") == POLICY_TEAM else 0.0


async def word_progress(state: vf.State, **kwargs) -> float:
    """Metric: fraction of the policy team's own words revealed by game end
    (by anyone — including opponent mistakes; see earned_progress)."""
    result = _result(state)
    total = result.get("own_words_total") or 0
    remaining = result.get("own_words_remaining") or 0
    if total <= 0:
        return 0.0
    return (total - remaining) / total


async def earned_progress(state: vf.State, **kwargs) -> float:
    """Fraction of the policy team's own words revealed BY the policy team.

    Words the opponent reveals by mistake do not count: progress must be
    earned through the team's own legal clue-and-guess play. (Without this,
    a policy can learn to forfeit every turn and let a weak opponent
    self-destruct — a 100% win rate with zero legal moves.)"""
    result = _result(state)
    total = result.get("own_words_total") or 0
    if total <= 0:
        return 0.0
    return (result.get("own_words_by_team") or 0) / total


async def win_earned(state: vf.State, **kwargs) -> float:
    """Metric: 1.0 when the policy team won by revealing its own last word
    (as opposed to inheriting a win from opponent mistakes)."""
    result = _result(state)
    return (
        1.0
        if result.get("winner") == POLICY_TEAM
        and result.get("terminal_reason") == "all_team_words_revealed"
        else 0.0
    )


async def action_legality(state: vf.State, **kwargs) -> float:
    """Share of the policy's submitted actions the arbiter ruled legal."""
    result = _result(state)
    submitted = result.get("actions_submitted") or 0
    if submitted <= 0:
        return 0.0
    return 1.0 - (result.get("illegal_actions") or 0) / submitted


async def format_compliance(state: vf.State, **kwargs) -> float:
    """Share of policy turns that produced a parseable JSON action."""
    result = _result(state)
    turns = result.get("policy_turns") or 0
    if turns <= 0:
        return 0.0
    return 1.0 - (result.get("format_errors") or 0) / turns


async def illegal_action_penalty(state: vf.State, **kwargs) -> float:
    """0 when every submitted action was legal, down to -1 otherwise.

    Penalty-style on purpose: compliance must never be income. A flat bonus
    for legal actions taught a self-play policy to filibuster — emit legal
    'say' actions forever, never guess, collect the compliance floor with
    zero game risk."""
    return (await action_legality(state=state)) - 1.0


async def format_penalty(state: vf.State, **kwargs) -> float:
    """0 when every turn parsed cleanly, down to -1 otherwise."""
    return (await format_compliance(state=state)) - 1.0


async def assassin_loss(state: vf.State, **kwargs) -> float:
    """Metric: 1.0 when the policy team lost by revealing the assassin."""
    result = _result(state)
    lost = result.get("winner") not in (None, POLICY_TEAM)
    return 1.0 if lost and result.get("terminal_reason") == "assassin_revealed" else 0.0


async def game_finished(state: vf.State, **kwargs) -> float:
    """Metric: 1.0 when the game reached a natural end within max_turns."""
    return 1.0 if _result(state).get("winner") is not None else 0.0


async def turns_played(state: vf.State, **kwargs) -> float:
    """Metric: number of game turns (team alternations) played."""
    return float(_result(state).get("turn_number") or 0)


async def other_seat_fallbacks(state: vf.State, **kwargs) -> float:
    """Metric: LLM-driven non-policy turns that fell back to the baseline."""
    return float(_result(state).get("other_seat_fallbacks") or 0)


def build_rubric() -> vf.Rubric:
    # Reward design, learned the hard way (both failure modes observed in
    # real training runs):
    # 1. Only outcomes the policy can influence pay: a win earned by
    #    revealing the team's own words, and progress from the team's own
    #    reveals. Rewarding raw wins taught a policy to forfeit every turn
    #    and let a weak scripted opponent self-destruct.
    # 2. Protocol quality is a penalty, never income: flat compliance
    #    bonuses taught a self-play policy to filibuster (legal 'say'
    #    actions forever, zero game risk, collect the floor).
    return vf.Rubric(
        funcs=[
            win_earned,
            earned_progress,
            illegal_action_penalty,
            format_penalty,
            action_legality,
            format_compliance,
            team_win,
            word_progress,
            assassin_loss,
            game_finished,
            turns_played,
            other_seat_fallbacks,
        ],
        weights=[1.0, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class CodewordsEnv(vf.MultiTurnEnv):
    """Multi-turn verifiers environment over the turngames Codewords engine."""

    def __init__(
        self,
        role: str = "team",
        opponent: Any = "scripted",
        partner: Any = "scripted",
        guessers_per_team: int = 1,
        deck: str = "classic-en",
        num_train_examples: int = 2000,
        num_eval_examples: int = 200,
        seed: int = 0,
        max_turns: int = 50,
        max_consecutive_says: int = 2,
        cache_endpoint_seats: bool = True,
        render_deliberation: bool = False,
        expose_partner_labels: bool = False,
        endpoint_seat_memory: str = "digest",
        partner_dossiers: dict[str, str] | None = None,
        dossier_rate: float = 0.0,
        false_dossier_rate: float = 0.0,
        **kwargs,
    ):
        if role not in ("team", "cluegiver", "guesser"):
            raise ValueError("role must be one of: team, cluegiver, guesser")
        if guessers_per_team < 1:
            raise ValueError("guessers_per_team must be at least 1")

        self.role = role
        self.max_consecutive_says = max_consecutive_says
        # E5 hook: render table talk into other seats' transcripts. Trade-off:
        # adversary context then depends on free text, not just structured
        # actions, so endpoint-cache hit rates drop (divergence propagates
        # through deliberation). Off for cheap training, on for accommodation
        # experiments.
        self.render_deliberation = render_deliberation
        # E4 hook (named condition): show the policy which model plays each
        # non-policy seat.
        self.expose_partner_labels = expose_partner_labels
        # How endpoint-driven seats see the game. "digest" (default): each
        # call is stateless — a compact game record (clues given, cards
        # revealed and their colors, per turn) rebuilt from the event log,
        # plus the current observation. The guesser role is nearly Markovian,
        # so this is ~1/20th the tokens of a conversation and makes the
        # response cache path-independent (a true transposition table: same
        # game state via different paths → same key). "conversation": full
        # per-seat growing history — use when partner statefulness matters
        # (e.g. an endpoint cluegiver that should remember its own intent).
        if endpoint_seat_memory not in ("digest", "conversation"):
            raise ValueError("endpoint_seat_memory must be 'digest' or 'conversation'")
        self.endpoint_seat_memory = endpoint_seat_memory
        # E11 hook: behavioral dossiers about partners, shown to policy seats.
        # `partner_dossiers` maps a driver label (e.g. "openai/gpt-4.1-mini")
        # to a summary of that partner's previous games. `dossier_rate` is
        # the fraction of examples where the policy sees a dossier for its
        # seated partner; of those, `false_dossier_rate` get a mismatched
        # dossier instead (deterministic per board seed, shared across a GRPO
        # group). Training with a mix of real and false dossiers teaches
        # calibrated dossier use — a defeasible prior weighed against
        # observed behavior — and prevents dossier-only shortcut policies.
        # The condition ("real"/"false"/none) is recorded in game_result.
        self.partner_dossiers = partner_dossiers or {}
        self.dossier_rate = dossier_rate
        self.false_dossier_rate = false_dossier_rate
        if false_dossier_rate > 0 and len(self.partner_dossiers) < 2:
            raise ValueError("false dossiers need at least two entries in partner_dossiers")
        # Endpoint-seat responses are a deterministic function of (model,
        # seat history): everything in a seat's conversation derives from the
        # seeded event log. Rollouts of the same board therefore share
        # adversary turns until the policy's actions diverge, so we memoize
        # them (and deduplicate in-flight identical calls, so a GRPO group
        # hitting the same opening fires one API call, not eight racing
        # ones). "policy"-driven seats are never cached: their weights change
        # every training step.
        self.cache_endpoint_seats = cache_endpoint_seats
        self._seat_response_cache: dict[str, asyncio.Future] = {}
        self.guessers_per_team = guessers_per_team
        self.deck = deck
        self.seed = seed
        self.spec = CodewordsSpec()
        self.role_specs = self.spec.role_specs()
        self.policy_seats = self._policy_seats(role, guessers_per_team)

        self.opponent_pool = _normalize_pool(opponent, "opponent")
        self.partner_pool = _normalize_pool(partner, "partner")
        self._endpoint_clients: dict[EndpointSpec, Any] = {}
        for combo in self.opponent_pool + self.partner_pool:
            for driver in combo.values():
                if isinstance(driver, EndpointSpec):
                    self._endpoint_client(driver)  # fail fast on missing key/module

        dataset, eval_dataset = self._build_datasets(
            num_train_examples, num_eval_examples
        )
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=SYSTEM_PROMPT,
            rubric=build_rubric(),
            message_type="chat",
            max_turns=max_turns,
            **kwargs,
        )

    # -- configuration -------------------------------------------------------

    @staticmethod
    def _team_seats(team: str, guessers_per_team: int) -> tuple[str, ...]:
        return (f"{team}_cluegiver",) + tuple(
            f"{team}_guesser_{index + 1}" for index in range(guessers_per_team)
        )

    @classmethod
    def _policy_seats(cls, role: str, guessers_per_team: int) -> tuple[str, ...]:
        seats = cls._team_seats(POLICY_TEAM, guessers_per_team)
        if role == "team":
            return seats
        if role == "cluegiver":
            return seats[:1]
        return seats[1:]

    def _game_info(self, index: int) -> dict[str, Any]:
        return {
            "game_seed": f"codewords-{self.seed}-{index}",
            "starting_team": POLICY_TEAM if index % 2 == 0 else OPPONENT_TEAM,
        }

    @staticmethod
    def _pick_combo(pool: list[Combo], game_seed: str, which: str) -> Combo:
        digest = hashlib.sha256(f"{game_seed}:{which}".encode()).hexdigest()
        return pool[int(digest, 16) % len(pool)]

    def _endpoint_client(self, spec: EndpointSpec) -> Any:
        client = self._endpoint_clients.get(spec)
        if client is None:
            import openai

            api_key = os.environ.get(spec.api_key_var)
            if not api_key and spec.api_key_var == "PRIME_API_KEY":
                # Mirror verifiers' behavior: fall back to the `prime login`
                # credentials so PI-inference seats work out of the box.
                config_path = os.path.expanduser("~/.prime/config.json")
                if os.path.exists(config_path):
                    with open(config_path) as handle:
                        api_key = json.load(handle).get("api_key", "")
            if not api_key:
                raise ValueError(
                    f"environment variable {spec.api_key_var} is empty or unset "
                    f"(needed for endpoint model {spec.model!r})"
                )
            client = openai.AsyncOpenAI(
                base_url=spec.base_url, api_key=api_key, timeout=spec.timeout
            )
            self._endpoint_clients[spec] = client
        return client

    # -- rendering -----------------------------------------------------------

    def _format_events(self, events: list[Event]) -> str:
        lines: list[str] = []
        for event in events:
            if "audience" not in event.visibility:
                continue
            p = event.payload
            if (
                self.render_deliberation
                and event.type == "actor_responded"
                and p.get("public_deliberation")
            ):
                lines.append(
                    f"- {p.get('seat_id')} said: \"{p.get('public_deliberation')}\""
                )
            elif event.type == "clue_given":
                clue = p.get("clue") or {}
                lines.append(
                    f"- {p.get('seat_id')} gave the clue: "
                    f"{str(clue.get('word', '')).upper()} ({clue.get('count')})"
                )
            elif event.type == "guess_proposed":
                lines.append(f"- {p.get('seat_id')} proposed: {p.get('word')}")
            elif event.type == "guess_confirmed":
                lines.append(f"- {p.get('seat_id')} confirmed: {p.get('word')}")
            elif event.type == "guess_rejected":
                lines.append(f"- {p.get('seat_id')} rejected: {p.get('word')}")
            elif event.type == "word_revealed":
                lines.append(
                    f"- {p.get('word')} was revealed: it belongs to "
                    f"{p.get('assignment')}"
                )
            elif event.type == "guesser_stopped":
                lines.append(f"- {p.get('seat_id')} stopped the team's turn")
            elif event.type == "illegal_action_applied":
                lines.append(
                    f"- {p.get('seat_id')} made an illegal move "
                    f"({p.get('rule_id')}): {p.get('message')}"
                )
            elif event.type == "game_ended":
                lines.append(
                    f"- Game over. Winner: {p.get('winner')} "
                    f"({p.get('terminal_reason')})"
                )
        return "\n".join(lines)

    def _allowed_actions(
        self, state: vf.State, seat_id: str
    ) -> tuple[dict[str, Any], ...]:
        """Legal actions for a seat, minus 'say' once the seat has exhausted
        its consecutive-say quota (a structural cap on filibustering)."""
        machine: StateMachine = state["machine"]
        allowed = self.spec.legal_actions(machine.state, seat_id)
        if (
            self.max_consecutive_says > 0
            and state.get("say_streak", {}).get(seat_id, 0)
            >= self.max_consecutive_says
        ):
            allowed = tuple(
                schema for schema in allowed if schema["type"] != "say"
            )
        return allowed

    def _render_turn_message(
        self, state: vf.State, seat_id: str, transcript: str
    ) -> str:
        machine: StateMachine = state["machine"]
        seat = machine.state.seat(seat_id)
        observation = self.spec.observe(machine.state, seat_id)
        data = observation.data
        role_spec = self.role_specs[seat.role]
        sections: list[str] = []
        if transcript:
            sections.append(f"Since your seat last acted:\n{transcript}")
        sections.append(
            f"You are acting as seat {seat.id}: the {seat.team_id} team's "
            f"{'cluegiver (you see the key)' if seat.role == CLUEGIVER else 'guesser (no key)'}. "
            f"{role_spec.description}"
        )
        dossier = state.get("dossier")
        if dossier and seat_id in self.policy_seats:
            sections.append(
                "Notes on your guesser from previous games:\n" + dossier["text"]
            )
        if self.expose_partner_labels:
            labels = state.get("seat_driver_labels") or {}
            if labels:
                sections.append(
                    "Seat assignments: "
                    + "; ".join(
                        f"{other} is played by {label}"
                        for other, label in sorted(labels.items())
                    )
                )
        sections.append("Board:\n" + render_board_for_prompt(data))
        state_lines = [
            f"Turn {data.get('turn_number')} — acting team: {data.get('current_team')}.",
            f"Current clue: {data.get('current_clue')}",
            f"Guesses made this turn: {data.get('guesses_this_turn')}",
        ]
        if data.get("pending_proposal") is not None:
            state_lines.append(f"Pending proposal: {data.get('pending_proposal')}")
        sections.append("\n".join(state_lines))
        sections.append(
            "Allowed actions right now: "
            + json.dumps(list(self._allowed_actions(state, seat_id)))
        )
        sections.append(
            'Respond with ONLY the JSON object: {"public_deliberation": "...", '
            '"action": {"type": "...", ...}}'
        )
        return "\n\n".join(sections)

    def _game_digest(self, machine: StateMachine) -> str:
        """Compact public game record: per clue, what was revealed and its
        color (plus table talk when render_deliberation is on).

        Sufficient for game *mechanics*; deliberately NOT sufficient for
        theory of mind — a guesser's accumulated model of its cluegiver
        (inferences from failed guesses, partner style, cross-game priors)
        lives in interaction history, which digest mode ablates. The
        digest/conversation performance gap for the same partner model is
        therefore a measurement of what partner-memory is worth."""
        entries: list[tuple[str, list[str]]] = []
        for event in machine.log.events:
            p = event.payload
            if event.type == "clue_given":
                clue = p.get("clue") or {}
                entries.append(
                    (
                        f"{clue.get('team_id')} clue "
                        f"{str(clue.get('word', '')).upper()} ({clue.get('count')})",
                        [],
                    )
                )
            elif event.type == "word_revealed" and entries:
                entries[-1][1].append(f"{p.get('word')} ({p.get('assignment')})")
            elif event.type == "guesser_stopped" and entries:
                entries[-1][1].append("stopped")
            elif (
                self.render_deliberation
                and event.type == "actor_responded"
                and p.get("public_deliberation")
                and entries
            ):
                entries[-1][1].append(
                    f"{p.get('seat_id')} said \"{p.get('public_deliberation')}\""
                )
            elif event.type == "illegal_action_applied" and p.get(
                "action_type"
            ) == "give_clue":
                entries.append(("illegal clue, turn forfeited", []))
        if not entries:
            return ""
        lines = [
            f"- {head} -> {', '.join(reveals) if reveals else 'no guesses'}"
            for head, reveals in entries
        ]
        return "Game so far:\n" + "\n".join(lines)

    def _seat_system_prompt(self, state: vf.State, seat_id: str) -> str:
        machine: StateMachine = state["machine"]
        prompt = system_prompt_for(machine.state.seat(seat_id).team_id)
        driver = state.get("seat_drivers", {}).get(seat_id)
        if isinstance(driver, EndpointSpec) and driver.persona:
            prompt = f"{prompt}\n{driver.persona}"
        return prompt

    def _seat_turn_message(self, state: vf.State, seat_id: str) -> dict[str, Any]:
        """Render the next user message for a conversational seat and advance
        its private event cursor."""
        machine: StateMachine = state["machine"]
        cursors = state["seat_event_seq"]
        transcript = self._format_events(list(machine.log.events[cursors[seat_id]:]))
        cursors[seat_id] = len(machine.log.events)
        content = self._render_turn_message(state, seat_id, transcript)
        return {"role": "user", "content": content}

    def _note_action(self, state: vf.State, seat_id: str, action_type: str) -> None:
        streaks = state.setdefault("say_streak", {})
        streaks[seat_id] = streaks.get(seat_id, 0) + 1 if action_type == "say" else 0

    # -- rollout state -------------------------------------------------------

    async def setup_state(self, state: vf.State) -> None:
        info = state.get("info") or {}
        if isinstance(info, str):
            info = json.loads(info)
        config = CodewordsConfig(
            deck=self.deck,
            guessers_per_team=self.guessers_per_team,
            starting_team=info["starting_team"],
            turn_time_limit_words=None,
        )
        machine = StateMachine.new(self.spec, info["game_seed"], config)
        state["machine"] = machine

        # Resolve this game's matchup: one combo per team, chosen
        # deterministically from the pools by the board seed (all GRPO
        # rollouts of the same example face the same matchup).
        opponent_combo = self._pick_combo(self.opponent_pool, info["game_seed"], "opponent")
        partner_combo = self._pick_combo(self.partner_pool, info["game_seed"], "partner")
        seat_drivers: dict[str, str | EndpointSpec] = {}
        for team, combo in ((OPPONENT_TEAM, opponent_combo), (POLICY_TEAM, partner_combo)):
            for seat_id in self._team_seats(team, self.guessers_per_team):
                if seat_id not in self.policy_seats:
                    seat = machine.state.seat(seat_id)
                    seat_drivers[seat_id] = combo[seat.role]
        state["seat_drivers"] = seat_drivers
        state["seat_driver_labels"] = {
            seat_id: _driver_label(driver) for seat_id, driver in seat_drivers.items()
        }

        # Every non-policy seat gets a seeded baseline actor: it either drives
        # the seat directly ("scripted") or serves as the fallback when an
        # LLM-driven seat misbehaves.
        state["baseline_actors"] = {
            seat_id: BaselineCodewordsActor(seed=f"{info['game_seed']}:{seat_id}")
            for seat_id in seat_drivers
        }
        # Seats that keep growing conversations: policy seats always; other
        # seats only when their driver needs statefulness ("policy" driver,
        # or endpoint seats in "conversation" memory mode). Digest-mode
        # endpoint seats are stateless — no history, no cursor.
        conversational = list(self.policy_seats) + [
            seat_id
            for seat_id, driver in seat_drivers.items()
            if driver == "policy"
            or (
                isinstance(driver, EndpointSpec)
                and self.endpoint_seat_memory == "conversation"
            )
        ]
        state["seat_histories"] = {
            seat_id: [
                {
                    "role": "system",
                    "content": self._seat_system_prompt(state, seat_id),
                }
            ]
            for seat_id in conversational
        }
        state["seat_event_seq"] = {
            seat_id: len(machine.log.events) for seat_id in conversational
        }
        state["counters"] = {
            "policy_turns": 0,
            "format_errors": 0,
            "actions_submitted": 0,
            "illegal_actions": 0,
            "other_seat_fallbacks": 0,
            "endpoint_cache_hits": 0,
            "endpoint_calls": 0,
        }

        self._assign_dossier(state, info)
        await self._advance_others(state)
        self._snapshot_result(state)
        if self.spec.is_terminal(machine.state):
            # The game ended before the policy ever acted (possible when a
            # non-scripted opponent starts and immediately ends the game).
            state["final_env_response"] = [self._final_message(state)]
            return

        acting_seat = self.spec.acting_seat(machine.state)
        first_message = self._seat_turn_message(state, acting_seat)
        state["acting_seat"] = acting_seat
        history = state["seat_histories"][acting_seat]
        history.append(first_message)
        # Replace the dataset's placeholder prompt with what the model will
        # actually see; the trajectory's first step prompts come from here.
        state["prompt"] = list(history)

    def _assign_dossier(self, state: vf.State, info: dict[str, Any]) -> None:
        """Deterministically decide whether this rollout's policy sees a
        partner dossier, and whether it is the real one or a decoy."""
        state["dossier"] = None
        if not self.partner_dossiers or self.dossier_rate <= 0:
            return
        partner_labels = [
            label
            for seat_id, label in state["seat_driver_labels"].items()
            if seat_id.startswith(f"{POLICY_TEAM}_") and label in self.partner_dossiers
        ]
        if not partner_labels:
            return
        digest = hashlib.sha256(f"{info['game_seed']}:dossier".encode()).hexdigest()
        roll = int(digest[:8], 16) / 0xFFFFFFFF
        if roll >= self.dossier_rate:
            return
        true_label = partner_labels[0]
        false_roll = int(digest[8:16], 16) / 0xFFFFFFFF
        if false_roll < self.false_dossier_rate:
            decoys = sorted(set(self.partner_dossiers) - {true_label})
            label = decoys[int(digest[16:24], 16) % len(decoys)]
            kind = "false"
        else:
            label, kind = true_label, "real"
        state["dossier"] = {"kind": kind, "text": self.partner_dossiers[label]}

    def _snapshot_result(self, state: vf.State) -> None:
        machine = state.get("machine")
        if machine is None:
            return
        game_state = machine.state
        own_total = 9 if game_state.config.starting_team == POLICY_TEAM else 8
        own_words_by_team = sum(
            1
            for event in machine.log.events
            if event.type == "word_revealed"
            and event.payload.get("team_id") == POLICY_TEAM
            and event.payload.get("assignment") == POLICY_TEAM
        )
        state["game_result"] = {
            "policy_team": POLICY_TEAM,
            "winner": game_state.winner,
            "terminal_reason": game_state.terminal_reason,
            "turn_number": game_state.turn_number,
            "own_words_total": own_total,
            "own_words_by_team": own_words_by_team,
            "own_words_remaining": game_state.board.remaining_for_team(POLICY_TEAM),
            "opponent_words_remaining": game_state.board.remaining_for_team(
                OPPONENT_TEAM
            ),
            "illegal_clues": dict(game_state.illegal_clues),
            "seat_models": state.get("seat_driver_labels", {}),
            "dossier": (state.get("dossier") or {}).get("kind"),
            **state.get("counters", {}),
        }

    def _final_message(self, state: vf.State) -> dict[str, Any]:
        result = state["game_result"]
        won = result.get("winner") == POLICY_TEAM
        return {
            "role": "user",
            "content": (
                f"Game over: the {result.get('winner')} team wins "
                f"({result.get('terminal_reason')}). "
                f"You {'won' if won else 'lost'}."
            ),
        }

    @vf.cleanup
    async def cleanup_game(self, state: vf.State) -> None:
        self._snapshot_result(state)
        state.pop("machine", None)
        state.pop("baseline_actors", None)
        state.pop("seat_histories", None)
        state.pop("seat_event_seq", None)
        state.pop("seat_drivers", None)

    # -- non-policy seats ----------------------------------------------------

    def _baseline_output(self, state: vf.State, seat_id: str) -> ActorOutput:
        machine: StateMachine = state["machine"]
        seat = machine.state.seat(seat_id)
        constraints: dict[str, Any] = (
            {"final_guess_one_word": True} if machine.state.final_guess_only else {}
        )
        actor_input = ActorInput(
            seat=seat,
            role=self.role_specs[seat.role],
            observation=self.spec.observe(machine.state, seat_id),
            constraints=constraints,
        )
        return state["baseline_actors"][seat_id].act(actor_input)

    def _coerce_action(
        self, state: vf.State, seat_id: str, text: str
    ) -> ActorOutput | None:
        """Parse and sanitize an LLM reply for a non-policy seat. Returns None
        when the reply cannot be turned into a legal-looking action."""
        machine: StateMachine = state["machine"]
        parsed = extract_json(text)
        if parsed is None:
            return None
        action_data = parsed.get("action")
        if not isinstance(action_data, dict):
            return None
        allowed = {
            schema["type"] for schema in self._allowed_actions(state, seat_id)
        }
        action_type = str(action_data.get("type", ""))
        if action_type not in allowed:
            return None
        deliberation = str(parsed.get("public_deliberation") or "")

        board = machine.state.board
        unrevealed = {word.casefold(): word for word in board.unrevealed_words()}

        if action_type in ("propose", "choose"):
            word = str(action_data.get("word", "")).strip().casefold()
            if word not in unrevealed:
                return None
            return ActorOutput(
                public_deliberation=deliberation,
                action=Action.of(action_type, word=unrevealed[word]),
                raw_response=text,
            )
        if action_type == "give_clue":
            word = str(action_data.get("word", "")).strip()
            if not re.fullmatch(r"[A-Za-z]+", word):
                return None
            clue_norm = word.casefold()
            if any(clue_norm in b or b in clue_norm for b in unrevealed):
                return None
            try:
                count = max(1, min(9, int(action_data.get("count", 1))))
            except (TypeError, ValueError):
                return None
            return ActorOutput(
                public_deliberation=deliberation,
                action=Action.of("give_clue", word=word, count=count),
                raw_response=text,
            )
        # say / confirm / reject / stop carry no payload worth sanitizing.
        return ActorOutput(
            public_deliberation=deliberation,
            action=Action.of(action_type),
            raw_response=text,
        )

    async def _endpoint_completion(
        self, state: vf.State, driver: EndpointSpec, history: list[dict[str, Any]]
    ) -> str:
        """Call an endpoint seat, memoized on (endpoint, exact history)."""

        async def call() -> str:
            client = self._endpoint_client(driver)
            messages = (
                _annotate_cache_control(history)
                if _wants_cache_control(driver)
                else history
            )
            completion = await client.chat.completions.create(
                model=driver.model,
                messages=messages,
                temperature=driver.temperature,
                max_tokens=driver.max_tokens,
                extra_body=json.loads(driver.extra_body) if driver.extra_body else None,
            )
            return completion.choices[0].message.content or ""

        counters = state["counters"]
        if not self.cache_endpoint_seats:
            counters["endpoint_calls"] += 1
            return await call()
        key = hashlib.sha256(
            json.dumps(
                [
                    driver.model,
                    driver.base_url,
                    driver.temperature,
                    driver.max_tokens,
                    driver.extra_body,
                    history,
                ],
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cached = self._seat_response_cache.get(key)
        if cached is not None:
            text = await asyncio.shield(cached)
            counters["endpoint_cache_hits"] += 1
            return text
        if len(self._seat_response_cache) > 50_000:
            self._seat_response_cache.clear()
        counters["endpoint_calls"] += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._seat_response_cache[key] = future
        try:
            text = await call()
        except BaseException as error:
            # Don't cache failures: let later rollouts retry the endpoint.
            self._seat_response_cache.pop(key, None)
            if not future.done():
                future.set_exception(error)
            raise
        future.set_result(text)
        return text

    async def _llm_seat_output(
        self, state: vf.State, seat_id: str, driver: str | EndpointSpec
    ) -> ActorOutput:
        """One turn of an LLM-driven non-policy seat, with baseline fallback."""
        machine: StateMachine = state["machine"]
        stateless = (
            isinstance(driver, EndpointSpec)
            and self.endpoint_seat_memory == "digest"
        )
        if stateless:
            digest = self._game_digest(machine)
            turn = self._render_turn_message(state, seat_id, transcript="")
            messages = [
                {
                    "role": "system",
                    "content": self._seat_system_prompt(state, seat_id),
                },
                {
                    "role": "user",
                    "content": f"{digest}\n\n{turn}" if digest else turn,
                },
            ]
            history = None
        else:
            history = state["seat_histories"][seat_id]
            history.append(self._seat_turn_message(state, seat_id))
            messages = history
        text = ""
        try:
            if driver == "policy":
                response = await self.get_model_response(state, prompt=messages)
                text = _message_text(response.message)
            else:
                text = await self._endpoint_completion(state, driver, messages)
        except Exception as error:  # keep the game going on any endpoint failure
            self.logger.warning(
                "endpoint failure for seat %s (%s); using baseline fallback",
                seat_id,
                error,
            )
        output = self._coerce_action(state, seat_id, text) if text else None
        if output is None:
            state["counters"]["other_seat_fallbacks"] += 1
            output = self._baseline_output(state, seat_id)
            text = json.dumps(
                {
                    "public_deliberation": output.public_deliberation,
                    "action": {
                        "type": output.action.type,
                        **output.action.payload,
                    },
                }
            )
        if history is not None:
            history.append({"role": "assistant", "content": text})
        return output

    def _process_output(
        self, state: vf.State, seat_id: str, spec: ProcessSpec
    ) -> ActorOutput:
        """One action of a process-driven seat: the team's turn is planned as
        k own-card reveals plus one random non-assassin terminal card, then
        executed through the ordinary engine actions."""
        machine: StateMachine = state["machine"]
        game_state = machine.state
        seat = game_state.seat(seat_id)
        team = seat.team_id
        rng = state["baseline_actors"][seat_id].rng
        plans = state.setdefault("process_plans", {})

        if seat.role == CLUEGIVER:
            board = game_state.board
            own = [
                word
                for word in board.unrevealed_words()
                if board.assignment_for_word(word) == team
            ]
            k = max(1, min(rng.randint(spec.min_reveals, spec.max_reveals), len(own)))
            targets = rng.sample(own, k)
            terminal_pool = [
                word
                for word in board.unrevealed_words()
                if word not in targets
                and board.assignment_for_word(word) != ASSASSIN
            ]
            plan = targets + ([rng.choice(terminal_pool)] if terminal_pool else [])
            plans[team] = plan
            board_norms = [w.casefold() for w in board.unrevealed_words()]
            vocab = [
                w
                for w in BASELINE_CLUE_VOCAB
                if all(w not in b and b not in w for b in board_norms)
            ]
            clue_word = rng.choice(vocab or ["signal"])
            return ActorOutput(
                public_deliberation="",
                action=Action.of("give_clue", word=clue_word, count=k),
            )

        plan = plans.get(team) or []
        allowed = {
            schema["type"]
            for schema in self.spec.legal_actions(machine.state, seat_id)
        }
        board = game_state.board
        plan[:] = [w for w in plan if board.cell_id_for_word(w) not in board.revealed]
        if "confirm" in allowed and game_state.pending_proposal is not None:
            return ActorOutput(public_deliberation="", action=Action.of("confirm"))
        if plan and "choose" in allowed:
            return ActorOutput(
                public_deliberation="", action=Action.of("choose", word=plan.pop(0))
            )
        if plan and "propose" in allowed:
            return ActorOutput(
                public_deliberation="", action=Action.of("propose", word=plan.pop(0))
            )
        return ActorOutput(public_deliberation="", action=Action.of("stop"))

    async def _advance_others(self, state: vf.State) -> None:
        """Play non-policy seats until a policy seat must act or the game ends."""
        machine: StateMachine = state["machine"]
        for _ in range(MAX_OTHER_STEPS):
            if self.spec.is_terminal(machine.state):
                return
            seat_id = self.spec.acting_seat(machine.state)
            if seat_id is None or seat_id in self.policy_seats:
                return
            driver = state["seat_drivers"][seat_id]
            if isinstance(driver, ProcessSpec):
                output = self._process_output(state, seat_id, driver)
            elif driver == "scripted":
                output = self._baseline_output(state, seat_id)
            else:
                output = await self._llm_seat_output(state, seat_id, driver)
            machine.submit(
                seat_id,
                output.action,
                public_deliberation=output.public_deliberation,
                raw_response=output.raw_response,
            )
            self._note_action(state, seat_id, output.action.type)
        raise RuntimeError("non-policy seats exceeded MAX_OTHER_STEPS")

    # -- turn processing -----------------------------------------------------

    async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
        if len(state["trajectory"]) == 0:
            return state["prompt"]

        seat_id = state["acting_seat"]
        completion = state["trajectory"][-1]["completion"]
        history = state["seat_histories"][seat_id]
        history.extend(_message_dict(message) for message in completion)

        env_messages = await self.env_response(history, state)
        if state.get("final_env_response") is not None:
            return env_messages

        next_seat = state["acting_seat"]
        next_history = state["seat_histories"][next_seat]
        next_history.extend(_message_dict(message) for message in env_messages)
        return list(next_history)

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        """Apply the acting policy seat's action and play the other seats
        until the policy's next turn (or the end of the game)."""
        machine: StateMachine = state["machine"]
        seat_id = state["acting_seat"]
        counters = state["counters"]
        counters["policy_turns"] += 1

        parsed = extract_json(_last_assistant_text(messages))
        action_data = (parsed or {}).get("action")
        allowed = {
            schema["type"] for schema in self._allowed_actions(state, seat_id)
        }
        if (
            not isinstance(action_data, dict)
            or str(action_data.get("type", "")) not in allowed
        ):
            counters["format_errors"] += 1
            reminder = (
                f"{FORMAT_REMINDER}\n\n"
                + self._render_turn_message(state, seat_id, transcript="")
            )
            return [{"role": "user", "content": reminder}]

        payload = {key: value for key, value in action_data.items() if key != "type"}
        if "count" in payload:
            try:
                payload["count"] = int(payload["count"])
            except (TypeError, ValueError):
                pass  # let the arbiter rule it illegal
        action = Action(type=str(action_data["type"]), payload=payload)
        deliberation = str(parsed.get("public_deliberation") or "")

        ruling, _ = machine.submit(
            seat_id,
            action,
            public_deliberation=deliberation,
            raw_response=_last_assistant_text(messages),
        )
        counters["actions_submitted"] += 1
        if not ruling.is_legal:
            counters["illegal_actions"] += 1
        self._note_action(state, seat_id, action.type)

        await self._advance_others(state)
        self._snapshot_result(state)

        if self.spec.is_terminal(machine.state):
            final_message = [self._final_message(state)]
            state["final_env_response"] = final_message
            return final_message

        next_seat = self.spec.acting_seat(machine.state)
        if next_seat not in self.policy_seats:
            raise RuntimeError(f"expected a policy seat, got {next_seat!r}")
        state["acting_seat"] = next_seat
        return [self._seat_turn_message(state, next_seat)]

    # -- dataset -------------------------------------------------------------

    def _build_datasets(
        self, num_train_examples: int, num_eval_examples: int
    ) -> tuple[Dataset, Dataset | None]:
        rows = []
        for index in range(num_train_examples + num_eval_examples):
            info = self._game_info(index)
            # Placeholder question: setup_state replaces state["prompt"] with
            # the real first observation once the (possibly LLM-driven)
            # opening turns have been played.
            question = (
                f"Codewords game {info['game_seed']}: you play the "
                f"{POLICY_TEAM} team ({self.role}); {info['starting_team']} "
                "moves first. The board is dealt when the game starts."
            )
            rows.append({"question": question, "answer": "", "info": info})
        dataset = (
            Dataset.from_list(rows[:num_train_examples])
            if num_train_examples > 0
            else None
        )
        eval_dataset = (
            Dataset.from_list(rows[num_train_examples:])
            if num_eval_examples > 0
            else None
        )
        return dataset, eval_dataset


def load_environment(
    role: str = "team",
    opponent: Any = "scripted",
    partner: Any = "scripted",
    guessers_per_team: int = 1,
    deck: str = "classic-en",
    num_train_examples: int = 2000,
    num_eval_examples: int = 200,
    seed: int = 0,
    max_turns: int = 50,
    max_consecutive_says: int = 2,
    cache_endpoint_seats: bool = True,
    render_deliberation: bool = False,
    expose_partner_labels: bool = False,
    endpoint_seat_memory: str = "digest",
    partner_dossiers: dict[str, str] | None = None,
    dossier_rate: float = 0.0,
    false_dossier_rate: float = 0.0,
    **kwargs,
) -> vf.Environment:
    """Load the Codewords environment.

    Args:
        role: which seats the trained policy controls — "team" (cluegiver and
            guessers of the red team, each seat with its own private
            conversation), "cluegiver", or "guesser".
        opponent: driver(s) for the blue team. A driver is "scripted"
            (deterministic baseline), "policy" (the same model/client being
            trained), or an OpenAI-compatible endpoint spec dict:
            {"model": "...", "base_url": "...", "api_key_var": "...",
             "temperature": 0.7, "max_tokens": 400}.
            Accepts a single driver, a role-split dict assigning different
            drivers per role ({"cluegiver": <driver>, "guesser": <driver>}),
            or a list of either — one entry is picked per example,
            deterministically from the board seed, so all GRPO rollouts of an
            example face the same matchup. The chosen models are recorded in
            game_result["seat_models"].
        partner: driver(s) for red seats the policy does not control (only
            relevant for role="cluegiver"/"guesser"); same forms as
            `opponent`.
        guessers_per_team: guessers per team; 1 keeps the simple choose/stop
            action space, 2+ enables the propose/confirm protocol.
        deck: word deck name from turngames (default "classic-en").
        num_train_examples / num_eval_examples: dataset sizes; each example is
            a distinct seeded board.
        seed: base seed for boards and scripted/fallback actors.
        max_turns: cap on policy turns per rollout.
        max_consecutive_says: after this many consecutive "say" actions, a
            seat must commit to a real action ("say" leaves its allowed set);
            0 disables the cap. Structural guard against filibustering.
        cache_endpoint_seats: memoize endpoint-seat responses on (endpoint,
            exact seat history) and deduplicate in-flight identical calls.
            Seat conversations are deterministic in the transcript prefix, so
            rollouts of the same board share adversary turns until the policy
            diverges — a large token saving for GRPO groups and repeated
            eval. Never applies to "policy"-driven seats (weights change).
    """
    return CodewordsEnv(
        role=role,
        opponent=opponent,
        partner=partner,
        guessers_per_team=guessers_per_team,
        deck=deck,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        max_turns=max_turns,
        max_consecutive_says=max_consecutive_says,
        cache_endpoint_seats=cache_endpoint_seats,
        render_deliberation=render_deliberation,
        expose_partner_labels=expose_partner_labels,
        endpoint_seat_memory=endpoint_seat_memory,
        partner_dossiers=partner_dossiers,
        dossier_rate=dossier_rate,
        false_dossier_rate=false_dossier_rate,
        **kwargs,
    )
