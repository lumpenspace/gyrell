"""LLM actor backed by OpenRouter's OpenAI-compatible chat completions API.

One actor instance per seat, each configured with an OpenRouter model slug so
different labs' models can fill different seats in the same match. Uses only
the standard library (urllib); call `act` from a worker thread when driving
an asyncio loop.

Every response is sanitized against the current observation; anything that
can't be repaired falls back to the seat's BaselineCodewordsActor so a match
never stalls on a misbehaving model.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace

from turngames.core import Action, ActorInput, ActorOutput

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Per-slug `reasoning` overrides for OpenRouter. DeepSeek V3.1 and GLM-4.6
# are hybrid thinkers with no budget knob: OpenRouter converts a token cap
# into "thinking: on, unbounded" and turns crawl for minutes — so both play
# with thinking off. Shared by the live channel and the tournament runner.
REASONING_OVERRIDES: dict[str, dict] = {
    "deepseek/deepseek-chat-v3.1": {"enabled": False},
    "z-ai/glm-4.6": {"enabled": False},
}

RULES_COMMON = (
    "You are playing a Codenames-style word game on a 5x5 board. Two teams "
    "(red, blue) race to have their guessers find all their team's words. "
    "Revealing the assassin word loses instantly. Words you say out loud in "
    "public_deliberation are broadcast to the audience AND count against your "
    "team's word clock when one is active, so be brief and entertaining."
)

RESPONSE_FORMAT = (
    'Respond with ONLY a JSON object, no markdown fences, of the form: '
    '{"public_deliberation": "<1-2 short sentences spoken aloud>", '
    '"action": {"type": "<one of the allowed action types>", ...action fields '
    'at this same level — do NOT nest them under "schema"}, '
    '"notes": "<OPTIONAL one-line private note to your future self — your plan, '
    "candidates you're weighing, what to avoid. Not spoken, costs no clock. "
    'Keep it under 40 words>"}'
)

# Per-seat memory limits: how much history a prompt carries.
TRANSCRIPT_LINES = 40
NOTES_KEPT = 5
NOTES_MAX_CHARS = 400

# One-shot scratchpad: each seat gets exactly ONE private planning beat — its
# first act of the match (native thinkers already plan before their content;
# they can ignore this). Repeating the offer on every opening-turn exchange
# made round one crawl; once per seat keeps the show moving.
THINKING_OFFER = (
    "THIS TURN ONLY (the offer will not repeat): before the JSON you may plan "
    "privately inside <thinking>...</thinking> — shortlist candidates, note "
    "traps. HARD LIMIT 40 words. Nobody ever sees it and it costs no clock."
)
THINKING_MAX_CHARS = 1200


def extract_thinking(text: str) -> str | None:
    """The model's <thinking> scratchpad, when it took the opening-turn offer.
    Routed into ActorOutput.reasoning so tag-based thinking and native hidden
    reasoning land in the same diagnostic sidecar (never broadcast)."""
    match = re.search(r"<think(?:ing)?>(.*?)</think(?:ing)?>", text, flags=re.DOTALL)
    if not match:
        return None
    body = match.group(1).strip()
    return body[:THINKING_MAX_CHARS] or None


def render_memory(actor_input: ActorInput, notes: tuple[str, ...]) -> list[str]:
    """The two memory sections shared by every game's prompter: recent public
    table talk, and the seat's own private notes from earlier turns."""
    lines: list[str] = []
    if actor_input.transcript:
        recent = actor_input.transcript[-TRANSCRIPT_LINES:]
        lines += ["Table talk so far (public, oldest first):", *recent, ""]
    if notes:
        lines += [
            "Your private notes from earlier turns (only you see these; "
            "trust the CURRENT board state over anything stale in them):",
            *[f"- {note}" for note in notes],
            "",
        ]
    return lines


def render_board_for_prompt(data: dict) -> str:
    cells = data["board"]["cells"]
    lines = []
    for cell in cells:
        word = cell["word"]
        revealed = cell.get("revealed")
        key = cell.get("key")
        if revealed is not None:
            lines.append(f"- {word}: REVEALED ({revealed})")
        elif key is not None:
            lines.append(f"- {word}: hidden, key={key}")
        else:
            lines.append(f"- {word}: hidden")
    return "\n".join(lines)


def build_messages(
    actor_input: ActorInput,
    notes: tuple[str, ...] = (),
    offer_thinking: bool = False,
) -> list[dict]:
    seat = actor_input.seat
    data = actor_input.observation.data
    allowed = list(actor_input.observation.allowed_actions)

    role_line = (
        f"You are seat {seat.id}: the {seat.team_id} team's "
        f"{'cluegiver (you see the key)' if seat.role == 'cluegiver' else 'guesser (no key)'}. "
        f"{actor_input.role.description}"
    )
    state_lines = [
        f"Turn {data.get('turn_number')} — acting team: {data.get('current_team')}.",
        f"Current clue: {data.get('current_clue')}",
        f"Pending proposal: {data.get('pending_proposal')}",
        f"Guesses made this turn: {data.get('guesses_this_turn')}",
    ]
    if data.get("time_left") is not None:
        limit = data.get("time_limit")
        budget = (
            f"{data['time_left']} of {limit}" if limit else f"{data['time_left']}"
        )
        state_lines.append(
            f"WORD CLOCK: {budget} words left for your team this turn. Every "
            "word anyone on your team says out loud costs one. When it hits 0 "
            "your team is forced into a single final guess — budget your speech."
        )
    if any(a.get("type") == "give_clue" for a in allowed):
        # The engine enforces this and smaller models walk into it unless
        # the rule is spelled out at the moment of the clue.
        state_lines.append(
            "CLUE LEGALITY: your clue must be ONE word that is NOT on the "
            "board — it may not match, contain, or sit inside any unrevealed "
            "board word (e.g. you cannot clue SEAL while SEAL is unrevealed)."
        )
    if offer_thinking:
        state_lines.append(THINKING_OFFER)
    if actor_input.constraints.get("final_guess_one_word"):
        state_lines.append(
            "TIMER EXPIRED: you must respond with action type 'choose' and say "
            "nothing (empty public_deliberation)."
        )
    if actor_input.constraints.get("bonus_guess_two_words"):
        state_lines.append(
            "BONUS GUESS: you cleared the clue. Snap decision — action "
            "'choose' with one more board word, or 'stop' to bank the points. "
            "Say at most TWO words aloud; anything longer is cut off."
        )

    user = "\n".join(
        [
            role_line,
            "",
            "Board:",
            render_board_for_prompt(data),
            "",
            *state_lines,
            "",
            *render_memory(actor_input, notes),
            f"Allowed actions right now: {json.dumps(allowed)}",
            RESPONSE_FORMAT,
        ]
    )
    return [
        {"role": "system", "content": RULES_COMMON},
        {"role": "user", "content": user},
    ]


def unwrap_action(action_data: dict) -> dict:
    """Models sometimes echo the allowed-action envelope verbatim, nesting
    the payload under "schema" ({"type": "choose", "schema": {"word": ...}}).
    The intent is unambiguous — lift the nested fields up (explicit top-level
    keys win)."""
    nested = action_data.get("schema")
    if isinstance(nested, dict):
        action_data = {**nested, **{k: v for k, v in action_data.items() if k != "schema"}}
    return action_data


def clip_note(value: object) -> str | None:
    """Sanitize a model's optional note-to-self field."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:NOTES_MAX_CHARS]


def extract_json(text: str) -> dict | None:
    # Thinking models wrap reasoning in XML-ish tags whose body may contain
    # braces; strip closed tags (and anything before a dangling close-tag)
    # so the brace search finds the actual JSON action.
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*?</think(?:ing)?>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class CodewordsPrompter:
    """Prompt building + response sanitization for Codewords seats.

    The default when OpenRouterActor gets no prompter; other games plug in
    their own object with the same two methods (see turngames.actors.llm_taboo).
    """

    def build_messages(
        self,
        actor_input: ActorInput,
        notes: tuple[str, ...] = (),
        offer_thinking: bool = False,
    ) -> list[dict]:
        return build_messages(actor_input, notes, offer_thinking=offer_thinking)

    def sanitize(self, raw: str, actor_input: ActorInput) -> ActorOutput | None:
        return OpenRouterActor._parse(raw, actor_input)


@dataclass
class OpenRouterActor:
    """One seat driven by an OpenRouter-served model, with a safe fallback.

    `fallback` is any actor with the same act() interface (baseline for the
    seat's game); `prompter` localizes prompts/sanitization to the game and
    defaults to Codewords.
    """

    model: str
    api_key: str
    fallback: object
    # Any OpenAI-compatible chat-completions endpoint; defaults to OpenRouter.
    # Point it at a local shim (e.g. the Codex SDK bridge) to drive a seat
    # from a subscription-backed model instead of a per-token API.
    base_url: str = OPENROUTER_URL
    # Wall-clock cap per turn — the show's pacing depends on it.
    timeout: float = 45.0
    # Total completion budget. Reasoning models spend from this before any
    # content appears, so it must sit far above the reasoning cap: laguna
    # ignores the cap entirely (and `effort`) — at 400 it produced zero
    # content on every turn and silently handed whole matches to its
    # fallback. Non-reasoning models only use what they need, so the
    # ceiling mainly bounds worst-case latency.
    max_tokens: int = 6000
    # Cap on hidden thinking (OpenRouter's unified `reasoning` parameter,
    # normalized per provider; ignored by non-reasoning models). Brutally
    # tight: this is a fast-paced party game, and at 512 the thinky seats
    # made every turn drag. Actual spend per turn lands in the traces
    # sidecar as `reasoning_tokens` — check there before raising it.
    reasoning_max_tokens: int | None = 128
    # Verbatim replacement for the `reasoning` request field, for models the
    # token cap can't govern. Hybrid thinkers with only an on/off switch
    # (e.g. deepseek-chat-v3.1) treat any budget as "on, unbounded" — pass
    # {"enabled": False} to keep them in non-thinking mode.
    reasoning_override: dict | None = None
    prompter: object = field(default_factory=CodewordsPrompter)
    last_error: str | None = field(default=None, init=False)
    # Hidden-thinking spend reported by the provider for the last call, when
    # available — the audit trail behind the reasoning cap.
    last_reasoning_tokens: int | None = field(default=None, init=False)
    # This seat's private notes-to-self, model-authored, one per past turn.
    _notes: list[str] = field(default_factory=list, init=False)
    _acts: int = field(default=0, init=False)

    def act(self, actor_input: ActorInput) -> ActorOutput:
        reasoning: str | None = None
        raw: str | None = None
        # The <thinking> scratchpad is offered exactly once per seat per
        # match — the seat's first act — so it can't slow later exchanges.
        offer_thinking = self._acts == 0
        self._acts += 1
        try:
            messages = self.prompter.build_messages(
                actor_input, notes=tuple(self._notes), offer_thinking=offer_thinking
            )
            raw, reasoning = self._chat(messages)
            if reasoning is None:
                # No native trace — a <thinking> scratchpad counts as one.
                reasoning = extract_thinking(raw)
            output = self.prompter.sanitize(raw, actor_input)
            if output is not None:
                self.last_error = None
                if output.notes:
                    self._notes.append(output.notes)
                    del self._notes[:-NOTES_KEPT]
                return replace(output, reasoning=reasoning)
            self.last_error = "unparseable response"
        except urllib.error.HTTPError as error:
            # Read the body — "HTTP 400" alone is undiagnosable; the payload
            # names the provider and the rejected parameter.
            try:
                detail = error.read().decode("utf-8", "replace")[:300]
            except Exception:
                detail = ""
            self.last_error = f"HTTP {error.code}: {detail}"
        except Exception as error:  # network, timeout — keep the show going
            self.last_error = repr(error)
        fallback_output = self.fallback.act(actor_input)
        # Keep whatever we did capture: the thinking trace and the raw
        # response are exactly the evidence that explains a fallback turn.
        return replace(fallback_output, reasoning=reasoning, raw_response=raw)

    # -- OpenRouter call -----------------------------------------------------

    def _chat(self, messages: list[dict]) -> tuple[str, str | None]:
        """Returns (content, reasoning). Reasoning is the model's hidden
        thinking when the provider exposes it, else None."""
        request_body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if self.reasoning_override:
            request_body["reasoning"] = self.reasoning_override
        elif self.reasoning_max_tokens:
            request_body["reasoning"] = {"max_tokens": self.reasoning_max_tokens}
        body = json.dumps(request_body).encode()
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/lumpenspace/codenames-llms",
                "X-Title": "codenames-llms",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        usage = payload.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        self.last_reasoning_tokens = details.get("reasoning_tokens")
        if not payload.get("choices"):
            # OpenRouter reports some provider failures as a 200 with an
            # error body — surface it instead of a bare KeyError.
            raise RuntimeError(f"no choices in response: {json.dumps(payload)[:300]}")
        message = payload["choices"][0]["message"]
        return message.get("content") or "", message.get("reasoning") or None

    # -- response sanitization -------------------------------------------------

    @staticmethod
    def _parse(raw: str, actor_input: ActorInput) -> ActorOutput | None:
        parsed = extract_json(raw)
        if parsed is None:
            return None
        action_data = parsed.get("action")
        if not isinstance(action_data, dict) or "type" not in action_data:
            return None
        action_data = unwrap_action(action_data)

        allowed_types = {a["type"] for a in actor_input.observation.allowed_actions}
        action_type = str(action_data["type"])
        if action_type not in allowed_types:
            return None

        deliberation = str(parsed.get("public_deliberation") or "")
        note = clip_note(parsed.get("notes"))
        data = actor_input.observation.data
        unrevealed = {
            c["word"].casefold(): c["word"]
            for c in data["board"]["cells"]
            if c["revealed"] is None
        }

        if action_type in ("propose", "choose"):
            word = str(action_data.get("word", "")).strip().casefold()
            if word not in unrevealed:
                return None
            if actor_input.constraints.get("final_guess_one_word"):
                deliberation = ""
            return ActorOutput(
                public_deliberation=deliberation,
                action=Action.of(action_type, word=unrevealed[word]),
                raw_response=raw,
                notes=note,
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
                raw_response=raw,
                notes=note,
            )

        # say / confirm / reject / stop carry no payload worth sanitizing.
        return ActorOutput(
            public_deliberation=deliberation,
            action=Action.of(action_type),
            raw_response=raw,
            notes=note,
        )
