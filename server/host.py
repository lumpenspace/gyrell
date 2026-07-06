"""LLM color commentator for the live broadcast.

The host is a broadcast-only role (never a game seat — see spec.md): it sees
only audience-visible state and its lines are streamed to spectators between
turns and at match end. Backed by an OpenRouter model (Grok by default,
override with HOST_MODEL) with the channel's canned lines as fallback so the
show never stalls on a slow or misbehaving model.
"""

from __future__ import annotations

import json
import os
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_HOST_MODEL = "x-ai/grok-2"

DEFAULT_GAME_BLURB = "a Codenames-style word game"

SYSTEM_PROMPT_TEMPLATE = (
    "You are the host of LUDICA, a live broadcast where AI language models "
    "play {game_blurb} against each other. You are witty, "
    "quick, and a little mischievous — sports commentator meets late-night "
    "monologue. Depending on the moment you might introduce the lineup, recap "
    "the score at the top of a round, announce the colour of a card that was "
    "just revealed, call a foul, or crown the winner. React to what just "
    "happened and tee up what's next; feel free to rib the models by name. "
    "Rules: reply with ONE line of commentary, at most two short sentences, "
    "plain text only, no quotes around it, at most one emoji. You may talk "
    "about cards/words that have ALREADY been revealed or resolved, but never "
    "speculate about ones still hidden or in play."
)


class LLMHost:
    """Generates host lines; every call degrades to the given fallback."""

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        timeout: float = 20.0,
        game_blurb: str = DEFAULT_GAME_BLURB,
    ) -> None:
        self.api_key = api_key
        self.model = model or os.environ.get("HOST_MODEL", DEFAULT_HOST_MODEL)
        self.timeout = timeout
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(game_blurb=game_blurb)
        self.last_error: str | None = None

    def line(self, moment: str, context: dict, fallback: str) -> str:
        try:
            text = " ".join(self._chat(moment, context).split())
            if text:
                self.last_error = None
                return text[:280]
            self.last_error = "empty response"
        except Exception as error:  # network, HTTP, timeout — keep the show going
            self.last_error = repr(error)
        return fallback

    def maybe_line(self, moment: str, context: dict) -> str | None:
        """Optional commentary: the host may decline. Returns None when it has
        nothing worth saying (an explicit PASS, an empty reply, or any error) so
        the caller can simply stay silent."""
        try:
            text = " ".join(self._chat(moment, context).split())
            self.last_error = None
            if not text or text.strip(" .!").upper() in {"PASS", "SKIP", "NOTHING"}:
                return None
            return text[:280]
        except Exception as error:  # never let the optional beat break the show
            self.last_error = repr(error)
            return None

    def _chat(self, moment: str, context: dict) -> str:
        user = (
            f"Moment: {moment}\n"
            f"Audience-visible match state:\n{json.dumps(context)}\n"
            "Your one-line commentary:"
        )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 90,
            }
        ).encode()
        request = urllib.request.Request(
            OPENROUTER_URL,
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
        return str(payload["choices"][0]["message"]["content"] or "").strip()
