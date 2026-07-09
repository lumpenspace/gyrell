"""Prompt building + response sanitization for Taboo seats.

Plugs into OpenRouterActor as its `prompter`. Sanitization only enforces
shape (allowed action type, one-word guesses); it deliberately does NOT
pre-screen descriptions against the card — walking into a buzz is part of
the game, and the arbiter rules on what was actually said.
"""

from __future__ import annotations

import json

from turngames.actors.llm import (
    RESPONSE_FORMAT,
    THINKING_OFFER,
    clip_note,
    extract_json,
    render_memory,
    unwrap_action,
)
from turngames.core import Action, ActorInput, ActorOutput
from turngames.core.timer import count_time_words

RULES_COMMON = (
    "You are playing Taboo on a broadcast stage. Two teams (red, blue) "
    "alternate rounds, and the card rotates: each round one player on the "
    "describing team holds it and talks teammates into saying the TARGET "
    "word. THE CARD ONLY BINDS ITS HOLDER: if the describer says the target, "
    "any part or form of it, or any forbidden word (or a form of one), "
    "that's a BUZZ — card lost, point to the other team, and YOUR TEAM'S "
    "ROUND ENDS on the spot (the floor crosses over). Clueing the word's "
    "FORM is also a buzz: no letter counts, no 'starts with', no 'rhymes "
    "with', no initials or spelling — describe the MEANING. Guessers are "
    "free: forbidden words don't apply to them, and if the target crosses a "
    "guesser's lips it scores, even mid-sentence. Everything said in "
    "public_deliberation is broadcast and costs one word from the speaker's "
    "personal clock; the round ends when the DESCRIBER's clock hits 0. Most "
    "points after all rounds wins."
)


def build_taboo_messages(
    actor_input: ActorInput,
    notes: tuple[str, ...] = (),
    offer_thinking: bool = False,
) -> list[dict]:
    seat = actor_input.seat
    data = actor_input.observation.data
    allowed = list(actor_input.observation.allowed_actions)

    describing = data.get("describer") == seat.id
    if describing:
        role_line = (
            f"You are seat {seat.id} on the {seat.team_id} team, and THIS "
            "round the card is yours: you are the describer. Your "
            "public_deliberation IS your description — choose every word "
            "carefully. Action 'describe' hands the floor to a guesser; "
            "'skip' burns the card unscored. The round ends when YOUR word "
            "clock runs dry."
        )
    else:
        role_line = (
            f"You are seat {seat.id} on the {seat.team_id} team, guessing "
            "this round (the card rotates — you'll describe another round). "
            "You never see the card, and the forbidden words do NOT apply to "
            "you — only the describer is restricted, so say whatever helps. "
            "If the target word crosses your lips it counts, even "
            "mid-sentence — think out loud freely and use action 'guess' "
            "(exactly one word) only to be deliberate, or 'pass' the floor "
            "back. Keep spoken thoughts short, they cost your personal clock."
        )

    state_lines = [
        f"Round {data.get('round_number')} of {data.get('total_rounds')} — "
        f"describing team: {data.get('current_team')}.",
        f"Points: {data.get('points')}. Violations: {data.get('violations')}.",
        f"Card #{(data.get('cards_played') or 0) + 1} is in play. Every "
        "earlier card is DEAD (guessed, buzzed or skipped) — table talk "
        "about it is history, not a lead on this card.",
    ]
    card = data.get("current_card")
    if card:
        state_lines.append(
            f"CURRENT CARD — target: {str(card.get('target', '')).upper()}; "
            f"forbidden: {', '.join(card.get('forbidden', ()))}. Never say the "
            "target, part of it, or any forbidden word."
        )
    hints = data.get("current_hints") or []
    if hints:
        state_lines.append("Hints for THIS card so far: " + " | ".join(hints))
    elif not describing:
        state_lines.append(
            "No hints yet for this card — wait for the describer before "
            "committing to a guess."
        )
    wrong = data.get("wrong_guesses_this_card") or []
    if wrong:
        state_lines.append(
            "Already guessed on THIS card, all wrong — do NOT repeat: "
            + ", ".join(wrong)
        )
    if data.get("time_left") is not None:
        state_lines.append(
            f"WORD CLOCK: {data['time_left']} of {data.get('time_limit')} words "
            "left this round. Every spoken word costs one."
        )
    skips_left = None
    if data.get("max_skips_per_round") is not None:
        skips_left = data["max_skips_per_round"] - (data.get("skips_this_round") or 0)
        state_lines.append(f"Skips left this round: {skips_left}.")
    if offer_thinking:
        state_lines.append(THINKING_OFFER)

    user = "\n".join(
        [
            role_line,
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


def sanitize_taboo(raw: str, actor_input: ActorInput) -> ActorOutput | None:
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
    if action_type == "guess":
        word = str(action_data.get("word", "")).strip()
        if count_time_words(word) != 1:
            return None
        return ActorOutput(
            public_deliberation=deliberation,
            action=Action.of("guess", word=word),
            raw_response=raw,
            notes=note,
        )
    # describe / skip / pass carry no payload.
    return ActorOutput(
        public_deliberation=deliberation,
        action=Action.of(action_type),
        raw_response=raw,
        notes=note,
    )


class TabooPrompter:
    def build_messages(
        self,
        actor_input: ActorInput,
        notes: tuple[str, ...] = (),
        offer_thinking: bool = False,
    ) -> list[dict]:
        return build_taboo_messages(actor_input, notes, offer_thinking=offer_thinking)

    def sanitize(self, raw: str, actor_input: ActorInput) -> ActorOutput | None:
        return sanitize_taboo(raw, actor_input)
