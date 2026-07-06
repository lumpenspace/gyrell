from __future__ import annotations

import re
from dataclasses import dataclass

# One shared definition of a "timer word" so counting and clipping never drift.
_WORD_RE = re.compile(r"[A-Za-z0-9_']+")
ELLIPSIS = "…"


def count_time_words(text: str | None) -> int:
    """Count timer words in public deliberation.

    The first timer model is intentionally simple: every alphanumeric word in
    the actor's public deliberation costs one time unit.
    """

    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def clip_to_word_budget(text: str | None, budget: int) -> str:
    """Truncate `text` to its first `budget` timer-words, trailing off with an
    ellipsis when anything was cut.

    Used to cut a speaker off exactly where their word clock runs out: the
    clipped text costs precisely `budget` (the ellipsis is free), so it spends
    the last of the clock rather than being spoken in full and then discarded.
    Text that already fits the budget is returned unchanged; a speaker with no
    clock left (`budget <= 0`) says nothing.
    """

    if not text:
        return text or ""
    if budget <= 0:
        return ""
    matches = list(_WORD_RE.finditer(text))
    if len(matches) <= budget:
        return text
    head = text[: matches[budget - 1].end()].rstrip()
    # drop a dangling clause separator so the ellipsis reads as a clean cut-off
    head = re.sub(r"[\s,;:.!?—-]+$", "", head)
    return f"{head}{ELLIPSIS}"


@dataclass(frozen=True)
class TimeSpend:
    before: int | None
    spent: int
    after: int | None
    expired: bool = False

