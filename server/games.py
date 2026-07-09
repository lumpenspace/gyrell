"""Searchable index of archived matches.

Replays are the single source of truth (same doctrine as server.leaderboard):
each archive is summarized once into a compact card — lineup, winner, score,
and a set of derived "notable moment" tags — and the summaries are cached by
file mtime so repeated queries don't re-read gigabytes of gzip.

Moment tags are deliberately coarse: they exist so a spectator can ask for
"assassin endings" or "comebacks", not to be a stats engine.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from server.leaderboard import _chrono_key, _seed_of

# tag -> human blurb, also serves as the canonical tag list for the client.
# Deliberately few: blowout covers every flavor of domination (sweeps,
# flawless routs), photo_finish covers draws.
MOMENTS = {
    "assassin": "the match ended on the assassin",
    "blowout": "one side dominated — sweep, rout, or flawless win",
    "comeback": "the winner trailed badly before turning it around",
    "photo_finish": "decided by the narrowest margin (or not at all)",
    "marathon": "an unusually long match",
}

BLOWOUT_TABOO_MARGIN = 10
PHOTO_FINISH_TABOO_MARGIN = 1
COMEBACK_TABOO_DEFICIT = 4
COMEBACK_CODEWORDS_DEFICIT = 3
SWEEP_REMAINING = 4
FLAWLESS_TABOO_POINTS = 15
MARATHON_EVENTS = 450

# Scripted stand-ins (canned-line fallback lineups) aren't worth searching
# for — matches featuring them are hidden from the index entirely.
SCRIPTED_PREFIXES = ("baseline-", "scripted-")

# path -> (mtime, summary-or-None). None caches "unreadable/incomplete".
_cache: dict[Path, tuple[float, dict | None]] = {}


def _team_labels(seats: list[dict]) -> dict[str, list[str]]:
    teams: dict[str, list[str]] = {}
    for seat in seats:
        label = seat.get("label")
        if label:
            teams.setdefault(seat["team_id"], []).append(label)
    return teams


def _codewords_moments(records: list[dict], score: dict, winner: str | None) -> set[str]:
    tags: set[str] = set()
    if score.get("terminal_reason") == "assassin_revealed":
        tags.add("assassin")
    remaining = {t: score.get(f"{t}_remaining") for t in ("red", "blue")}
    if winner and remaining.get(winner) == 0:
        loser = "blue" if winner == "red" else "red"
        if (remaining.get(loser) or 0) >= SWEEP_REMAINING:
            tags.add("blowout")
    if winner and (remaining.get("blue" if winner == "red" else "red") or 0) <= 1:
        if score.get("terminal_reason") == "all_team_words_revealed":
            tags.add("photo_finish")
    # Comeback: walk reveals and see whether the eventual winner was ever
    # trailing by several of its own words.
    if winner:
        found = {"red": 0, "blue": 0}
        for record in records:
            if record.get("kind") != "event":
                continue
            event = record["event"]
            if event.get("type") != "word_revealed":
                continue
            assignment = event["payload"].get("assignment")
            if assignment in found:
                found[assignment] += 1
                loser = "blue" if winner == "red" else "red"
                if found[loser] - found[winner] >= COMEBACK_CODEWORDS_DEFICIT:
                    tags.add("comeback")
    return tags


def _taboo_moments(records: list[dict], score: dict, winner: str | None) -> set[str]:
    tags: set[str] = set()
    points = {t: score.get(f"{t}_points") or 0 for t in ("red", "blue")}
    margin = abs(points["red"] - points["blue"])
    if winner and margin >= BLOWOUT_TABOO_MARGIN:
        tags.add("blowout")
    if winner and margin <= PHOTO_FINISH_TABOO_MARGIN:
        tags.add("photo_finish")
    violations = score.get("violations") or {}
    if (
        winner
        and points[winner] >= FLAWLESS_TABOO_POINTS
        and not violations.get(winner)
    ):
        tags.add("blowout")
    if winner:
        loser = "blue" if winner == "red" else "red"
        for record in records:
            if record.get("kind") != "event":
                continue
            event = record["event"]
            if event.get("type") != "round_ended":
                continue
            round_points = event["payload"].get("points") or {}
            deficit = (round_points.get(loser) or 0) - (round_points.get(winner) or 0)
            if deficit >= COMEBACK_TABOO_DEFICIT:
                tags.add("comeback")
    return tags


def _match_words(records: list[dict]) -> list[str]:
    """The words a match was played on: the codewords board, or every taboo
    card that hit the table (guessed, skipped, or buzzed)."""
    words: set[str] = set()
    for record in records:
        kind = record.get("kind")
        if kind == "snapshot":
            cells = record.get("view", {}).get("board", {}).get("cells") or ()
            for cell in cells:
                if cell.get("word"):
                    words.add(str(cell["word"]).lower())
            if words:
                break  # a codewords board is fixed — first snapshot has it all
        elif kind == "event":
            event = record["event"]
            if event.get("type") in ("card_guessed", "card_skipped", "taboo_violation"):
                target = (event["payload"].get("card") or {}).get("target")
                if target:
                    words.add(str(target).lower())
    return sorted(words)


def summarize(records: list[dict], name: str) -> dict | None:
    """One match card, or None for an incomplete archive."""
    seats = None
    final = None
    game = "codewords"
    kind = "match"
    for record in records:
        record_kind = record.get("kind")
        if record_kind == "meta":
            game = str(record.get("game_id") or "codewords")
            if record.get("match_kind") == "duel":
                kind = "duel"
        elif seats is None and record_kind == "snapshot":
            view_seats = record.get("view", {}).get("seats")
            if view_seats:
                seats = view_seats
        elif record_kind == "final_score":
            final = record
    if not seats or final is None or "score" not in final:
        return None
    score = dict(final["score"])
    winner = score.pop("winner", None)
    event_count = score.pop("event_count", 0)
    teams = _team_labels(seats)
    if any(
        label.startswith(SCRIPTED_PREFIXES)
        for members in teams.values()
        for label in members
    ):
        return None

    moments = {"photo_finish"} if winner is None else set()
    if game == "taboo":
        moments |= _taboo_moments(records, score, winner)
    else:
        moments |= _codewords_moments(records, score, winner)
    if event_count >= MARATHON_EVENTS:
        moments.add("marathon")

    return {
        "name": name,
        "game": game,
        "kind": kind,
        "order": _chrono_key(_seed_of(records)),
        "teams": teams,
        "same_team_models": sorted(
            {members[0] for members in teams.values() if len(set(members)) == 1}
        ),
        "winner": winner,
        "score": score,
        "moments": sorted(moments & set(MOMENTS)),
        "words": _match_words(records),
    }


def _load(path: Path) -> dict | None:
    mtime = path.stat().st_mtime
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    opener = gzip.open if path.suffix == ".gz" else open
    name = path.name.removesuffix(".gz").removesuffix(".jsonl")
    try:
        with opener(path, "rt") as f:
            summary = summarize([json.loads(line) for line in f if line.strip()], name)
    except (OSError, json.JSONDecodeError):
        summary = None
    _cache[path] = (mtime, summary)
    return summary


def search(
    replay_dir: Path,
    *,
    model: str | None = None,
    game: str | None = None,
    kind: str | None = None,
    winner_model: str | None = None,
    team: str | None = None,  # "same" | "mixed", composition of `model`'s team
    moment: str | None = None,
    word: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """Filter the archive. All filters AND together; `team` and
    `winner_model` are interpreted relative to `model` when one is given."""
    cards = []
    if replay_dir.is_dir():
        for path in (*replay_dir.glob("*.jsonl"), *replay_dir.glob("*.jsonl.gz")):
            card = _load(path)
            if card:
                cards.append(card)
    # Newest first; the seed's timestamp is the play order.
    cards.sort(key=lambda c: c["order"], reverse=True)

    def keep(card: dict) -> bool:
        if game and card["game"] != game:
            return False
        if kind and card["kind"] != kind:
            return False
        if moment and moment not in card["moments"]:
            return False
        if word and word.strip().lower() not in card["words"]:
            return False
        if model:
            sides = [t for t, members in card["teams"].items() if model in members]
            if not sides:
                return False
            if team == "same" and model not in card["same_team_models"]:
                return False
            if team == "mixed" and set(sides) <= {
                t
                for t, members in card["teams"].items()
                if len(set(members)) == 1
            }:
                return False
            if winner_model == model and card["winner"] not in sides:
                return False
        if winner_model and winner_model != model:
            winning = card["teams"].get(card["winner"], [])
            if winner_model not in winning:
                return False
        return True

    kept = [card for card in cards if keep(card)]
    models = sorted({m for card in cards for members in card["teams"].values() for m in members})
    # The whole archive's vocabulary feeds the search box's autocomplete.
    words = sorted({w for card in cards for w in card["words"]})
    start = max(0, offset)
    page = kept[start : start + max(1, min(limit, 100))]
    return {
        "total": len(kept),
        "offset": start,
        "games": [
            {key: value for key, value in card.items() if key not in ("order", "words")}
            for card in page
        ],
        "models": models,
        "moments": MOMENTS,
        "words": words,
    }
