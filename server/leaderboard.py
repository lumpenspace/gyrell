"""Aggregate archived replays into a per-model leaderboard.

Replays are the single source of truth for statistics: each JSONL archive
carries the seat lineup (model labels) in its snapshots, the game in its meta
record, and the outcome in its final_score record. Matches are walked in play
order, so Elo ratings are recomputed deterministically from history on every
aggregation — no rating store to migrate or corrupt.

Elo model: each match is team vs team. A team's rating is the mean of its
members' ratings; every member's rating then moves by K * (score - expected)
against the opposing team's mean. Draws (possible in taboo) score 0.5. K is
40 for a model's first 10 matches (provisional), 20 after.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

from turngames.actors.baseline import (
    CLUE_LINES,
    CONFIRM_LINES,
    PROPOSE_LINES,
    REJECT_LINES,
    STOP_LINES,
    STRATEGY_LINES,
)

BUCKETS = ("total", "same_team", "mixed_team", "cluegiver", "guesser", "describer", "player")

# A seat mostly played by the scripted fallback is credited to this pseudo
# player instead of the model wearing the name tag: it stays in the Elo pools
# (team math needs a rating for every seat) but never appears in the output.
FALLBACK_LABEL = "__fallback__"
FALLBACK_SEAT_THRESHOLD = 0.5

# Fingerprints for archives that predate exact fallback accounting: the
# baseline actor only ever speaks these canned lines.
CANNED_LINES = frozenset(
    STRATEGY_LINES + CLUE_LINES + PROPOSE_LINES + CONFIRM_LINES + REJECT_LINES + STOP_LINES
)

# Games that get their own rating pool and standings tab. Unknown game ids
# still count toward overall Elo and the shared buckets.
GAME_IDS = ("codewords", "taboo")

ELO_INITIAL = 1000.0
ELO_PROVISIONAL_GAMES = 10


def _seed_of(records: list[dict]) -> str:
    for record in records:
        if record.get("kind") == "meta":
            return str(record.get("seed", ""))
    return ""


def _chrono_key(seed: str) -> tuple[int, int]:
    """Order matches as they were played. Live seeds are `live-<unix_ts>-<n>`;
    the recorded `live-match-<n>` batch has no timestamp, so it sorts first."""
    nums = [int(n) for n in re.findall(r"\d+", seed)]
    if seed.startswith("live-match"):
        return (0, nums[-1] if nums else 0)
    if len(nums) >= 2:
        return (nums[0], nums[1])
    return (-1, 0)


def _empty_entry() -> dict:
    entry = {bucket: {"games": 0, "wins": 0} for bucket in BUCKETS}
    entry["total"]["draws"] = 0
    entry["per_game"] = {
        game: {"games": 0, "wins": 0, "elo": None} for game in GAME_IDS
    }
    return entry


def _fallback_fractions(records: list[dict], final_record: dict) -> dict[str, float]:
    """Per seat, the fraction of its turns played by the scripted fallback.

    New archives carry exact counts on the final_score record; legacy ones
    are estimated from the baseline actor's canned-line fingerprints."""
    turns = final_record.get("turns") or {}
    fallback_turns = final_record.get("fallback_turns")
    if turns and fallback_turns is not None:
        return {
            seat: fallback_turns.get(seat, 0) / count
            for seat, count in turns.items()
            if count
        }
    canned: dict[str, int] = {}
    total: dict[str, int] = {}
    for record in records:
        if record.get("kind") != "event":
            continue
        event = record.get("event", {})
        if event.get("type") != "actor_responded":
            continue
        payload = event.get("payload", {})
        seat = payload.get("seat_id")
        if not seat:
            continue
        total[seat] = total.get(seat, 0) + 1
        if (payload.get("public_deliberation") or "") in CANNED_LINES:
            canned[seat] = canned.get(seat, 0) + 1
    return {seat: canned.get(seat, 0) / count for seat, count in total.items() if count}


def relabel_fallback_seats(seats: list[dict], fractions: dict[str, float]) -> list[dict]:
    """Credit fallback-dominated seats to the fallback, not the model."""
    relabeled = []
    for seat in seats:
        if fractions.get(seat.get("id"), 0.0) >= FALLBACK_SEAT_THRESHOLD:
            seat = {**seat, "label": FALLBACK_LABEL}
        relabeled.append(seat)
    return relabeled


def _match_result(records: list[dict]) -> tuple[list[dict], str | None, str, dict] | None:
    """Extract (seats-with-labels, winner, game_id, final_record) from one
    completed replay.

    Returns None for incomplete archives (no final_score). A completed match
    with winner None is a draw — it still counts for games, fairness seating,
    and Elo (score 0.5)."""
    seats: list[dict] | None = None
    final_record: dict | None = None
    game = "codewords"  # archives predating the registry carry no game_id
    for record in records:
        kind = record.get("kind")
        if kind == "meta":
            game = str(record.get("game_id") or "codewords")
        elif seats is None and kind == "snapshot":
            view_seats = record.get("view", {}).get("seats")
            if view_seats:
                seats = view_seats
        elif kind == "final_score":
            final_record = record
    if not seats or final_record is None or "score" not in final_record:
        return None
    return seats, final_record["score"].get("winner"), game, final_record


def _iter_replays(replay_dir: Path):
    """Yield the record list of every archived match (plain or gzipped)."""
    if not replay_dir.is_dir():
        return
    for path in sorted((*replay_dir.glob("*.jsonl"), *replay_dir.glob("*.jsonl.gz"))):
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt") as f:
                yield [json.loads(line) for line in f if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue


def matches_played(replay_dir: Path) -> dict[str, int]:
    """How many distinct matches each model (by label) has appeared in — the
    fairness signal for seating: fewer played -> higher priority for a seat."""
    counts: dict[str, int] = {}
    for records in _iter_replays(replay_dir):
        result = _match_result(records)
        if result is None:
            continue
        seats, _, _, _ = result
        for label in {s["label"] for s in seats if s.get("label")}:
            counts[label] = counts.get(label, 0) + 1
    return counts


class _EloPool:
    """One rating pool (overall, or a single game's)."""

    def __init__(self) -> None:
        self.rating: dict[str, float] = {}
        self.games: dict[str, int] = {}

    def _k(self, label: str) -> float:
        return 40.0 if self.games.get(label, 0) < ELO_PROVISIONAL_GAMES else 20.0

    def apply(self, labeled: list[dict], winner: str | None) -> None:
        teams: dict[str, list[str]] = {}
        for seat in labeled:
            teams.setdefault(seat["team_id"], []).append(seat["label"])
        if len(teams) != 2:
            return
        (team_a, members_a), (team_b, members_b) = teams.items()
        mean_a = sum(self.rating.get(m, ELO_INITIAL) for m in members_a) / len(members_a)
        mean_b = sum(self.rating.get(m, ELO_INITIAL) for m in members_b) / len(members_b)
        expected_a = 1.0 / (1.0 + 10.0 ** ((mean_b - mean_a) / 400.0))
        score_a = 0.5 if winner is None else (1.0 if winner == team_a else 0.0)
        # Deltas are computed against pre-match ratings for every member, then
        # applied — a model seated twice just takes the update twice.
        deltas: list[tuple[str, float]] = []
        for member in members_a:
            deltas.append((member, self._k(member) * (score_a - expected_a)))
        for member in members_b:
            deltas.append((member, self._k(member) * ((1.0 - score_a) - (1.0 - expected_a))))
        for member, delta in deltas:
            self.rating[member] = self.rating.get(member, ELO_INITIAL) + delta
        for member in {*members_a, *members_b}:
            self.games[member] = self.games.get(member, 0) + 1


def aggregate_matches(matches: list[tuple[list[dict], str | None, str]]) -> dict:
    """Build the leaderboard from completed matches in play order.

    Each match is (labeled_seats, winner_or_None, game_id). Split out from
    aggregate() so tests can feed synthetic histories without touching disk.
    """
    stats: dict[str, dict] = {}
    overall = _EloPool()
    per_game = {game: _EloPool() for game in GAME_IDS}
    cur_streak: dict[str, int] = {}  # consecutive wins up to the latest match
    best_streak: dict[str, int] = {}

    for labeled, winner, game in matches:
        team_models: dict[str, set[str]] = {}
        for seat in labeled:
            team_models.setdefault(seat["team_id"], set()).add(seat["label"])
        for seat in labeled:
            entry = stats.setdefault(seat["label"], _empty_entry())
            won = winner is not None and seat["team_id"] == winner
            same = len(team_models[seat["team_id"]]) == 1
            for bucket in (
                "total",
                "same_team" if same else "mixed_team",
                seat["role"],
            ):
                stat = entry.setdefault(bucket, {"games": 0, "wins": 0})
                stat["games"] += 1
                stat["wins"] += int(won)
            if winner is None:
                entry["total"]["draws"] += 1
            if game in GAME_IDS:
                game_stat = entry["per_game"][game]
                game_stat["games"] += 1
                game_stat["wins"] += int(won)

        overall.apply(labeled, winner)
        if game in GAME_IDS:
            per_game[game].apply(labeled, winner)

        # Streaks are per model per match (not per seat): a model on the winning
        # team extends its run; anyone else (including draws) resets to 0.
        winners = {
            seat["label"] for seat in labeled if winner and seat["team_id"] == winner
        }
        for label in {seat["label"] for seat in labeled}:
            if label in winners:
                cur_streak[label] = cur_streak.get(label, 0) + 1
                best_streak[label] = max(best_streak.get(label, 0), cur_streak[label])
            else:
                cur_streak[label] = 0

    rows = []
    for model, entry in stats.items():
        for game in GAME_IDS:
            pool = per_game[game]
            if entry["per_game"][game]["games"]:
                entry["per_game"][game]["elo"] = round(pool.rating.get(model, ELO_INITIAL))
        rows.append(
            {
                "model": model,
                **entry,
                "elo": round(overall.rating.get(model, ELO_INITIAL)),
                "provisional": overall.games.get(model, 0) < ELO_PROVISIONAL_GAMES,
                "streak": cur_streak.get(model, 0),
                "best_streak": best_streak.get(model, 0),
            }
        )
    rows = [row for row in rows if row["model"] != FALLBACK_LABEL]
    rows.sort(key=lambda row: (-row["elo"], -row["total"]["wins"], row["model"]))
    return {"matches": len(matches), "models": rows}


def aggregate(replay_dir: Path) -> dict:
    played: list[tuple[tuple[int, int], list[dict], str | None, str]] = []
    for records in _iter_replays(replay_dir):
        result = _match_result(records)
        if result is None:
            continue
        seats, winner, game, final_record = result
        fractions = _fallback_fractions(records, final_record)
        labeled = relabel_fallback_seats(
            [seat for seat in seats if seat.get("label")], fractions
        )
        if labeled:
            played.append((_chrono_key(_seed_of(records)), labeled, winner, game))
    played.sort(key=lambda match: match[0])
    return aggregate_matches([(labeled, winner, game) for _, labeled, winner, game in played])
