"""Head-to-head tournament ladder, aggregated from archived duel replays.

Every few matches the live channel runs a *duel*: three copies of one model on
red versus three copies of another on blue (see server.channel). Those matches
are tagged `match_kind: "duel"` in their meta record, which is the single source
of truth here — the ladder is recomputed from the replay archive on demand, the
same way the leaderboard is (see server.leaderboard), so there is no standings
store to keep in sync.
"""

from __future__ import annotations

from pathlib import Path

from server.leaderboard import (
    FALLBACK_LABEL,
    _chrono_key,
    _iter_replays,
    _match_result,
    _seed_of,
)

# How many most-recent duels the ladder surfaces for the "recent results" strip.
RECENT_DUELS = 8


def _iter_duels(replay_dir: Path):
    """Yield one dict per completed duel replay, in play order.

    A duel is a match tagged `match_kind: "duel"` whose two teams are each a
    single distinct model. Anything else (a normal mixed match, an untagged
    legacy archive, an incomplete or malformed duel) is skipped."""
    duels = []
    for records in _iter_replays(replay_dir):
        meta = next((r for r in records if r.get("kind") == "meta"), None)
        if not meta or meta.get("match_kind") != "duel":
            continue
        result = _match_result(records)
        if result is None:
            continue
        seats, winner, game, _ = result
        team_models: dict[str, set[str]] = {}
        for seat in seats:
            label = seat.get("label")
            if label:
                team_models.setdefault(seat["team_id"], set()).add(label)
        # A genuine duel: exactly two teams, each a single model, and neither
        # side degenerated into the scripted fallback.
        if len(team_models) != 2 or any(len(m) != 1 for m in team_models.values()):
            continue
        red = next(iter(team_models.get("red", set())), None)
        blue = next(iter(team_models.get("blue", set())), None)
        if not red or not blue or FALLBACK_LABEL in (red, blue):
            continue
        winner_model = None
        if winner == "red":
            winner_model = red
        elif winner == "blue":
            winner_model = blue
        duels.append(
            {
                "key": _chrono_key(_seed_of(records)),
                "a": red,
                "b": blue,
                "winner": winner_model,
                "game": game,
            }
        )
    duels.sort(key=lambda d: d["key"])
    return duels


def duel_counts(replay_dir: Path) -> dict[str, int]:
    """How many duels each model (by label) has fought — the fairness signal
    the channel uses to seed the next pairing toward the least-tested models."""
    counts: dict[str, int] = {}
    for duel in _iter_duels(replay_dir):
        for model in (duel["a"], duel["b"]):
            counts[model] = counts.get(model, 0) + 1
    return counts


def duel_standings(replay_dir: Path) -> dict:
    """The tournament ladder: per-model duel win/loss records plus the last few
    results, for the interstitial chart. Sorted by wins, then fewest losses."""
    duels = _iter_duels(replay_dir)
    stats: dict[str, dict] = {}

    def row(model: str) -> dict:
        return stats.setdefault(
            model, {"model": model, "wins": 0, "losses": 0, "draws": 0, "played": 0}
        )

    for duel in duels:
        a, b = duel["a"], duel["b"]
        row(a)["played"] += 1
        row(b)["played"] += 1
        if duel["winner"] is None:
            row(a)["draws"] += 1
            row(b)["draws"] += 1
        else:
            loser = b if duel["winner"] == a else a
            row(duel["winner"])["wins"] += 1
            row(loser)["losses"] += 1

    models = sorted(
        stats.values(),
        key=lambda s: (-s["wins"], s["losses"], -s["played"], s["model"]),
    )
    for s in models:
        s["win_pct"] = round(100 * s["wins"] / s["played"]) if s["played"] else 0

    recent = [
        {"a": d["a"], "b": d["b"], "winner": d["winner"], "game": d["game"]}
        for d in duels[-RECENT_DUELS:][::-1]
    ]
    return {"duels": len(duels), "models": models, "recent": recent}
