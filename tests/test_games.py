"""The /games search index: summaries, moment tags, and filters."""

import gzip
import json

from server import games


def _replay(seed, game, seats, score, events=(), match_kind=None):
    meta = {"kind": "meta", "game_id": game, "seed": seed}
    if match_kind:
        meta["match_kind"] = match_kind
    records = [
        meta,
        {
            "kind": "snapshot",
            "viewer": "audience_omniscient",
            "view": {"seats": seats},
        },
        *({"kind": "event", "event": e} for e in events),
        {"kind": "final_score", "score": score},
    ]
    return records


def _seats(red, blue):
    seats = []
    for i, label in enumerate(red):
        seats.append({"id": f"r{i}", "team_id": "red", "role": "guesser", "label": label})
    for i, label in enumerate(blue):
        seats.append({"id": f"b{i}", "team_id": "blue", "role": "guesser", "label": label})
    return seats


def _write(replay_dir, records):
    path = replay_dir / f"{records[0]['seed']}.jsonl.gz"
    with gzip.open(path, "wt") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_summarize_codewords_assassin_and_sweep():
    records = _replay(
        "live-100-1",
        "codewords",
        _seats(["grok", "grok"], ["gpt", "claude"]),
        {
            "winner": "red",
            "terminal_reason": "assassin_revealed",
            "red_remaining": 0,
            "blue_remaining": 5,
            "event_count": 500,
        },
    )
    card = games.summarize(records, "live-100-1")
    assert card["game"] == "codewords"
    assert card["winner"] == "red"
    assert card["same_team_models"] == ["grok"]
    assert {"assassin", "blowout", "marathon"} <= set(card["moments"])


def test_summarize_taboo_blowout_and_flawless():
    records = _replay(
        "live-100-2",
        "taboo",
        _seats(["a", "b"], ["c", "d"]),
        {
            "winner": "blue",
            "terminal_reason": "most_points",
            "red_points": 3,
            "blue_points": 16,
            "violations": {"red": 2},
            "event_count": 100,
        },
    )
    card = games.summarize(records, "live-100-2")
    assert "blowout" in card["moments"]
    assert card["same_team_models"] == []


def test_comeback_from_round_history():
    events = [
        {"type": "round_ended", "payload": {"round_number": 3, "points": {"red": 8, "blue": 2}}},
    ]
    records = _replay(
        "live-100-3",
        "taboo",
        _seats(["a"], ["b"]),
        {"winner": "blue", "red_points": 9, "blue_points": 10, "violations": {}, "event_count": 10},
        events,
    )
    assert "comeback" in games.summarize(records, "live-100-3")["moments"]


def test_incomplete_archive_is_skipped():
    records = _replay("live-100-4", "codewords", _seats(["a"], ["b"]), {})[:-1]
    assert games.summarize(records, "live-100-4") is None


def test_search_filters(tmp_path):
    _write(
        tmp_path,
        _replay(
            "live-200-1",
            "codewords",
            _seats(["grok", "grok"], ["gpt", "claude"]),
            {"winner": "red", "terminal_reason": "all_team_words_revealed",
             "red_remaining": 0, "blue_remaining": 4, "event_count": 10},
        ),
    )
    _write(
        tmp_path,
        _replay(
            "live-300-1",
            "taboo",
            _seats(["grok", "gpt"], ["claude", "claude"]),
            {"winner": "blue", "red_points": 2, "blue_points": 13,
             "violations": {}, "event_count": 10},
            match_kind="duel",
        ),
    )

    everything = games.search(tmp_path)
    assert everything["total"] == 2
    # Newest (higher seed timestamp) first.
    assert everything["games"][0]["name"] == "live-300-1"
    assert "grok" in everything["models"]

    grok_same = games.search(tmp_path, model="grok", team="same")
    assert [g["name"] for g in grok_same["games"]] == ["live-200-1"]

    grok_wins = games.search(tmp_path, model="grok", winner_model="grok")
    assert [g["name"] for g in grok_wins["games"]] == ["live-200-1"]

    duels = games.search(tmp_path, kind="duel")
    assert [g["name"] for g in duels["games"]] == ["live-300-1"]

    sweeps = games.search(tmp_path, moment="blowout")
    assert "live-200-1" in [g["name"] for g in sweeps["games"]]

    assert games.search(tmp_path, game="taboo", model="grok", team="mixed")["total"] == 1


def test_word_search(tmp_path):
    records = _replay(
        "live-500-1",
        "codewords",
        _seats(["grok"], ["gpt"]),
        {"winner": "red", "terminal_reason": "all_team_words_revealed",
         "red_remaining": 0, "blue_remaining": 3, "event_count": 10},
    )
    records[1]["view"]["board"] = {"cells": [{"word": "Volcano"}, {"word": "mirror"}]}
    _write(tmp_path, records)

    hit = games.search(tmp_path, word="volcano")
    assert hit["total"] == 1
    assert "words" not in hit["games"][0]  # per-card word lists stay server-side
    assert games.search(tmp_path, word="glacier")["total"] == 0
    assert hit["words"] == ["mirror", "volcano"]


def test_scripted_lineups_are_hidden(tmp_path):
    _write(
        tmp_path,
        _replay(
            "live-400-1",
            "codewords",
            _seats(["baseline-bravo", "grok"], ["gpt", "scripted-v1"]),
            {"winner": "red", "terminal_reason": "all_team_words_revealed",
             "red_remaining": 0, "blue_remaining": 2, "event_count": 10},
        ),
    )
    result = games.search(tmp_path)
    assert result["total"] == 0
    assert result["models"] == []
