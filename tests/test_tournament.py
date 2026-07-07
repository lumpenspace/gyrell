import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from server.channel import (
    DUEL_EVERY,
    duel_lineup,
    duel_pairing,
    is_duel_match,
)
from server.tournament import duel_counts, duel_standings


def _write_duel(replay_dir: Path, seed: str, a: str, b: str, winner: str | None) -> None:
    """Archive a minimal but well-formed duel replay: meta (duel-tagged) +
    one omniscient snapshot carrying the seats + a final_score."""
    seats = [
        {"id": f"red_{i}", "role": "guesser", "team_id": "red", "label": a}
        for i in range(3)
    ] + [
        {"id": f"blue_{i}", "role": "guesser", "team_id": "blue", "label": b}
        for i in range(3)
    ]
    records = [
        {"kind": "meta", "game_id": "codewords", "seed": seed, "match_kind": "duel",
         "duel": {"a": a, "b": b}},
        {"kind": "snapshot", "viewer": "audience_omniscient", "view": {"seats": seats}},
        {"kind": "final_score", "score": {"winner": winner}},
    ]
    with gzip.open(replay_dir / f"{seed}.jsonl.gz", "wt") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


class DuelSchedulingTest(unittest.TestCase):
    def test_cadence(self) -> None:
        # With the default cadence, every DUEL_EVERY-th match is a duel.
        self.assertTrue(is_duel_match(DUEL_EVERY))
        self.assertTrue(is_duel_match(DUEL_EVERY * 2))
        self.assertFalse(is_duel_match(DUEL_EVERY + 1))
        self.assertFalse(is_duel_match(1))

    def test_lineup_is_three_a_side(self) -> None:
        seat_ids = (
            "red_cluegiver", "red_guesser_1", "red_guesser_2",
            "blue_cluegiver", "blue_guesser_1", "blue_guesser_2",
        )
        lineup = duel_lineup(seat_ids, "openai/gpt-x", "x-ai/grok-y")
        red = {s for sid, s in lineup.items() if sid.startswith("red")}
        blue = {s for sid, s in lineup.items() if sid.startswith("blue")}
        self.assertEqual(red, {"openai/gpt-x"})
        self.assertEqual(blue, {"x-ai/grok-y"})

    def test_pairing_deterministic_and_distinct(self) -> None:
        a, b = duel_pairing(3, {})
        self.assertEqual((a, b), duel_pairing(3, {}))  # deterministic
        self.assertNotEqual(a, b)  # two different models

    def test_pairing_prefers_least_fought(self) -> None:
        # Everyone has fought a lot except two fresh models; those two get picked.
        from server.channel import ROSTER

        tails = [slug.split("/")[-1] for slug in ROSTER]
        fought = {t: 10 for t in tails}
        fresh = tails[3], tails[6]
        fought[fresh[0]] = 0
        fought[fresh[1]] = 0
        a, b = duel_pairing(99, fought)
        self.assertEqual({a.split("/")[-1], b.split("/")[-1]}, set(fresh))


class DuelLadderTest(unittest.TestCase):
    def test_standings_and_counts(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            _write_duel(d, "live-1-3", "alpha", "bravo", "red")   # alpha beats bravo
            _write_duel(d, "live-2-6", "alpha", "carol", "blue")  # carol beats alpha
            _write_duel(d, "live-3-9", "bravo", "carol", None)    # draw

            counts = duel_counts(d)
            self.assertEqual(counts["alpha"], 2)
            self.assertEqual(counts["bravo"], 2)
            self.assertEqual(counts["carol"], 2)

            standings = duel_standings(d)
            self.assertEqual(standings["duels"], 3)
            rows = {r["model"]: r for r in standings["models"]}
            self.assertEqual((rows["alpha"]["wins"], rows["alpha"]["losses"]), (1, 1))
            self.assertEqual((rows["carol"]["wins"], rows["carol"]["draws"]), (1, 1))
            self.assertEqual((rows["bravo"]["wins"], rows["bravo"]["losses"]), (0, 1))
            # Most recent duel first.
            self.assertEqual(standings["recent"][0]["winner"], None)

    def test_non_duel_replays_are_ignored(self) -> None:
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            # A normal (untagged) match must not enter the ladder.
            seats = [
                {"id": "red_0", "role": "guesser", "team_id": "red", "label": "alpha"},
                {"id": "blue_0", "role": "guesser", "team_id": "blue", "label": "bravo"},
            ]
            records = [
                {"kind": "meta", "game_id": "codewords", "seed": "live-1-1"},
                {"kind": "snapshot", "viewer": "audience_omniscient", "view": {"seats": seats}},
                {"kind": "final_score", "score": {"winner": "red"}},
            ]
            with gzip.open(d / "live-1-1.jsonl.gz", "wt") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            self.assertEqual(duel_standings(d)["duels"], 0)
            self.assertEqual(duel_counts(d), {})


if __name__ == "__main__":
    unittest.main()
