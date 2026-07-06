import unittest

from server.leaderboard import (
    ELO_INITIAL,
    FALLBACK_LABEL,
    _fallback_fractions,
    aggregate_matches,
    relabel_fallback_seats,
)


def seats(red: tuple[str, ...], blue: tuple[str, ...], master_role: str = "cluegiver") -> list[dict]:
    out = []
    for team, labels in (("red", red), ("blue", blue)):
        for index, label in enumerate(labels):
            role = master_role if index == 0 else "guesser"
            out.append({"id": f"{team}_{role}_{index}", "role": role, "team_id": team, "label": label})
    return out


RED_TEAM = ("alpha", "bravo", "carol")
BLUE_TEAM = ("delta", "echo", "fox")


class LeaderboardEloTest(unittest.TestCase):
    def rows(self, matches):
        return {row["model"]: row for row in aggregate_matches(matches)["models"]}

    def test_winner_gains_and_loser_loses_symmetrically(self) -> None:
        rows = self.rows([(seats(RED_TEAM, BLUE_TEAM), "red", "codewords")])
        # Even 1000-vs-1000 opener at K=40: winners +20, losers -20.
        self.assertEqual(rows["alpha"]["elo"], round(ELO_INITIAL + 20))
        self.assertEqual(rows["delta"]["elo"], round(ELO_INITIAL - 20))

    def test_draw_moves_nobody_at_equal_ratings(self) -> None:
        rows = self.rows([(seats(RED_TEAM, BLUE_TEAM, "describer"), None, "taboo")])
        self.assertEqual(rows["alpha"]["elo"], round(ELO_INITIAL))
        self.assertEqual(rows["alpha"]["total"]["draws"], 1)
        self.assertEqual(rows["alpha"]["total"]["games"], 1)
        self.assertEqual(rows["alpha"]["streak"], 0)

    def test_beating_stronger_opponents_pays_more(self) -> None:
        # bravo's team beats delta's twice; then carol's fresh team beats the
        # same weakened delta team once. carol's single win must be worth less
        # than alpha's first win against then-equal opposition was.
        matches = [
            (seats(RED_TEAM, BLUE_TEAM), "red", "codewords"),
            (seats(RED_TEAM, BLUE_TEAM), "red", "codewords"),
            (seats(("gina", "hank", "iris"), BLUE_TEAM), "red", "codewords"),
        ]
        rows = self.rows(matches)
        first_win_gain = 20  # from test above
        self.assertLess(rows["gina"]["elo"] - ELO_INITIAL, first_win_gain)

    def test_per_game_pools_are_independent(self) -> None:
        matches = [
            (seats(RED_TEAM, BLUE_TEAM), "red", "codewords"),
            (seats(RED_TEAM, BLUE_TEAM, "describer"), "blue", "taboo"),
        ]
        rows = self.rows(matches)
        alpha = rows["alpha"]
        self.assertGreater(alpha["per_game"]["codewords"]["elo"], ELO_INITIAL)
        self.assertLess(alpha["per_game"]["taboo"]["elo"], ELO_INITIAL)
        self.assertEqual(alpha["per_game"]["codewords"]["games"], 1)
        self.assertEqual(alpha["per_game"]["taboo"]["games"], 1)
        # overall pool saw one win and one loss from even positions: back near start
        self.assertAlmostEqual(alpha["elo"], ELO_INITIAL, delta=2)

    def test_role_buckets_cover_describer(self) -> None:
        rows = self.rows([(seats(RED_TEAM, BLUE_TEAM, "describer"), "red", "taboo")])
        self.assertEqual(rows["alpha"]["describer"], {"games": 1, "wins": 1})
        self.assertEqual(rows["bravo"]["guesser"], {"games": 1, "wins": 1})
        self.assertEqual(rows["alpha"]["cluegiver"], {"games": 0, "wins": 0})

    def test_provisional_flag_clears_after_ten_games(self) -> None:
        matches = [(seats(RED_TEAM, BLUE_TEAM), "red", "codewords")] * 10
        rows = self.rows(matches)
        self.assertFalse(rows["alpha"]["provisional"])
        rows = self.rows(matches[:9])
        self.assertTrue(rows["alpha"]["provisional"])

    def test_sorted_by_elo(self) -> None:
        result = aggregate_matches(
            [
                (seats(RED_TEAM, BLUE_TEAM), "red", "codewords"),
                (seats(RED_TEAM, BLUE_TEAM), "red", "codewords"),
            ]
        )
        elos = [row["elo"] for row in result["models"]]
        self.assertEqual(elos, sorted(elos, reverse=True))
        self.assertEqual(result["matches"], 2)


def response_event(seat_id: str, said: str) -> dict:
    return {
        "kind": "event",
        "event": {
            "type": "actor_responded",
            "payload": {"seat_id": seat_id, "public_deliberation": said},
        },
    }


class FallbackPurgeTest(unittest.TestCase):
    def test_exact_counts_take_precedence(self) -> None:
        final = {
            "kind": "final_score",
            "score": {"winner": "red"},
            "turns": {"red_cluegiver": 10, "red_guesser_1": 8},
            "fallback_turns": {"red_cluegiver": 9},
        }
        fractions = _fallback_fractions([], final)
        self.assertAlmostEqual(fractions["red_cluegiver"], 0.9)
        self.assertEqual(fractions["red_guesser_1"], 0.0)

    def test_legacy_fingerprints_estimate_fractions(self) -> None:
        records = [
            response_event("red_cluegiver", "This should cover a couple of ours."),
            response_event("red_cluegiver", "Keeping it conservative this turn."),
            response_event("red_guesser_1", "Hmm, OCEAN links wave and ship for me."),
            response_event("red_guesser_1", "Agreed."),  # canned, but 1 of 2
        ]
        fractions = _fallback_fractions(records, {"kind": "final_score", "score": {}})
        self.assertEqual(fractions["red_cluegiver"], 1.0)
        self.assertEqual(fractions["red_guesser_1"], 0.5)

    def test_relabel_only_dominated_seats(self) -> None:
        seat_list = [
            {"id": "red_cluegiver", "label": "laguna-xs-2.1", "team_id": "red", "role": "cluegiver"},
            {"id": "red_guesser_1", "label": "claude-haiku-4.5", "team_id": "red", "role": "guesser"},
        ]
        relabeled = relabel_fallback_seats(seat_list, {"red_cluegiver": 1.0, "red_guesser_1": 0.2})
        self.assertEqual(relabeled[0]["label"], FALLBACK_LABEL)
        self.assertEqual(relabeled[1]["label"], "claude-haiku-4.5")

    def test_fallback_pseudo_player_hidden_but_rated_against(self) -> None:
        lineup = seats(RED_TEAM, BLUE_TEAM)
        lineup[0] = {**lineup[0], "label": FALLBACK_LABEL}  # red master was a bot
        result = aggregate_matches([(lineup, "red", "codewords")])
        models = {row["model"] for row in result["models"]}
        self.assertNotIn(FALLBACK_LABEL, models)
        rows = {row["model"]: row for row in result["models"]}
        # humans-of-record on the winning team still get credit; the bot doesn't surface
        self.assertEqual(rows["bravo"]["total"], {"games": 1, "wins": 1, "draws": 0})
        self.assertLess(rows["delta"]["elo"], ELO_INITIAL)


if __name__ == "__main__":
    unittest.main()
