import unittest

from turngames.core.board import BoardLayer, GridBoard
from turngames.core.visibility import VisibilityContext, VisibilityRule


class BoardVisibilityTest(unittest.TestCase):
    def test_layers_are_projected_by_role(self) -> None:
        board = GridBoard.from_labels(2, 1, ("moon", "river"))
        board = board.with_layer(
            BoardLayer(
                id="key",
                values={"c0": "red", "c1": "blue"},
                visibility=VisibilityRule.for_roles("cluegiver"),
            )
        )

        guesser_view = board.project(VisibilityContext(role="guesser"))
        cluegiver_view = board.project(VisibilityContext(role="cluegiver"))

        self.assertEqual(guesser_view.cells[0]["word"], "moon")
        self.assertNotIn("key", guesser_view.cells[0])
        self.assertEqual(cluegiver_view.cells[0]["key"], "red")


if __name__ == "__main__":
    unittest.main()

