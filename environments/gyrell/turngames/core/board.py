from __future__ import annotations

from dataclasses import dataclass, field

from turngames.core.types import JsonValue
from turngames.core.visibility import VisibilityContext, VisibilityRule


@dataclass(frozen=True)
class BoardCell:
    """A stable board position."""

    id: str
    row: int
    col: int


@dataclass(frozen=True)
class BoardLayer:
    """A named layer of cell values with its own visibility rule."""

    id: str
    values: dict[str, JsonValue]
    visibility: VisibilityRule = field(default_factory=VisibilityRule.public)


@dataclass(frozen=True)
class BoardView:
    """Projected board for a specific viewer."""

    width: int
    height: int
    cells: tuple[dict[str, JsonValue], ...]

    def rows(self) -> tuple[tuple[dict[str, JsonValue], ...], ...]:
        return tuple(
            self.cells[row * self.width : (row + 1) * self.width]
            for row in range(self.height)
        )


@dataclass(frozen=True)
class GridBoard:
    """A rectangular board with independently visible layers."""

    width: int
    height: int
    cells: tuple[BoardCell, ...]
    layers: tuple[BoardLayer, ...]

    def __post_init__(self) -> None:
        if len(self.cells) != self.width * self.height:
            raise ValueError("cell count must match width * height")

    @classmethod
    def from_labels(
        cls,
        width: int,
        height: int,
        labels: tuple[str, ...],
        label_layer: str = "word",
    ) -> "GridBoard":
        if len(labels) != width * height:
            raise ValueError("label count must match width * height")
        cells = tuple(
            BoardCell(id=f"c{index}", row=index // width, col=index % width)
            for index in range(width * height)
        )
        layer = BoardLayer(
            id=label_layer,
            values={cell.id: labels[index] for index, cell in enumerate(cells)},
            visibility=VisibilityRule.public(),
        )
        return cls(width=width, height=height, cells=cells, layers=(layer,))

    def with_layer(self, layer: BoardLayer) -> "GridBoard":
        return GridBoard(
            width=self.width,
            height=self.height,
            cells=self.cells,
            layers=self.layers + (layer,),
        )

    def project(self, context: VisibilityContext) -> BoardView:
        visible_layers = tuple(
            layer for layer in self.layers if layer.visibility.allows(context)
        )
        projected: list[dict[str, JsonValue]] = []
        for cell in self.cells:
            cell_view: dict[str, JsonValue] = {
                "id": cell.id,
                "row": cell.row,
                "col": cell.col,
            }
            for layer in visible_layers:
                if cell.id in layer.values:
                    cell_view[layer.id] = layer.values[cell.id]
            projected.append(cell_view)
        return BoardView(width=self.width, height=self.height, cells=tuple(projected))

