import os

import pytest

from src.pipeline.segment_cells import split_grid_to_cells
from src.pipeline.solver import solve_sudoku


def test_split_grid_to_cells_returns_81_cells():
    import numpy as np
    grid = np.full((900, 900, 3), 255, dtype=np.uint8)
    cells = split_grid_to_cells(grid)
    assert len(cells) == 81
    assert cells[0].shape[0] == 100
    assert cells[-1].shape[1] == 100


def test_solver_consistency_on_empty_grid():
    grid = [[0] * 9 for _ in range(9)]
    assert solve_sudoku(grid)
    assert all(1 <= value <= 9 for row in grid for value in row)
