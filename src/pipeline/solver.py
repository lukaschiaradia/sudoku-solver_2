from copy import deepcopy


def is_valid(grid, row, col, value):
    if value == 0:
        return True
    for i in range(9):
        if i != col and grid[row][i] == value:
            return False
        if i != row and grid[i][col] == value:
            return False
    sr = (row // 3) * 3
    sc = (col // 3) * 3
    for r in range(sr, sr + 3):
        for c in range(sc, sc + 3):
            if (r != row or c != col) and grid[r][c] == value:
                return False
    return True


def find_empty(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None


def solve_sudoku(grid):
    location = find_empty(grid)
    if location is None:
        return True
    row, col = location
    for value in range(1, 10):
        if is_valid(grid, row, col, value):
            grid[row][col] = value
            if solve_sudoku(grid):
                return True
            grid[row][col] = 0
    return False


def _find_next_cell_with_candidates(grid, candidates):
    best = None
    best_options = None
    for row in range(9):
        for col in range(9):
            if grid[row][col] != 0:
                continue
            options = candidates[row][col] if candidates[row][col] else list(range(1, 10))
            valid_options = [value for value in options if is_valid(grid, row, col, value)]
            if not valid_options:
                return (row, col, [])
            if best is None or len(valid_options) < len(best_options):
                best = (row, col)
                best_options = valid_options
    if best is None:
        return None
    return best[0], best[1], best_options


def solve_sudoku_with_candidates(grid, candidates):
    location = _find_next_cell_with_candidates(grid, candidates)
    if location is None:
        return True
    row, col, options = location
    if not options:
        return False
    for value in options:
        if is_valid(grid, row, col, value):
            grid[row][col] = value
            if solve_sudoku_with_candidates(grid, candidates):
                return True
            grid[row][col] = 0
    return False


def format_grid(grid):
    return '\n'.join(' '.join(str(val) for val in row) for row in grid)
