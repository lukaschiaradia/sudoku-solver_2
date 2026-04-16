from copy import deepcopy


def is_valid(grid, row, col, value):
    if value == 0:
        return True
    for i in range(9):
        if grid[row][i] == value or grid[i][col] == value:
            return False
    sr = (row // 3) * 3
    sc = (col // 3) * 3
    for r in range(sr, sr + 3):
        for c in range(sc, sc + 3):
            if grid[r][c] == value:
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


def format_grid(grid):
    return '\n'.join(' '.join(str(val) for val in row) for row in grid)
