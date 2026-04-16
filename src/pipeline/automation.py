"""Automation utilities for screen-based Sudoku solving.

This module is intentionally decoupled from perception. The detection pipeline must remain vision-only.
"""

import time
from typing import List, Optional

import pyautogui


def get_cell_centers_from_bbox(bbox: List[int], size: int = 9):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    cell_w = width / size
    cell_h = height / size
    centers = []
    for row in range(size):
        for col in range(size):
            cx = x1 + (col + 0.5) * cell_w
            cy = y1 + (row + 0.5) * cell_h
            centers.append((int(cx), int(cy)))
    return centers


def fill_sudoku_on_screen(raw_grid: List[List[int]], solved_grid: List[List[int]], grid_bbox: Optional[List[int]], click_delay: float = 0.1):
    if grid_bbox is None:
        raise ValueError('Grid bounding box required for screen automation')

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = click_delay

    centers = get_cell_centers_from_bbox(grid_bbox)
    print('Starting screen fill. Move your mouse to a corner if you need to abort.')
    time.sleep(2)

    for row in range(9):
        for col in range(9):
            if raw_grid[row][col] == 0:
                x, y = centers[row * 9 + col]
                pyautogui.click(x, y)
                pyautogui.write(str(solved_grid[row][col]), interval=0.05)
                time.sleep(click_delay)

    print('Screen fill complete.')
