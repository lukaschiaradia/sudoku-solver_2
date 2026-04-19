import os
from pathlib import Path

import cv2

from src.pipeline.detect_grid import detect_cells_yolo, detect_grid_opencv, detect_grid_yolo_full
from src.pipeline.segment_cells import crop_grid_from_bbox, warp_grid_from_corners, split_grid_to_cells, assign_detections_to_cells
from src.pipeline.recognize_digits import (
    recognize_digit,
    recognize_digit_candidates,
    is_cell_blank,
    build_templates,
    build_easyocr_reader,
)
from src.pipeline.solver import solve_sudoku, solve_sudoku_with_candidates, format_grid


def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image


def infer_sudoku_from_image(
    image_path: str,
    weights_path: str = 'weights/best.pt',
    grid_weights_path: str | None = 'weights2/best.pt',
    recognition_method: str = 'easyocr',
    template_folder: str | None = None,
    yolo_conf: float = 0.2,
    grid_conf: float = 0.2,
    debug: bool = False,
    debug_dir: str = 'outputs/debug',
):
    image = load_image(image_path)
    grid = None
    bbox = None
    if debug:
        Path(debug_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, 'raw_screenshot.png'), image)

    if grid_weights_path is not None:
        try:
            grid_detection = detect_grid_yolo_full(image_path, weights_path=grid_weights_path, conf=grid_conf)
            if grid_detection is not None:
                bbox = grid_detection['xyxy']
                grid = crop_grid_from_bbox(image, bbox)
        except Exception:
            grid = None
    if grid is None:
        corners = detect_grid_opencv(image_path)
        bbox = [int(min(c[0] for c in corners)), int(min(c[1] for c in corners)), int(max(c[0] for c in corners)), int(max(c[1] for c in corners))]
        grid = warp_grid_from_corners(image, corners)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, 'warped_grid.png'), grid)

    cells = split_grid_to_cells(grid)

    cell_detections = []
    try:
        cell_detections = detect_cells_yolo(grid, weights_path=weights_path, conf=yolo_conf)
    except Exception:
        cell_detections = []

    assignments = assign_detections_to_cells(cell_detections, grid.shape)
    templates = build_templates(template_folder) if recognition_method in ('template', 'best') else None
    reader = None
    if recognition_method in ('easyocr', 'best'):
        try:
            reader = build_easyocr_reader()
        except ImportError:
            reader = None

    raw_grid = []
    candidate_grid = []
    for row in range(9):
        row_values = []
        row_candidates = []
        for col in range(9):
            idx = row * 9 + col
            cell = cells[idx]
            if debug:
                debug_path = os.path.join(debug_dir, f'cell_{row}_{col}.png')
                cv2.imwrite(debug_path, cell)
            assignment = assignments[idx]
            if is_cell_blank(cell):
                value = 0
                candidates = list(range(1, 10))
            else:
                candidates = recognize_digit_candidates(
                    cell,
                    method=recognition_method,
                    templates=templates,
                    reader=reader,
                )
                if candidates:
                    value = candidates[0][0]
                    candidates = [digit for digit, _ in candidates]
                else:
                    value = 0
                    candidates = list(range(1, 10))
            row_values.append(value)
            row_candidates.append(candidates)
        raw_grid.append(row_values)
        candidate_grid.append(row_candidates)

    filled_cells = sum(1 for row in raw_grid for value in row if value != 0)
    if filled_cells == 0:
        raise ValueError(
            'No digits recognized in the detected Sudoku grid. '
            'Check that the screenshot contains a visible grid, that the YOLO weights are correct, '
            'and rerun with --debug to inspect the captured grid and cell crops.'
        )

    solved_grid = [list(row) for row in raw_grid]
    if not solve_sudoku(solved_grid):
        solved_grid = [list(row) for row in raw_grid]
        if not solve_sudoku_with_candidates(solved_grid, candidate_grid):
            raw_grid_str = '\n'.join(''.join(str(v) for v in row) for row in raw_grid)
            raise ValueError(
                f'Sudoku grid could not be solved. raw_grid:\n{raw_grid_str}\n' \
                f'cell_detections={len(cell_detections)}, assignments={sum(1 for a in assignments if a is not None)}'
            )
    return {
        'raw_grid': raw_grid,
        'solved_grid': solved_grid,
        'grid_image': grid,
        'cell_detections': cell_detections,
        'assignments': assignments,
        'grid_bbox': bbox,
    }


def save_grid_as_image(grid_image, destination_path):
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(destination_path, grid_image)
