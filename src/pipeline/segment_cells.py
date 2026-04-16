import cv2
import numpy as np


def crop_grid_from_bbox(image, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    grid = image[y1:y2, x1:x2]
    side = max(grid.shape[:2])
    if grid.shape[0] != grid.shape[1]:
        diff = abs(grid.shape[0] - grid.shape[1])
        if grid.shape[0] < grid.shape[1]:
            pad = diff // 2
            grid = cv2.copyMakeBorder(grid, pad, diff - pad, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        else:
            pad = diff // 2
            grid = cv2.copyMakeBorder(grid, 0, 0, pad, diff - pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return grid


def warp_grid_from_corners(image, corners, output_size=900):
    corners = np.array(corners, dtype='float32')
    dst = np.array([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, M, (output_size, output_size))


def split_grid_to_cells(grid_image, size=9):
    h, w = grid_image.shape[:2]
    cell_h = h // size
    cell_w = w // size
    cells = []
    for row in range(size):
        for col in range(size):
            y1 = row * cell_h
            x1 = col * cell_w
            y2 = y1 + cell_h
            x2 = x1 + cell_w
            cell = grid_image[y1:y2, x1:x2]
            cells.append(cell)
    return cells


def annotate_cells(grid_image, size=9):
    annotated = grid_image.copy()
    h, w = annotated.shape[:2]
    cell_h = h // size
    cell_w = w // size
    for i in range(1, size):
        cv2.line(annotated, (i * cell_w, 0), (i * cell_w, h), (0, 255, 0), 1)
        cv2.line(annotated, (0, i * cell_h), (w, i * cell_h), (0, 255, 0), 1)
    return annotated


def assign_detections_to_cells(detections, grid_shape, size=9):
    h, w = grid_shape[:2]
    cell_h = h / size
    cell_w = w / size
    assignments = [None] * (size * size)
    for detection in detections:
        x1, y1, x2, y2 = detection['xyxy']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        row = min(int(cy // cell_h), size - 1)
        col = min(int(cx // cell_w), size - 1)
        index = row * size + col
        current = assignments[index]
        if current is None or (detection['score'] or 0) > (current['score'] or 0):
            assignments[index] = detection
    return assignments
