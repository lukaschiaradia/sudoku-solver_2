import os
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def load_yolo_model(weights_path: str):
    if YOLO is None:
        raise ImportError('ultralytics package is required for YOLO detection')
    if not Path(weights_path).exists():
        raise FileNotFoundError(f'YOLO weights not found: {weights_path}')
    return YOLO(weights_path)


def detect_grid_yolo(image_path: str, weights_path: str = 'weights/best.pt', conf: float = 0.4):
    model = load_yolo_model(weights_path)
    results = model(image_path, conf=conf)
    detections = []
    for result in results:
        boxes = getattr(result, 'boxes', None)
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int).flatten().tolist()
                score = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else None
                detections.append({'xyxy': xyxy, 'score': score})
    return detections


def detect_grid_yolo_full(image_path: str, weights_path: str = 'weights2/best.pt', conf: float = 0.2):
    model = load_yolo_model(weights_path)
    results = model(image_path, conf=conf)
    best_box = None
    best_area = 0
    for result in results:
        boxes = getattr(result, 'boxes', None)
        names = getattr(result, 'names', {})
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
            class_name = names.get(cls)
            if class_name is not None and class_name != 'grid':
                continue
            xyxy = box.xyxy.cpu().numpy().astype(int).flatten().tolist()
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area > best_area:
                best_area = area
                best_box = {'xyxy': xyxy, 'score': float(box.conf.cpu().numpy()), 'class_name': class_name}
    return best_box


def detect_cells_yolo(image, weights_path: str = 'weights/best.pt', conf: float = 0.2):
    model = load_yolo_model(weights_path)
    if isinstance(image, (str, Path)):
        results = model(str(image), conf=conf)
    else:
        results = model(image, conf=conf)

    detections = []
    for result in results:
        boxes = getattr(result, 'boxes', None)
        names = getattr(result, 'names', {})
        if boxes is not None:
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy().astype(int).flatten().tolist()
                score = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else None
                class_id = int(box.cls.cpu().numpy()) if hasattr(box, 'cls') else None
                detections.append({
                    'xyxy': xyxy,
                    'score': score,
                    'class_id': class_id,
                    'class_name': names.get(class_id) if names is not None else None,
                })
    return detections


def detect_grid_opencv(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10000:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            grid_contour = approx
            max_area = area

    if grid_contour is None:
        raise RuntimeError('No Sudoku grid contour detected with OpenCV')

    grid_contour = grid_contour.reshape(4, 2)
    rect = order_points(grid_contour)
    return rect.tolist()


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
