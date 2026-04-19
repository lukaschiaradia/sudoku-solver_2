"""
Phase 1 : extrait tous les crops de cellules non-vides depuis raw/
et les labellise automatiquement avec EasyOCR (confidence > 0.85).

Structure de sortie :
  data/digit_crops/
    labeled/1/, labeled/2/, ... labeled/9/   ← crops avec label sûr
    unknown/                                  ← crops à labelliser manuellement
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import glob

from src.pipeline.pipeline import load_image
from src.pipeline.detect_grid import detect_grid_yolo_full, detect_grid_opencv
from src.pipeline.segment_cells import crop_grid_from_bbox, warp_grid_from_corners, split_grid_to_cells
from src.pipeline.recognize_digits import is_cell_blank, build_easyocr_reader, recognize_digit_easyocr_candidates

RAW_DIR = 'raw'
OUT_DIR = 'data/digit_crops'
GRID_WEIGHTS = 'weights2/best.pt'
CONF_THRESHOLD = 0.85  # confiance minimale pour auto-label


def detect_grid(image_path):
    image = load_image(image_path)
    try:
        det = detect_grid_yolo_full(image_path, weights_path=GRID_WEIGHTS, conf=0.1)
        if det is not None:
            return crop_grid_from_bbox(image, det['xyxy'])
    except Exception:
        pass
    corners = detect_grid_opencv(image_path)
    return warp_grid_from_corners(image, corners)


def main():
    reader = build_easyocr_reader()

    for digit in range(1, 10):
        Path(f'{OUT_DIR}/labeled/{digit}').mkdir(parents=True, exist_ok=True)
    Path(f'{OUT_DIR}/unknown').mkdir(parents=True, exist_ok=True)

    images = sorted(glob.glob(f'{RAW_DIR}/*.png'))
    print(f'Trouvé {len(images)} images dans {RAW_DIR}/')

    total_labeled = 0
    total_unknown = 0

    for img_path in images:
        name = Path(img_path).stem
        print(f'  Traitement {name}...', end=' ')

        try:
            grid = detect_grid(img_path)
        except Exception as e:
            print(f'ERREUR détection grille: {e}')
            continue

        cells = split_grid_to_cells(grid)

        labeled = 0
        unknown = 0
        for idx, cell in enumerate(cells):
            row, col = divmod(idx, 9)

            if is_cell_blank(cell):
                continue

            candidates = recognize_digit_easyocr_candidates(cell, reader=reader)

            if candidates and candidates[0][1] >= CONF_THRESHOLD:
                digit = candidates[0][0]
                out_path = f'{OUT_DIR}/labeled/{digit}/{name}_r{row}_c{col}.png'
                cv2.imwrite(out_path, cell)
                labeled += 1
            else:
                out_path = f'{OUT_DIR}/unknown/{name}_r{row}_c{col}.png'
                cv2.imwrite(out_path, cell)
                unknown += 1

        total_labeled += labeled
        total_unknown += unknown
        print(f'{labeled} labelisés, {unknown} inconnus')

    print(f'\nTerminé : {total_labeled} crops labelisés, {total_unknown} à vérifier manuellement')
    print(f'Regarde data/digit_crops/unknown/ et déplace chaque image dans le bon dossier labeled/1..9/')


if __name__ == '__main__':
    main()
