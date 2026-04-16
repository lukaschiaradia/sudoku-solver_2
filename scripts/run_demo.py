import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.pipeline import infer_sudoku_from_image
from src.pipeline.solver import format_grid


def main():
    parser = argparse.ArgumentParser(description='Run Sudoku vision pipeline on a screenshot')
    parser.add_argument('--image', default='raw/sudoku_0.png', help='Path to Sudoku screenshot')
    parser.add_argument('--weights', default='weights/best.pt', help='Cell YOLO weights file')
    parser.add_argument('--grid-weights', default='weights2/best.pt', help='Grid YOLO weights file')
    parser.add_argument('--grid-conf', type=float, default=0.2, help='YOLO confidence threshold for grid detection')
    parser.add_argument('--output', default='outputs/predictions/result.json', help='Output JSON file')
    parser.add_argument('--method', default='best', choices=['best', 'easyocr', 'tesseract', 'template'], help='Digit recognition method')
    parser.add_argument('--template-folder', default='data/templates', help='Folder containing digit templates for template matching')
    parser.add_argument('--yolo-conf', type=float, default=0.2, help='YOLO confidence threshold for cell detection')
    args = parser.parse_args()

    result = infer_sudoku_from_image(
        args.image,
        weights_path=args.weights,
        grid_weights_path=args.grid_weights,
        recognition_method=args.method,
        template_folder=args.template_folder,
        yolo_conf=args.yolo_conf,
        grid_conf=args.grid_conf,
    )

    output = {
        'image': args.image,
        'raw_grid': result['raw_grid'],
        'solved_grid': result['solved_grid'],
        'assignments': result['assignments'],
        'cell_detections': result['cell_detections'],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print('Recognition method:', args.method)
    print('YOLO confidence threshold:', args.yolo_conf)
    print('Raw grid:')
    print(format_grid(result['raw_grid']))
    print('\nSolved grid:')
    print(format_grid(result['solved_grid']))
    print(f'Output written to {args.output}')


if __name__ == '__main__':
    main()
