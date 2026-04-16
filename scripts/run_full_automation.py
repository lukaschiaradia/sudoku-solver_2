import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.automation import fill_sudoku_on_screen
from src.pipeline.pipeline import infer_sudoku_from_image
from src.pipeline.screen_capture import capture_screen
from src.pipeline.solver import format_grid


def main():
    parser = argparse.ArgumentParser(description='Capture screen, solve Sudoku and fill it on screen')
    parser.add_argument('--screenshot', default='outputs/screenshot.png', help='Path to save the screenshot')
    parser.add_argument('--image', default=None, help='Use existing screenshot instead of capturing a new one')
    parser.add_argument('--weights', default='weights/best.pt', help='Cell detection YOLO weights file')
    parser.add_argument('--grid-weights', default='weights2/best.pt', help='Grid detection YOLO weights file')
    parser.add_argument('--grid-conf', type=float, default=0.01, help='YOLO confidence threshold for grid detection')
    parser.add_argument('--yolo-conf', type=float, default=0.2, help='YOLO confidence threshold for cell detection')
    parser.add_argument('--output', default='outputs/predictions/full_automation.json', help='Output JSON file')
    parser.add_argument('--debug', action='store_true', help='Save debug images for the grid and cells')
    parser.add_argument('--debug-dir', default='outputs/debug', help='Directory to save debug images')
    parser.add_argument('--dry-run', action='store_true', help='Do not click or type on screen')
    args = parser.parse_args()

    image_path = args.image if args.image else capture_screen(args.screenshot)
    try:
        result = infer_sudoku_from_image(
            image_path,
            weights_path=args.weights,
            grid_weights_path=args.grid_weights,
            recognition_method='best',
            template_folder=None,
            yolo_conf=args.yolo_conf,
            grid_conf=args.grid_conf,
            debug=args.debug,
            debug_dir=args.debug_dir,
        )
    except ValueError as exc:
        print('ERROR: ', exc)
        raise

    output_data = {
        'image': image_path,
        'raw_grid': result['raw_grid'],
        'solved_grid': result['solved_grid'],
        'cell_detections': len(result['cell_detections']),
        'assignments': sum(1 for a in result['assignments'] if a is not None),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print('Raw grid:')
    print(format_grid(result['raw_grid']))
    print('\nSolved grid:')
    print(format_grid(result['solved_grid']))
    print(f'Output written to {args.output}')

    if not args.dry_run:
        fill_sudoku_on_screen(result['raw_grid'], result['solved_grid'], result.get('grid_bbox'))
    else:
        print('Dry run: no clicks performed.')


if __name__ == '__main__':
    main()
