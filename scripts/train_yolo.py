import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Train a YOLOv8 model on a Sudoku grid dataset')
    parser.add_argument('--data', default='weights/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', default='yolov8n.pt', help='Base YOLOv8 model to fine-tune')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--project', default='runs/train', help='Project output folder')
    parser.add_argument('--name', default='sudoku_grid', help='Run name')
    args = parser.parse_args()

    import subprocess
    cmd = [
        'python3', '-m', 'ultralytics',
        'yolo', 'task=detect', 'mode=train',
        f'model={args.model}',
        f'data={args.data}',
        f'epochs={args.epochs}',
        f'imgsz={args.imgsz}',
        f'batch={args.batch}',
        f'project={args.project}',
        f'name={args.name}'
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
