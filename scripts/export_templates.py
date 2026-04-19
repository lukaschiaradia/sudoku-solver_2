import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2

from src.pipeline.recognize_digits import build_digit_templates


def main():
    output_dir = Path('data/templates')
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = build_digit_templates()
    for digit, template_list in templates.items():
        for idx, template in enumerate(template_list, start=1):
            file_name = f'{digit}_{idx}.png' if len(template_list) > 1 else f'{digit}.png'
            path = output_dir / file_name
            cv2.imwrite(str(path), template)
            print(f'Saved template: {path}')

    print(f'Exported {sum(len(v) for v in templates.values())} templates to {output_dir}')


if __name__ == '__main__':
    main()
