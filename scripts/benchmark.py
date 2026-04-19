"""
Benchmark des méthodes de reconnaissance de chiffres.

Métriques mesurées sur les images raw/ :
  - solve_rate    : % d'images où le sudoku a pu être résolu (proxy de précision)
  - fill_rate     : % moyen de cellules reconnues comme non-vides
  - avg_time      : temps moyen de traitement par image (secondes)
  - digits_found  : nombre moyen de chiffres reconnus par image

Usage :
    python scripts/benchmark.py
    python scripts/benchmark.py --images raw/ --methods cnn easyocr tesseract template
"""

import argparse
import glob
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.pipeline.pipeline import infer_sudoku_from_image

METHODS = ['cnn', 'easyocr', 'tesseract']


def run_method_on_image(image_path, method, grid_weights, cell_weights):
    start = time.time()
    try:
        result = infer_sudoku_from_image(
            image_path,
            weights_path=cell_weights,
            grid_weights_path=grid_weights,
            recognition_method=method,
            template_folder='data/templates',
            yolo_conf=0.3,
            grid_conf=0.1,
            debug=False,
        )
        elapsed = time.time() - start
        raw = result['raw_grid']
        solved = result['solved_grid']
        digits = sum(1 for row in raw for v in row if v != 0)
        solved_ok = all(v != 0 for row in solved for v in row)
        return {
            'solved': solved_ok,
            'digits': digits,
            'time': elapsed,
            'error': None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            'solved': False,
            'digits': 0,
            'time': elapsed,
            'error': str(e)[:60],
        }


def benchmark(args):
    images = sorted(glob.glob(str(Path(args.images) / '*.png')))[:args.limit]
    print(f'Images : {len(images)} | Méthodes : {args.methods}')
    print()

    results = {}
    for method in args.methods:
        print(f'--- {method.upper()} ---')
        stats = []
        for i, img in enumerate(images):
            r = run_method_on_image(img, method, args.grid_weights, args.weights)
            stats.append(r)
            status = 'OK' if r['solved'] else 'FAIL'
            print(f'  [{i+1:3d}/{len(images)}] {Path(img).stem:15s} | {status} | {r["digits"]:2d} digits | {r["time"]:.2f}s')

        solve_rate = sum(1 for s in stats if s['solved']) / len(stats) * 100
        fill_rate = sum(s['digits'] for s in stats) / len(stats) / 81 * 100
        avg_time = sum(s['time'] for s in stats) / len(stats)
        avg_digits = sum(s['digits'] for s in stats) / len(stats)

        results[method] = {
            'solve_rate': solve_rate,
            'fill_rate': fill_rate,
            'avg_time': avg_time,
            'avg_digits': avg_digits,
        }
        print(f'  → solve_rate={solve_rate:.1f}% | fill_rate={fill_rate:.1f}% | avg_time={avg_time:.2f}s | avg_digits={avg_digits:.1f}/81')
        print()

    # Tableau comparatif final
    print('=' * 70)
    print(f'{"Méthode":<12} {"Solve rate":>12} {"Fill rate":>10} {"Avg time":>10} {"Digits/81":>10}')
    print('-' * 70)
    for method, r in results.items():
        print(f'{method:<12} {r["solve_rate"]:>11.1f}% {r["fill_rate"]:>9.1f}% {r["avg_time"]:>9.2f}s {r["avg_digits"]:>9.1f}')
    print('=' * 70)

    # Identifier le meilleur
    best = max(results, key=lambda m: (results[m]['solve_rate'], -results[m]['avg_time']))
    print(f'\nMeilleure méthode : {best.upper()} (solve_rate={results[best]["solve_rate"]:.1f}%)')

    # Comparaisons chiffrées (format attendu par le sujet)
    print('\n--- Comparaisons chiffrées ---')
    methods_list = list(results.keys())
    for i in range(len(methods_list)):
        for j in range(i + 1, len(methods_list)):
            a, b = methods_list[i], methods_list[j]
            ra, rb = results[a], results[b]
            if rb['avg_time'] > 0:
                speedup = rb['avg_time'] / ra['avg_time']
                if speedup > 1:
                    print(f'{a} est {speedup:.1f}x plus rapide que {b} ({ra["avg_time"]:.2f}s vs {rb["avg_time"]:.2f}s)')
                else:
                    print(f'{b} est {1/speedup:.1f}x plus rapide que {a} ({rb["avg_time"]:.2f}s vs {ra["avg_time"]:.2f}s)')
            diff = ra['solve_rate'] - rb['solve_rate']
            if abs(diff) > 0.1:
                better = a if diff > 0 else b
                worse = b if diff > 0 else a
                print(f'{better} résout {abs(diff):.1f}% de grilles en plus que {worse}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='raw')
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    parser.add_argument('--limit', type=int, default=20, help='Nombre max d\'images à tester')
    parser.add_argument('--weights', default='weights/best.pt')
    parser.add_argument('--grid-weights', default='weights2/best.pt')
    args = parser.parse_args()
    benchmark(args)


if __name__ == '__main__':
    main()
