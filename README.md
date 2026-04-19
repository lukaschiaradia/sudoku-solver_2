# Sudoku Solver — Computer Vision

Projet Epitech — Résolution automatique de grilles Sudoku par vision par ordinateur.

Le programme capture visuellement une grille Sudoku affichée dans un navigateur, reconnaît les chiffres, résout la grille, puis interagit automatiquement avec le site pour saisir la solution — **sans jamais utiliser le DOM HTML**.

---

## Pipeline

```
Screenshot → Détection grille (YOLO) → Warp & crop
    ↓
Segmentation 9×9 → Détection cellules (YOLO)
    ↓
Reconnaissance chiffres (CNN / EasyOCR / Tesseract)
    ↓
Résolution (backtracking + contraintes)
    ↓
Interaction automatique (pyautogui)
```

---

## Modèles et approches

### Détection de la grille
- **YOLO fine-tuné** (`weights2/best.pt`) — détecte la grille entière, effectue un warp de perspective
- **OpenCV** (fallback) — détection par contours si YOLO échoue

### Détection des cellules
- **YOLO fine-tuné** (`weights/best.pt`) — classifie chaque cellule en `cell_filled` / `cell_empty`

### Reconnaissance des chiffres
Trois approches implémentées et comparées :

| Méthode | Description |
|---|---|
| **CNN** | Réseau de neurones entraîné sur 3000+ crops réels de sudoku.com |
| **EasyOCR** | Modèle OCR pré-entraîné avec preprocessing adaptatif |
| **Tesseract** | OCR classique (PSM 10, whitelist 1-9) |

### Résolution
- Backtracking standard
- Backtracking avec propagation de contraintes (MRV heuristic)

---

## Installation

### Prérequis
- Python 3.12
- Tesseract OCR : [installer ici](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup
```bash
git clone <repo>
cd sudoku-solver_2

python -m venv .venv
# Windows :
.venv\Scripts\activate
# Linux/Mac :
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Utilisation

### Tester sur une image
```bash
python scripts/run_demo.py --image raw/sudoku_0.png
```

Avec debug (sauvegarde les crops de cellules) :
```bash
python scripts/run_demo.py --image raw/sudoku_0.png --debug --debug-dir outputs/debug
```

Options disponibles :
```
--image           Image à analyser
--weights         Poids YOLO détection cellules (défaut: weights/best.pt)
--grid-weights    Poids YOLO détection grille  (défaut: weights2/best.pt)
--method          Méthode OCR : cnn, easyocr, tesseract, best (défaut: best)
--yolo-conf       Seuil de confiance YOLO cellules (défaut: 0.3)
--grid-conf       Seuil de confiance YOLO grille   (défaut: 0.1)
--debug           Activer le mode debug
```

### Automation complète (screenshot → résolution → saisie automatique)
```bash
# Dry-run : vérifie la reconnaissance sans cliquer
python scripts/run_full_automation.py --method cnn --dry-run

# Automation complète (sudoku.com doit être ouvert et visible)
python scripts/run_full_automation.py --method cnn
```

> Lancer depuis **PowerShell ou CMD Windows** (pas WSL) — pyautogui requiert l'accès à l'écran.

### Benchmark des méthodes
```bash
python scripts/benchmark.py --limit 20 --methods cnn easyocr tesseract
```

---

## Entraîner le CNN

### 1. Extraire les crops de chiffres depuis les screenshots
```bash
python scripts/extract_digit_crops.py
```
Génère `data/digit_crops/labeled/1/` ... `/9/` et `data/digit_crops/unknown/`.  
Déplacer manuellement les images de `unknown/` dans le bon dossier chiffre.

### 2. Entraîner
```bash
python scripts/train_digit_cnn.py --epochs 30 --output weights/digit_cnn.pth
```

Options :
```
--data      Dossier des crops labelisés (défaut: data/digit_crops/labeled)
--epochs    Nombre d'époques              (défaut: 30)
--batch     Taille de batch               (défaut: 32)
--output    Chemin du modèle sauvegardé   (défaut: weights/digit_cnn.pth)
```

### Entraîner les modèles YOLO (Colab recommandé)
```bash
# Cellules
python scripts/train_yolo.py --data data/cells.yaml --model yolov8n.pt --epochs 50

# Grille
python scripts/train_yolo.py --data data/grid.yaml --model yolov8n.pt --epochs 30
```

---

## Docker

### Build
```bash
docker build -t sudoku-solver .
```

### Run
```bash
# Sur une image
docker run --rm -v $(pwd)/raw:/app/raw sudoku-solver \
    python scripts/run_demo.py --image raw/sudoku_0.png

# Avec les poids personnalisés
docker run --rm \
    -v $(pwd)/raw:/app/raw \
    -v $(pwd)/weights:/app/weights \
    -v $(pwd)/outputs:/app/outputs \
    sudoku-solver python scripts/run_demo.py --image raw/sudoku_0.png --method cnn
```

---

## Structure du projet

```
sudoku-solver_2/
├── src/pipeline/
│   ├── detect_grid.py       # Détection YOLO + OpenCV
│   ├── segment_cells.py     # Warp perspective + split 9×9
│   ├── recognize_digits.py  # CNN / EasyOCR / Tesseract
│   ├── solver.py            # Backtracking + MRV
│   ├── screen_capture.py    # Screenshot pyautogui
│   ├── automation.py        # Clic automatique
│   └── pipeline.py          # Orchestration
├── scripts/
│   ├── run_demo.py               # Test sur image
│   ├── run_full_automation.py    # Automation complète
│   ├── extract_digit_crops.py    # Extraction dataset CNN
│   ├── train_digit_cnn.py        # Entraînement CNN
│   ├── train_yolo.py             # Entraînement YOLO
│   └── benchmark.py              # Comparaison des méthodes
├── weights/
│   ├── best.pt              # YOLO cellules (fine-tuné)
│   └── digit_cnn.pth        # CNN chiffres (entraîné sur sudoku.com)
├── weights2/
│   └── best.pt              # YOLO grille (fine-tuné)
├── raw/                     # 100 screenshots sudoku.com
├── data/digit_crops/        # Dataset CNN (3000+ crops)
├── tests/                   # Tests unitaires
├── Dockerfile
└── requirements.txt
```

---

## Stack technique

- **Vision** : OpenCV, Ultralytics YOLO
- **Deep Learning** : PyTorch, torchvision
- **OCR** : EasyOCR, Tesseract / pytesseract
- **Automation** : pyautogui, Selenium
- **Déploiement** : Docker

---

## Contrainte fondamentale

Conformément aux exigences du projet, **le DOM HTML n'est jamais utilisé** pour localiser la grille ou identifier les chiffres. Toute la perception passe exclusivement par analyse d'image (computer vision).

L'utilisation du HTML est limitée aux interactions annexes uniquement (acceptation de cookies, etc.).
