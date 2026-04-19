# Sudoku Solver 2

Projet de reconnaissance visuelle et résolution automatique de Sudoku.

## Objectif

- Détecter une grille de Sudoku uniquement par vision (captures d'écran).
- Segmenter les 81 cases de la grille.
- Reconnaître les chiffres présents dans les cases.
- Résoudre la grille avec un solveur déterministe.
- Préparer l'automatisation de saisie locale.

## Structure du dépôt

- `raw/` : captures d'écran réelles de grilles Sudoku.
- `weights/best.pt` : modèle YOLO fourni pour la détection.
- `src/pipeline/` : modules de détection, segmentation, OCR, et solveur.
- `scripts/run_demo.py` : démonstration de pipeline sur image.
- `tests/` : tests unitaires de base.
- `requirements.txt` : dépendances Python.
- `Dockerfile` : image de base pour exécution locale.

## Installation

1. Crée un environnement Python 3.12 :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Exécute le démonstrateur sur une image du dossier `raw/` :
   ```bash
   python3 scripts/run_demo.py --image raw/sudoku_0.png --weights weights/best.pt
   ```

3. Si tu veux utiliser le nouveau modèle de détection de grille disponible dans `weights2/best.pt` :
   ```bash
   python3 scripts/run_demo.py --image raw/sudoku_0.png --weights weights/best.pt --grid-weights weights2/best.pt --grid-conf 0.1
   ```

## Entraînement d'un modèle YOLOv8 sur Colab

Pour entraîner le modèle sur Colab avec ton dataset Roboflow :

1. Télécharge le dataset YOLO dans Colab et vérifie le dossier :
   ```python
   !pip install roboflow ultralytics

   from roboflow import Roboflow
   rf = Roboflow(api_key="50DYirGbCOmEfYQRzJbw")
   project = rf.workspace("lukass-workspace-2myrq").project("sudoku-grid2")
   version = project.version(1)
   dataset = version.download("yolov8")
   print(dataset.location)
   !find "{dataset.location}" -maxdepth 2 -type f
   ```

2. Entraîne le modèle YOLOv8 sur le dataset :
   ```bash
   !python3 -m ultralytics yolo task=detect mode=train model=yolov8n.pt data="/content/sudoku-grid2-1/data.yaml" epochs=20 imgsz=640 batch=16 project=/content/runs/train name=sudoku_grid
   ```

3. Télécharge le poids entraîné :
   ```python
   from google.colab import files
   files.download('/content/runs/train/sudoku_grid/weights/best.pt')
   ```

4. Place ensuite `best.pt` dans `weights/` de ton dépôt, par exemple `weights/grid.pt`.

## Entraînement local

Si tu veux entraîner localement à partir de ton dataset téléchargé :

```bash
python3 scripts/train_yolo.py --data weights/data.yaml --model yolov8n.pt --epochs 20 --imgsz 640 --batch 16 --project runs/train --name sudoku_grid
```

Le modèle final sera disponible dans `runs/train/sudoku_grid/weights/best.pt`.

## Usage

- `scripts/run_demo.py` lance le pipeline complet : détection, extraction, reconnaissance et résolution.
- `scripts/run_full_automation.py` capture l'écran principal, détecte la grille, résout le Sudoku et remplit les cases avec `pyautogui`.
- `scripts/export_templates.py` génère un dossier `data/templates` avec des modèles de chiffres réutilisables.
- Les résultats sont écrits dans `outputs/predictions/result.json` ou `outputs/predictions/full_automation.json`.

### Exemple d'automatisation écran

```bash
python3 scripts/run_full_automation.py
```

Sans options supplémentaires, le script utilise les poids par défaut et capture l'écran.

```bash
python3 scripts/run_full_automation.py --weights weights/best.pt --grid-weights weights2/best.pt --grid-conf 0.01
```

### Générer des templates de chiffres

Pour créer un dossier de templates réutilisables :

```bash
python3 scripts/export_templates.py
```

Cela crée `data/templates/1_1.png`, `data/templates/1_2.png`, etc. et améliore la reconnaissance par `--method template`.

Pour obtenir des fichiers de debug et inspecter la grille détectée :

```bash
python3 scripts/run_full_automation.py --weights weights/best.pt --grid-weights weights2/best.pt --grid-conf 0.01 --debug --debug-dir outputs/debug --dry-run
```

Ce script :
- prend un screenshot de l'écran principal,
- détecte la grille entière avec `weights2/best.pt`,
- lit les chiffres par OCR,
- résout le Sudoku,
- remplit les cases vides sur l'écran en cliquant.

> Note : pour la saisie automatique avec `pyautogui`, exécute ce script depuis un terminal Windows (PowerShell ou CMD) et non depuis WSL, car la capture d'écran et le clic nécessitent l'accès à l'écran Windows.

## Provenance des données

- Dossier `raw/` : captures réelles fournies par l'utilisateur.
- Le projet est construit pour fonctionner avec des captures de `sudoku.com`.

## Notes

- La perception est conçue pour être uniquement basée sur image.
- `src/pipeline/automation.py` contient une base Selenium pour la saisie, mais la logique de sélection de cellule doit être adaptée au DOM du site.
- Le solveur est un backtracking déterministe.

## Prochaines étapes

1. Évaluer la performance du modèle YOLO existant sur les images `raw/`.
2. Ajouter un second canal de reconnaissance (`template matching` ou CNN).
3. Construire un jeu de validation et produire des métriques objectives.
4. Finaliser l'automatisation de la saisie avec Selenium localement.
