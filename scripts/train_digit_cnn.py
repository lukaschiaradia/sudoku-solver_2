"""
Phase 2 : entraîne un CNN sur les crops extraits par extract_digit_crops.py

Utilisation :
    python scripts/train_digit_cnn.py
    python scripts/train_digit_cnn.py --data data/digit_crops/labeled --epochs 30 --output weights/digit_cnn.pth
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 9),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_transforms(augment=True):
    base = [
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    if augment:
        base = [
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=5, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    return transforms.Compose(base)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    dataset = datasets.ImageFolder(args.data, transform=get_transforms(augment=True))

    # classes doivent être 1-9 (dossiers nommés "1", "2", ..., "9")
    print(f'Classes trouvées : {dataset.classes}')
    assert all(c in [str(i) for i in range(1, 10)] for c in dataset.classes), \
        'Les dossiers doivent être nommés 1, 2, ..., 9'

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    val_ds.dataset.transform = get_transforms(augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    print(f'Train: {train_size} samples, Val: {val_size} samples')

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

        val_acc = correct / total if total > 0 else 0
        scheduler.step()
        print(f'Epoch {epoch:3d}/{args.epochs} | loss={total_loss/len(train_loader):.4f} | val_acc={val_acc:.3f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'classes': dataset.classes}, args.output)
            print(f'  → Meilleur modèle sauvegardé ({val_acc:.3f})')

    print(f'\nEntraînement terminé. Meilleure val_acc: {best_val_acc:.3f}')
    print(f'Modèle sauvegardé dans {args.output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/digit_crops/labeled')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', default='weights/digit_cnn.pth')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
