"""Simple PyTorch training script for binary Crested Myna classifier using ResNet.

Usage example:
  python train_pytorch.py --data_dir ./data --arch resnet50 --epochs 10 --batch_size 32 --output models/crested_myna_model.pth

The script expects data organized as ImageFolder (each class in a subfolder).
It saves a checkpoint dict with keys: arch, state_dict, classes (class_to_idx mapping).
Supports: resnet18, resnet34, resnet50, resnet101
"""
import argparse
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet18":
        model = models.resnet18(pretrained=True)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif arch == "resnet101":
        model = models.resnet101(pretrained=True)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)
    elif arch in ("efficientnet_b0", "efficientnetb0"):
        try:
            model = models.efficientnet_b0(pretrained=True)
            in_f = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_f, num_classes)
        except Exception:
            raise RuntimeError("EfficientNet not available in this torchvision version")
    else:
        raise ValueError(f"Unsupported arch: {arch}. Choose from: resnet18, resnet34, resnet50, resnet101, efficientnet_b0")
    return model


def train(args):
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir {data_dir} not found")

    # Data transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(str(data_dir))
    # split into train/val
    n = len(full_dataset)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])
    # assign transforms
    train_set.dataset.transform = train_transforms
    val_set.dataset.transform = val_transforms

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model = build_model(args.arch, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running += loss.item() * imgs.size(0)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        train_loss = running / total
        train_acc = correct / total

        # validation
        model.eval()
        v_running = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                v_running += loss.item() * imgs.size(0)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)

        val_loss = v_running / v_total if v_total else 0.0
        val_acc = v_correct / v_total if v_total else 0.0

        print(f"Epoch {epoch}/{args.epochs} — train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} dt: {time.time()-t0:.1f}s")

        # simple checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
            }
            torch.save(ckpt, str(out_path))
            # also save human readable mapping
            with open(str(out_path) + ".json", "w") as f:
                json.dump({"class_to_idx": class_to_idx}, f)
            print(f"Saved best model to {out_path} (val_acc={best_val_acc:.4f})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--arch", default="resnet50", help="resnet18, resnet34, resnet50, resnet101, or efficientnet_b0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--output", default="models/crested_myna_model.pth")
    p.add_argument("--use_cuda", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
