import argparse
import datetime
import json
import os
import random
import time
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--train-subdir", type=str, default=None)
    p.add_argument("--val-subdir", type=str, default=None)
    p.add_argument("--test-subdir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="runs/resnet50")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--split-train", type=float, default=0.8)
    p.add_argument("--split-val", type=float, default=0.1)
    p.add_argument("--split-test", type=float, default=0.1)
    return p.parse_args()

def is_image(path: Path):
    return path.suffix.lower() in IMAGE_EXTS

def enumerate_dataset(root: Path):
    classes = []
    items = {}
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        cls = d.name
        classes.append(cls)
        cnt = 0
        for dirpath, _, filenames in os.walk(d):
            for f in filenames:
                if is_image(Path(f)):
                    cnt += 1
        items[cls] = cnt
    return classes, items

def _auto_descend(root: Path):
    cur = root
    while True:
        dirs = [p for p in cur.iterdir() if p.is_dir()]
        files = [p for p in cur.iterdir() if p.is_file() and is_image(p)]
        if len(dirs) == 1 and len(files) == 0:
            cur = dirs[0]
            continue
        return cur

def dry_run_only(args):
    root = Path(args.data_dir)
    if not root.exists():
        print(f"数据路径不存在: {root}")
        sys.exit(1)
    root = _auto_descend(root)
    if args.train_subdir or args.val_subdir or args.test_subdir:
        subdirs = [args.train_subdir, args.val_subdir, args.test_subdir]
        for name in subdirs:
            if name:
                p = (root / name)
                if not p.exists():
                    print(f"子目录不存在: {p}")
                    sys.exit(1)
                classes, counts = enumerate_dataset(p)
                print(f"{name} 类别数: {len(classes)}")
                for c in classes:
                    print(f"{name}/{c}: {counts.get(c, 0)} 张")
    else:
        classes, counts = enumerate_dataset(root)
        if len(classes) == 0:
            print("未发现类别子目录，请确保数据按类别分目录存放")
            sys.exit(1)
        print(f"类别数: {len(classes)}")
        for c in classes:
            print(f"{c}: {counts.get(c, 0)} 张")
    print("干运行检查通过")

def build_dataloaders(args):
    import torch
    from torch.utils.data import DataLoader, random_split, Subset
    from torchvision import datasets, transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    data_root = _auto_descend(Path(args.data_dir))

    def find_splits(root):
        names = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
        train = names.get("train")
        val = names.get("val") or names.get("valid") or names.get("validation")
        test = names.get("test")
        return train, val, test

    if args.train_subdir or args.val_subdir or args.test_subdir:
        train_root = data_root / (args.train_subdir or "train")
        val_root = data_root / (args.val_subdir or "val")
        if not val_root.exists():
            alt = data_root / "valid"
            val_root = alt if alt.exists() else val_root
        test_root = data_root / (args.test_subdir or "test")
        train_ds = datasets.ImageFolder(str(train_root), transform=train_tf)
        val_ds = datasets.ImageFolder(str(val_root), transform=eval_tf)
        test_ds = datasets.ImageFolder(str(test_root), transform=eval_tf) if test_root.exists() else val_ds
    else:
        tr, vl, ts = find_splits(data_root)
        if tr is not None and vl is not None:
            train_ds = datasets.ImageFolder(str(tr), transform=train_tf)
            val_ds = datasets.ImageFolder(str(vl), transform=eval_tf)
            test_ds = datasets.ImageFolder(str(ts), transform=eval_tf) if ts is not None else val_ds
        else:
            full_ds = datasets.ImageFolder(str(data_root), transform=train_tf)
            n = len(full_ds)
            n_train = int(args.split_train * n)
            n_val = int(args.split_val * n)
            n_test = n - n_train - n_val
            generator = torch.Generator().manual_seed(args.seed)
            train_subset, val_subset, test_subset = random_split(full_ds, [n_train, n_val, n_test], generator=generator)
            val_subset.dataset.transform = eval_tf
            test_subset.dataset.transform = eval_tf
            train_ds, val_ds, test_ds = train_subset, val_subset, test_subset

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def build_model(num_classes, pretrained):
    import torch
    from torchvision import models
    if pretrained:
        try:
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            m = models.resnet50(weights=None)
    else:
        m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = torch.nn.Linear(in_features, num_classes)
    return m

def evaluate(model, loader, device, num_classes):
    import torch
    model.eval()
    total = 0
    correct = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1
    acc = correct / max(total, 1)
    per_class_acc = (cm.diag().float() / cm.sum(dim=1).clamp(min=1).float()).tolist()
    return acc, cm, per_class_acc

def train(args):
    import torch
    from torch import nn, optim
    from torchvision import datasets

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_loader, val_loader, test_loader = build_dataloaders(args)
    num_classes = len(train_loader.dataset.dataset.classes) if hasattr(train_loader.dataset, "dataset") else len(train_loader.dataset.classes)
    model = build_model(num_classes, args.pretrained)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Fine-tuning: unfreeze some layers
    for name, param in model.named_parameters():
        if "layer4" in name or "layer3" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Ensure the classifier layer is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    # Create parameter groups for optimizer
    params_to_update = [
        {'params': model.layer3.parameters(), 'lr': args.lr * 0.1},
        {'params': model.layer4.parameters(), 'lr': args.lr * 0.1},
        {'params': model.fc.parameters(), 'lr': args.lr}
    ]

    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        val_acc, _, _ = evaluate(model, val_loader, device, num_classes)
        scheduler.step()

        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "classes": train_loader.dataset.dataset.classes if hasattr(train_loader.dataset, "dataset") else train_loader.dataset.classes,
        }, os.path.join(args.output_dir, "last.pth"))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pth"))
            # Save class names
            with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
                json.dump(train_loader.dataset.dataset.classes if hasattr(train_loader.dataset, "dataset") else train_loader.dataset.classes, f)

    test_acc, cm, per_class_acc = evaluate(model, test_loader, device, num_classes)
    print(f"Test accuracy: {test_acc:.4f}")
    print("Per-class accuracy:")
    for i, a in enumerate(per_class_acc):
        print(f"class_{i}: {a:.4f}")
    print("Confusion matrix:")
    print(cm.cpu().numpy())

def main():
    args = parse_args()
    if args.dry_run:
        dry_run_only(args)
        return
    train(args)

if __name__ == "__main__":
    main()