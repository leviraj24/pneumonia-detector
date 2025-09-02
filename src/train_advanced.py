# src/train_advanced.py
import os, time, argparse, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import models
from sklearn.metrics import roc_auc_score, confusion_matrix
from datasets import get_dataloaders, get_class_weights
from torchvision import datasets as tvdatasets

# ---------- Utilities ----------
def set_fast_flags():
    torch.backends.cudnn.benchmark = True  # autotune best conv kernels

def to_channels_last(model):
    # ConvNets get memory bandwidth wins on many GPUs with channels_last
    return model.to(memory_format=torch.channels_last)

def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def softmax_probs(logits):
    return torch.softmax(logits, dim=1)

def compute_metrics(all_logits, all_labels, pos_index=1):
    with torch.no_grad():
        probs = softmax_probs(all_logits)[:, pos_index].cpu().numpy()
        y_true = all_labels.cpu().numpy()
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = float("nan")
        preds = (probs >= 0.5).astype("int64")
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
        acc = (tp + tn) / max(tn + fp + fn + tp, 1)
        sensitivity = tp / max(tp + fn, 1)    # recall for pneumonia
        specificity = tn / max(tn + fp, 1)
        precision   = tp / max(tp + fp, 1)
    return {
        "auc": auc, "acc": acc,
        "sensitivity": sensitivity, "specificity": specificity,
        "precision": precision
    }

def freeze_backbone(model, trainable_head_names=("classifier", "fc")):
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze typical head(s)
    for name in trainable_head_names:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def export_torchscript(model, img_size, out_path, device):
    model.eval()
    ex = torch.randn(1, 3, img_size, img_size, device=device).to(memory_format=torch.channels_last)
    with torch.no_grad(), autocast():
        scripted = torch.jit.trace(model, ex, strict=False)
    torch.jit.save(scripted, out_path)
    print(f"🧪 TorchScript saved → {out_path}")

# ---------- Model builder ----------
def build_model(name: str, num_classes: int, pretrained=True):
    name = name.lower()
    if name in ("mobilenet_v2", "mobilenetv2"):
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.last_channel, num_classes)
        return m
    if name in ("resnet18",):
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name in ("efficientnet_b0", "effnet_b0", "efficientnet"):
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    if name in ("efficientnet_b3", "effnet_b3"):
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {name}")

# ---------- Train / Validate ----------
def run_epoch(model, loader, criterion, optimizer, device, scaler, use_amp=True):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
    return running_loss / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        all_logits.append(logits.float().detach().cpu())
        all_labels.append(y.detach().cpu())
    if not all_logits:  # empty loader guard
        return {"val_loss": float("nan"), "auc": float("nan"), "acc": float("nan"),
                "sensitivity": float("nan"), "specificity": float("nan"), "precision": float("nan")}
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels, pos_index=1)
    metrics["val_loss"] = total_loss / max(len(loader), 1)
    return metrics

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--model", default="efficientnet_b0",
                   choices=["mobilenet_v2", "resnet18", "efficientnet_b0", "efficientnet_b3"])
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-head", type=int, default=5, help="freeze backbone; train head")
    p.add_argument("--epochs-ft", type=int, default=10, help="fine-tune all layers")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-ft", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--compile", action="store_true", help="use torch.compile for potential speedup")
    p.add_argument("--early-stop", type=int, default=3, help="patience on val AUC")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--export-ts", action="store_true", help="export TorchScript after training")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_fast_flags()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device} | CUDA: {torch.cuda.is_available()}")

    # Data
    train_loader, val_loader, classes = get_dataloaders(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        img_size=args.img_size,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2
    )
    print(f"Classes: {classes}")
    train_ds = tvdatasets.ImageFolder(os.path.join(args.data, "train"))
    class_weights = get_class_weights(train_ds).to(device)

    # Model
    model = build_model(args.model, num_classes=2, pretrained=True).to(device)
    model = to_channels_last(model)

    # (optional) torch.compile in PyTorch 2.8+
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, dynamic=False)
        print("⚙️ Enabled torch.compile")

    # Label smoothing helps generalization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    scaler = GradScaler(enabled=not args.no_amp)

    # ---- Phase 1: freeze backbone, train head ----
    freeze_backbone(model)
    opt_head = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr_head, weight_decay=args.weight_decay)
    sch_head = optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=args.epochs_head, eta_min=1e-6)

    best_auc = -math.inf
    best_path = os.path.join(args.out_dir, f"best_{args.model}.pth")

    print("\n🚀 Phase 1: Training classifier head (frozen backbone)")
    patience = args.early_stop
    for epoch in range(1, args.epochs_head + 1):
        tr_loss = run_epoch(model, train_loader, criterion, opt_head, device, scaler, use_amp=not args.no_amp)
        metrics = evaluate(model, val_loader, criterion, device)
        sch_head.step()

        print(f"[Head {epoch:02d}/{args.epochs_head}] "
              f"train_loss={tr_loss:.4f} | val_loss={metrics['val_loss']:.4f} "
              f"| AUC={metrics['auc']:.4f} | ACC={metrics['acc']:.3f} "
              f"| SEN={metrics['sensitivity']:.3f} | SPE={metrics['specificity']:.3f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]; patience = args.early_stop
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best (AUC {best_auc:.4f}) → {best_path}")
        else:
            patience -= 1
            if patience <= 0:
                print("  ⏹️ Early-stop triggered in head phase (AUC plateau).")
                break

    # ---- Phase 2: unfreeze & fine-tune all ----
    unfreeze_all(model)
    opt_ft = optim.AdamW(model.parameters(), lr=args.lr_ft, weight_decay=args.weight_decay)
    sch_ft = optim.lr_scheduler.OneCycleLR(
        opt_ft, max_lr=args.lr_ft, steps_per_epoch=max(len(train_loader),1),
        epochs=args.epochs_ft, pct_start=0.3
    )

    print("\n🎯 Phase 2: Fine-tuning all layers")
    patience = args.early_stop
    for epoch in range(1, args.epochs_ft + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt_ft.zero_grad(set_to_none=True)
            if not args.no_amp:
                with autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt_ft)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt_ft.step()
            running += loss.item()
            sch_ft.step()
        tr_loss = running / max(len(train_loader),1)

        metrics = evaluate(model, val_loader, criterion, device)
        print(f"[FT  {epoch:02d}/{args.epochs_ft}] "
              f"train_loss={tr_loss:.4f} | val_loss={metrics['val_loss']:.4f} "
              f"| AUC={metrics['auc']:.4f} | ACC={metrics['acc']:.3f} "
              f"| SEN={metrics['sensitivity']:.3f} | SPE={metrics['specificity']:.3f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]; patience = args.early_stop
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best (AUC {best_auc:.4f}) → {best_path}")
        else:
            patience -= 1
            if patience <= 0:
                print("  ⏹️ Early-stop triggered in fine-tune phase (AUC plateau).")
                break

    print(f"\n🏁 Done. Best AUC: {best_auc:.4f}. Saved → {best_path}")

    # Optional TorchScript export (fast offline load)
    if args.export_ts:
        # Rebuild same model, load best weights, export
        model2 = build_model(args.model, num_classes=2, pretrained=False).to(device)
        model2 = to_channels_last(model2)
        model2.load_state_dict(torch.load(best_path, map_location=device))
        export_path = os.path.join(args.out_dir, f"{args.model}_ts.pt")
        export_torchscript(model2, args.img_size, export_path, device)

if __name__ == "__main__":
    main()
