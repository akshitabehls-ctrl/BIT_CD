import os, time, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import SimplifiedBIT
from dataset import LevirCdDataset


def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou   = inter / (union + 1e-8)
    dice  = 2 * inter / (y_true.sum() + y_pred.sum() + 1e-8)

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, iou=iou, dice=dice)


def dice_loss_from_logits(logits, targets, smooth=1.0):
    # logits: [B,2,H,W], targets: [B,H,W]
    probs = torch.softmax(logits, dim=1)[:, 1]
    targets_f = (targets == 1).float()

    inter = (probs * targets_f).sum(dim=(1,2))
    union = probs.sum(dim=(1,2)) + targets_f.sum(dim=(1,2))
    dice = (2. * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def train_main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print("Device:", device)

    train_ds = LevirCdDataset(args.data, "train", args.img_size)
    val_ds   = LevirCdDataset(args.data, "val",   args.img_size)

    nw = 2 if device.type == "cuda" else 0
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=nw)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=nw)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = SimplifiedBIT(num_classes=2).to(device)

    # CHANGED: Removed aggressive manual class weights.
    # We rely on Dice Loss + standard CE to balance training.
    ce_loss = nn.CrossEntropyLoss() 

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = 1e9
    patience = args.early_stop
    es = 0

    csv_path = os.path.join(args.ckpt_dir, "metrics.csv")
    # Write header if file doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss",
                 "acc", "prec", "rec", "f1", "iou", "dice", "lr"]
            )

    print("🚀 Starting training...")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0

        for batch in train_dl:
            a = batch["image_a"].to(device)
            b = batch["image_b"].to(device)
            m = batch["mask"].to(device).long()

            logits = model(a, b)

            # Loss combination: CE for pixel accuracy, Dice for overlap/imbalance
            loss_ce   = ce_loss(logits, m)
            loss_dice = dice_loss_from_logits(logits, m)
            loss = loss_ce + loss_dice

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item()

        tr_loss /= len(train_dl)

        # ----- Validation -----
        model.eval()
        vloss = 0.0
        preds, gts = [], []

        with torch.no_grad():
            for batch in val_dl:
                a = batch["image_a"].to(device)
                b = batch["image_b"].to(device)
                m = batch["mask"].to(device).long()

                logits = model(a, b)
                loss = ce_loss(logits, m) + dice_loss_from_logits(logits, m)
                vloss += loss.item()

                p = torch.argmax(logits, dim=1)
                preds.append(p.cpu().numpy().ravel())
                gts.append(m.cpu().numpy().ravel())

        vloss /= len(val_dl)
        
        # Metrics calculation
        y_pred = np.concatenate(preds)
        y_true = np.concatenate(gts)
        
        # Quick check to see if model is predicting anything other than 0
        unique_preds = np.unique(y_pred)
        
        metrics = compute_metrics(y_true, y_pred)
        sched.step()
        lr = opt.param_groups[0]["lr"]
 
        print(
            f"Ep {ep:03d} | TrLoss {tr_loss:.4f} | ValLoss {vloss:.4f} | "
            f"IoU {metrics['iou']:.4f} | F1 {metrics['f1']:.4f} | "
            f"Preds: {unique_preds}"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, tr_loss, vloss,
                 metrics['acc'], metrics['prec'], metrics['rec'],
                 metrics['f1'], metrics['iou'], metrics['dice'], lr]
            )

        if vloss < best_val:
            best_val = vloss
            es = 0
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, args.model_name))
            print("✅ Saved best model")
        else:
            es += 1
            if es >= patience:
                print("⏹ Early stopping triggered.")
                break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",     type=str,   default="/home/23ucs622/BIT_CD/LEVIR-CD-256")
    p.add_argument("--ckpt_dir", type=str,   default="/home/23ucs622/BIT_CD/checkpoints")
    # CHANGE 1: Increase max epochs
    p.add_argument("--epochs",   type=int,   default=200)
    
    # CHANGE 2: Increase batch size for faster training (if GPU allows)
    p.add_argument("--bs",       type=int,   default=16) 
    
    p.add_argument("--lr", type=float, default=5e-5) # Reduced by half for stability
    p.add_argument("--wd",       type=float, default=1e-6)
    p.add_argument("--img_size", type=int,   default=256)
    p.add_argument("--model_name", type=str, default="best_bit.pth")
    
    # CHANGE 3: Increase patience so it doesn't stop too quickly
    p.add_argument("--early_stop", type=int, default=30)
    args = p.parse_args()
    train_main(args)