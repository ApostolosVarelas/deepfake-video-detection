import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.constants import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block that reweights the channels of a
    feature map.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16
    ) -> None:
        """
        Initialize the SE block.

        Args:
            channels (int): Number of channels of the feature map.
            reduction (int): Reduction ratio of the bottleneck.

        Returns:
            None
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale each channel of the feature map by its learned weight.

        Args:
            x (torch.Tensor): Feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: The reweighted feature map.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class XceptionSE(nn.Module):
    """
    Xception backbone with an SE block on its final feature map and a
    linear classifier for real/fake.
    """

    def __init__(
        self,
        base: nn.Module
    ) -> None:
        """
        Initialize the model.

        Args:
            base (nn.Module): The timm Xception backbone.

        Returns:
            None
        """
        super().__init__()
        self.base = base
        self.se = SEBlock(channels=2048, reduction=16)
        self.fc = nn.Linear(2048, 2)


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify a batch of images.

        Args:
            x (torch.Tensor): Images of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 2).
        """
        f = self.base.forward_features(x)
        f = self.se(f)
        p = self.base.global_pool(f)
        p = torch.flatten(p, 1)
        return self.fc(p)


def _validate_video_level(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, Optional[str], Dict[str, Dict[str, Any]]]:
    """
    Validate the model at video level with multiple aggregation methods.

    The frame predictions of each video are combined with 6 methods
    (mean, max, min, median and trimmed mean of the fake probability,
    and the percentage of fake votes), and the best one is chosen by F1.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): Loader of the validation frames.
        device (torch.device): Device to run the model on.

    Returns:
        Tuple[float, float, float, Optional[str],
        Dict[str, Dict[str, Any]]]: The frame-level validation loss, the
            accuracy and F1 of the best method, the name of the best
            method, and the metrics of every method.
    """
    model.eval()
    video_dict = defaultdict(lambda: {"labels": [], "probs": [], "preds": []})
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0

    with torch.no_grad():
        for images, labels, video_paths in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            for i, vp in enumerate(video_paths):
                video_dict[vp]["labels"].append(labels[i].item())
                video_dict[vp]["probs"].append(probs[i].cpu().numpy())
                video_dict[vp]["preds"].append(preds[i].item())

    val_loss = running_loss / total if total else 0.0

    methods = {
        'mean_prob': lambda x: np.mean([p[1] for p in x]),
        'max_prob': lambda x: np.max([p[1] for p in x]),
        'min_prob': lambda x: np.min([p[1] for p in x]),
        'median_prob': lambda x: np.median([p[1] for p in x]),
        'trimmed_mean_prob': lambda x: (
            np.mean(sorted([p[1] for p in x])[1:-1])
            if len(x) > 2 else np.mean([p[1] for p in x])
        ),
        'vote_percentage': lambda x: np.mean(x)
    }

    metrics_all = {}
    best_f1 = -1.0
    best_acc = 0.0
    best_method = None

    for name, agg in methods.items():
        y_true, y_pred, scores = [], [], []
        for info in video_dict.values():
            if not info["labels"]:
                continue
            true = info["labels"][0]
            score = agg(
                info["preds"] if name == 'vote_percentage' else info["probs"]
            )
            pred = 1 if score >= 0.5 else 0
            y_true.append(true)
            y_pred.append(pred)
            scores.append(score)

        if not y_true:
            continue

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        y_bin = np.array(y_true)
        fpr, tpr, _ = roc_curve(y_bin, scores)
        eer = fpr[np.nanargmin(np.abs((1 - tpr) - fpr))]
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(y_bin, scores)

        metrics_all[name] = {
            'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1,
            'mcc': mcc, 'auc': roc_auc,
            'ap': ap, 'eer': eer
        }

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_method = name

    return val_loss, best_acc, best_f1, best_method, metrics_all


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_folder: str
) -> Tuple[nn.Module, Optional[str]]:
    """
    Train the model with mixed precision, early stopping and a
    learning rate scheduler, and save the training curves.

    The weights with the lowest validation loss are saved. If a saved
    model already exists in the output folder, it is loaded instead of
    training.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Loader of the training frames.
        val_loader (DataLoader): Loader of the validation frames.
        device (torch.device): Device to train on.
        output_folder (str): Folder for the weights, metrics and plots.

    Returns:
        Tuple[nn.Module, Optional[str]]: The trained model and the best
            aggregation method at the best epoch (None when existing
            weights were loaded).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    scaler = GradScaler()
    early_stop = 5
    no_improve = 0
    best_val_loss = float('inf')
    model_path = os.path.join(output_folder, "Xception.pth")

    if os.path.exists(model_path):
        logging.warning("Xception already trained; loading.")
        model.load_state_dict(torch.load(model_path))
        return model, None

    train_losses, train_accs = [], []
    val_losses, val_accs, val_f1s = [], [], []

    for epoch in range(NUM_EPOCHS):
        # --- train ---
        model.train()
        running, corr, tot = 0.0, 0, 0
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False
        )
        for imgs, lbls, _ in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with autocast():
                outs = model(imgs)
                loss = criterion(outs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * imgs.size(0)
            preds = outs.argmax(dim=1)
            corr += (preds == lbls).sum().item()
            tot += lbls.size(0)
            pbar.set_postfix(
                loss=f"{running/tot:.4f}", acc=f"{corr/tot:.4f}"
            )

        train_losses.append(running / tot)
        train_accs.append(corr / tot)

        # --- validate ---
        val_loss, val_acc, val_f1, best_method, metrics_all = (
            _validate_video_level(model, val_loader, device)
        )
        scheduler.step(val_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        logging.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train loss {train_losses[-1]:.4f}, "
            f"acc {train_accs[-1]:.4f} | "
            f"Val loss {val_loss:.4f}, acc {val_acc:.4f}, "
            f"f1 {val_f1:.4f} ({best_method})"
        )

        out_txt = os.path.join(
            output_folder, "Xception_validation_metrics.txt"
        )
        with open(out_txt, "a") as f:
            f.write(f"\nEpoch {epoch+1}\n")
            for m, met in metrics_all.items():
                f.write(
                    f"{m} -> Acc: {met['accuracy']:.4f}, "
                    f"Prec: {met['precision']:.4f}, "
                    f"Rec: {met['recall']:.4f}, "
                    f"F1: {met['f1']:.4f}, MCC: {met['mcc']:.4f}, "
                    f"AUC: {met['auc']:.4f}, AP: {met['ap']:.4f}, "
                    f"EER: {met['eer']:.4f}\n"
                )
            f.write(f"Best: {best_method}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_method_overall = best_method
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved new best model (loss={val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= early_stop:
                logging.info("Early stopping triggered.")
                break

    # ---- PLOT TRAINING CURVES ----
    plt.figure()
    plt.plot(
        range(1, len(train_losses) + 1), train_losses, label='Train Loss'
    )
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_loss.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Acc')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(val_f1s) + 1), val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'validation_f1.png'))
    plt.close()

    # ---- PLOT F1 vs LOSS ----
    plt.figure()
    plt.plot(val_losses, val_f1s, marker='o')
    plt.xlabel('Validation Loss')
    plt.ylabel('F1 Score')
    plt.title('F1 vs Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'f1_vs_loss.png'))
    plt.close()

    return model, best_method_overall


def xception_main(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    device: torch.device,
    output_folder: str
) -> Tuple[DataLoader, nn.Module, Optional[str]]:
    """
    Build and train an Xception + SE deepfake classifier.

    Starts from ImageNet weights and fine-tunes blocks 6-8 and the last
    convolution (conv4, bn4), plus the SE block and the classifier.

    Args:
        train_ds (Dataset): Training frames.
        val_ds (Dataset): Validation frames.
        test_ds (Dataset): Test frames.
        device (torch.device): Device to train on.
        output_folder (str): Folder for the weights, metrics and plots.

    Returns:
        Tuple[DataLoader, nn.Module, Optional[str]]: The test loader,
            the trained model and the best aggregation method.
    """
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    backbone = timm.create_model('xception', pretrained=True)

    for p in backbone.parameters():
        p.requires_grad = False

    unfreeze_modules = ['block6', 'block7', 'block8', 'conv4', 'bn4']
    for name, module in backbone.named_children():
        if name in unfreeze_modules:
            for p in module.parameters():
                p.requires_grad = True

    model = XceptionSE(backbone).to(device)

    model, best_method = _train_model(
        model, train_loader, val_loader, device, output_folder
    )
    return test_loader, model, best_method
