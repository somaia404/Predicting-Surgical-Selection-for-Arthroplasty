
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# --- Metrics aggregation --------------------------------------------------
def aggregate_metrics(metrics_per_fold: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for k, vals in metrics_per_fold.items():
        arr = np.array(vals, dtype=float) if len(vals) else np.array([np.nan])
        summary[k] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "n_folds": int(len(arr)),
        }
    return summary

# --- Plotting helpers -----------------------------------------------------
def plot_confusion_matrix(cm, title: str = "Confusion Matrix"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.show()

def plot_roc_curve_from_probs(y_true_all, prob_pos_all, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true_all, prob_pos_all)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(title); plt.legend(loc="lower right")
    plt.show()

def plot_training_loss(train_losses: List[float], title: str = "Training loss over epochs"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
    plt.grid(True); plt.legend(); plt.show()
