import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, average_precision_score, classification_report
import pandas as pd

def plot_confusion(cm, title="Confusion Matrix"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No MCID","Yes MCID"],
                yticklabels=["No MCID","Yes MCID"])
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

def plot_roc(y_true, probs, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.title(title); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.show()
    return auc

def summarize_fold(y_true, probs, thr, model_name):
    y_pred = (np.array(probs) >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    ap = average_precision_score(y_true, probs)
    cr = classification_report(y_true, y_pred, target_names=["No MCID","Yes MCID"], output_dict=True)
    auc = roc_auc_score(y_true, probs)
    return {"auc": float(auc), "ap": float(ap), "thr": float(thr), "cm": cm, "report": cr, "model": model_name}

def results_table(fold_metrics):
    rows = []
    for i, m in enumerate(fold_metrics, start=1):
        rows.append({
            "Fold": i,
            "ROC-AUC": round(m["auc"], 3),
            "AUPRC": round(m["ap"], 3),
            "Best Thr": round(m["thr"], 3),
            "F1 (No MCID)": round(m["report"]["No MCID"]["f1-score"], 3),
            "F1 (Yes MCID)": round(m["report"]["Yes MCID"]["f1-score"], 3),
            "Accuracy": round(m["report"]["accuracy"], 3),
        })
    return pd.DataFrame(rows)
