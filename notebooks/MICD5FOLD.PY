# !pip install -q transformers torch scikit-learn tqdm matplotlib seaborn

# Cell 2: Imports & helper plotting (PATCHED: no AdamW import from transformers)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, average_precision_score
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ---- Helper plotting functions (added for script completeness)
def plot_confusion(cm, title="Confusion Matrix"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No MCID","Yes MCID"],
                yticklabels=["No MCID","Yes MCID"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_roc(y_true, probs, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return auc

# Cell 3: Load MCID dataset 
csv_file_path = "/content/hip_radiology_reports_with_MCID.csv"
if not os.path.exists(csv_file_path):
    raise FileNotFoundError("Upload your CSV to Colab and set csv_file_path accordingly.")

df = pd.read_csv(csv_file_path)

assert "Interpretation" in df.columns and "MCID" in df.columns, \
    "CSV must have 'Interpretation' and 'MCID' columns."

# Clean + labels
df["Interpretation"] = df["Interpretation"].fillna("").astype(str)
df = df[df["Interpretation"].str.len() >= 10].reset_index(drop=True)
df["label"] = df["MCID"].map({"No": 0, "Yes": 1}).astype(int)

texts = df["Interpretation"].tolist()
labels = df["label"].tolist()

print("Dataset size:", len(df))
print("Class counts:", df["label"].value_counts().to_dict())

# Cell 4: Dataset & fold training/eval with class weights + sampler (PATCHED)
class TxtDs(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=192):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def train_eval_fold(
    model_name, X_tr, y_tr, X_va, y_va,
    epochs=6, lr=2e-5, batch_size=16, max_len=192
):
    """
    Robust MCID classifier for one CV fold:
      • Class-weighted loss (inverse frequency)
      • WeightedRandomSampler (balanced mini-batches)
      • Best-epoch restore by validation AUROC
      • Threshold tuning via Youden’s J
      • Confusion Matrix + ROC plots
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ---- class weights & weighted sampler
    counts = Counter(y_tr)
    n0, n1 = counts.get(0, 1), counts.get(1, 1)
    w0, w1 = 1.0 / n0, 1.0 / n1
    class_weights = torch.tensor([w0, w1], dtype=torch.float)

    sample_weights = torch.tensor([w0 if y == 0 else w1 for y in y_tr], dtype=torch.float)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # ---- datasets & loaders
    train_ds = TxtDs(X_tr, y_tr, tokenizer, max_len=max_len)
    val_ds   = TxtDs(X_va, y_va, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- training setup (PATCH: use torch.optim.AdamW)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_auc, best_state = -1.0, None

    # ---- train epochs with quick val AUC
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y    = batch["labels"].to(device)

            logits = model(ids, attention_mask=mask).logits
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()

        # quick validation AUROC
        model.eval()
        probs_v, y_v = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y    = batch["labels"].cpu().numpy()
                p1   = torch.softmax(model(ids, attention_mask=mask).logits, dim=1)[:, 1].cpu().numpy()
                probs_v.extend(p1); y_v.extend(y)

        auc = roc_auc_score(y_v, probs_v)
        print(f"  Train loss: {running/len(train_loader):.4f} | Val AUROC: {auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- restore best epoch
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final eval + threshold tuning (Youden's J)
    model.eval()
    probs, y_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y    = batch["labels"].cpu().numpy()
            p1   = torch.softmax(model(ids, attention_mask=mask).logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p1); y_true.extend(y)

    fpr, tpr, thr = roc_curve(y_true, probs)
    youden_idx = int(np.argmax(tpr - fpr))
    thr_star   = float(thr[youden_idx]) if len(thr) else 0.5
    y_pred     = (np.array(probs) >= thr_star).astype(int)

    # ---- reports + plots
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, f"Confusion Matrix – {model_name}")
    auc_final = plot_roc(y_true, probs, f"ROC – {model_name}")

    cr = classification_report(y_true, y_pred, target_names=["No MCID","Yes MCID"], output_dict=True)
    ap = average_precision_score(y_true, probs)

    return {"auc": auc_final, "ap": ap, "thr": thr_star, "report": cr, "cm": cm}

# Cell 5: Run Stratified 5-fold CV
model_name = "UFNLP/gatortron-base"   # or "roberta-base"
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
fold_id = 1
for tr_idx, va_idx in skf.split(texts, labels):
    print(f"\n====== Fold {fold_id} ======")
    X_tr = [texts[i] for i in tr_idx]; y_tr = [labels[i] for i in tr_idx]
    X_va = [texts[i] for i in va_idx]; y_va = [labels[i] for i in va_idx]
    m = train_eval_fold(model_name, X_tr, y_tr, X_va, y_va,
                        epochs=6, lr=2e-5, batch_size=16, max_len=192)
    fold_metrics.append(m); fold_id += 1

print("\nPer-fold AUC:", [round(float(m['auc']), 3) for m in fold_metrics])
print("Mean AUC:", round(np.mean([m['auc'] for m in fold_metrics]), 3),
      "| Mean AUPRC:", round(np.mean([m['ap'] for m in fold_metrics]), 3))

# Cell 6: Make a compact results table
rows = []
for i, m in enumerate(fold_metrics, start=1):
    rows.append({
        "Fold": i,
        "ROC-AUC": round(float(m["auc"]), 3),
        "AUPRC": round(float(m["ap"]), 3),
        "Best Thr": round(float(m["thr"]), 3),
        "F1 (No MCID)": round(float(m["report"]["No MCID"]["f1-score"]), 3),
        "F1 (Yes MCID)": round(float(m["report"]["Yes MCID"]["f1-score"]), 3),
        "Accuracy": round(float(m["report"]["accuracy"]), 3),
    })
res_df = pd.DataFrame(rows)
print("\nResults table:\n", res_df.to_string(index=False))

# Optionally save results
out_csv = "/content/cv_results.csv"
try:
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")
except Exception as e:
    print("Could not save results CSV:", e)
