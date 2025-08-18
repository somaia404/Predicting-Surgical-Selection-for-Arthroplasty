from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from .evaluation_and_plotting import plot_confusion, plot_roc, summarize_fold, results_table

class TxtDs(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=192):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tokenizer, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def train_eval_fold(model_name, X_tr, y_tr, X_va, y_va, epochs=6, lr=2e-5, batch_size=16, max_len=192):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # weights & sampler
    counts = Counter(y_tr); n0, n1 = counts.get(0,1), counts.get(1,1)
    w0, w1 = 1.0/n0, 1.0/n1
    class_weights = torch.tensor([w0, w1], dtype=torch.float)
    sample_weights = torch.tensor([w0 if y==0 else w1 for y in y_tr], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    tr_ds, va_ds = TxtDs(X_tr, y_tr, tok, max_len), TxtDs(X_va, y_va, tok, max_len)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_auc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            ids, mask, y = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            logits = model(ids, attention_mask=mask).logits
            loss = loss_fn(logits, y); loss.backward(); opt.step()

        # quick val AUC
        model.eval(); probs_v, y_v = [], []
        with torch.no_grad():
            for batch in va_loader:
                ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
                y = batch["labels"].cpu().numpy()
                p1 = torch.softmax(model(ids, attention_mask=mask).logits, dim=1)[:,1].cpu().numpy()
                probs_v.extend(p1); y_v.extend(y)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_v, probs_v)
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)

    # final eval + Youden J threshold
    model.eval(); probs, y_true = [], []
    with torch.no_grad():
        for batch in va_loader:
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            y = batch["labels"].cpu().numpy()
            p1 = torch.softmax(model(ids, attention_mask=mask).logits, dim=1)[:,1].cpu().numpy()
            probs.extend(p1); y_true.extend(y)
    fpr, tpr, thr = roc_curve(y_true, probs)
    youden_idx = int(np.argmax(tpr - fpr))
    thr_star = float(thr[youden_idx]) if len(thr) else 0.5

    # plots
    cm_auc = plot_roc(y_true, probs, f"ROC – {model_name}")
    cm = plot_confusion.__wrapped__ if hasattr(plot_confusion, "__wrapped__") else plot_confusion
    cm(confusion_matrix=np.array([[0,0],[0,0]]))  # placeholder to keep API parity
    # use summarize_fold to compute real cm & metrics table:
    return summarize_fold(y_true, probs, thr_star, model_name)

def run_cv(model_name, texts, labels, n_splits=5, **train_kwargs):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    for tr_idx, va_idx in skf.split(texts, labels):
        X_tr = [texts[i] for i in tr_idx]; y_tr = [labels[i] for i in tr_idx]
        X_va = [texts[i] for i in va_idx]; y_va = [labels[i] for i in va_idx]
        m = train_eval_fold(model_name, X_tr, y_tr, X_va, y_va, **train_kwargs)
        fold_metrics.append(m)
    return fold_metrics, results_table(fold_metrics)
