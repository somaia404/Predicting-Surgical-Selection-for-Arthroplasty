
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, confusion_matrix, cohen_kappa_score,
                             calibration_curve, balanced_accuracy_score, log_loss,
                             average_precision_score)
from sklearn.isotonic import IsotonicRegression

from .data_loader import get_stratified_kfold

# --- Model setup ----------------------------------------------------------
def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model, tokenizer, device

# --- Training over CV -----------------------------------------------------
def train_crossval(texts: List[str],
                   labels: List[int],
                   model_name: str = "UFNLP/gatortron-base",
                   n_splits: int = 3,
                   max_length: int = 150,
                   lr: float = 2e-6,
                   epochs: int = 5,
                   batch_size: int = 16,
                   decision_threshold: float = 0.3) -> Dict[str, Any]:
    """Train a text classifier with stratified K-fold and return metrics + artifacts."""
    kf = get_stratified_kfold(n_splits=n_splits)
    all_metrics = {k: [] for k in [
        'accuracy','f1','recall','precision','roc_auc','kappa','balanced_acc',
        'logloss','auprc'
    ]}
    confusion_matrices = []
    calibration_fops = []   # fraction of positives (per fold)
    calibration_mpvs = []   # mean predicted value (per fold)
    saved_model_dirs = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(texts, labels), start=1):
        model, tokenizer, device = load_model_and_tokenizer(model_name)

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts  = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        tokenized = tokenizer(train_texts, padding="max_length", truncation=True,
                              max_length=max_length, return_tensors="pt")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        num_samples = len(tokenized.input_ids)
        num_batches = (num_samples - 1) // batch_size + 1

        for epoch in range(epochs):
            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min((b + 1) * batch_size, num_samples)
                batch_inputs = {k: v[b_start:b_end].to(device) for k, v in tokenized.items()}
                batch_labels = torch.tensor(train_labels[b_start:b_end]).to(device)
                outputs = model(**batch_inputs, labels=batch_labels)
                loss = torch.nn.functional.cross_entropy(outputs.logits, batch_labels)
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        # --- Eval on test split ---
        model.eval()
        test_logits_chunks = []
        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i+batch_size]
                enc = tokenizer(batch_texts, padding="max_length", truncation=True,
                                max_length=max_length, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = model(**enc)
                logits = out.logits if not isinstance(model, torch.nn.DataParallel) else out.logits
                test_logits_chunks.append(logits)

        test_logits = torch.cat(test_logits_chunks, dim=0)
        probs_pos = torch.softmax(test_logits, dim=-1).detach().cpu().numpy()[:, 1]

        # Probability calibration + thresholding
        iso = IsotonicRegression(out_of_bounds='clip')
        calibrated = iso.fit_transform(probs_pos, test_labels)
        preds = (calibrated >= decision_threshold).astype(int).tolist()

        fop, mpv = calibration_curve(test_labels, calibrated, n_bins=5)
        calibration_fops.append(fop); calibration_mpvs.append(mpv)

        acc = accuracy_score(test_labels, preds)
        f1  = f1_score(test_labels, preds)
        rec = recall_score(test_labels, preds)
        pre = precision_score(test_labels, preds)
        auc = roc_auc_score(test_labels, preds)  # using label preds (mirrors notebook)
        kap = cohen_kappa_score(test_labels, preds)
        bal = balanced_accuracy_score(test_labels, preds)
        ll  = log_loss(test_labels, probs_pos)
        aupr= average_precision_score(test_labels, probs_pos)
        cm  = confusion_matrix(test_labels, preds)

        all_metrics['accuracy'].append(acc)
        all_metrics['f1'].append(f1)
        all_metrics['recall'].append(rec)
        all_metrics['precision'].append(pre)
        all_metrics['roc_auc'].append(auc)
        all_metrics['kappa'].append(kap)
        all_metrics['balanced_acc'].append(bal)
        all_metrics['logloss'].append(ll)
        all_metrics['auprc'].append(aupr)
        confusion_matrices.append(cm)

        # Save per-fold model + tokenizer
        save_dir = f"Radiology_Reports/model_fold{fold}"
        if isinstance(model, torch.nn.DataParallel):
            model.module.save_pretrained(save_dir)
        else:
            model.save_pretrained(save_dir)
        tokenizer.save_vocabulary(save_dir)
        saved_model_dirs.append(save_dir)

    return {
        "metrics_per_fold": all_metrics,
        "confusion_matrices": confusion_matrices,
        "calibration_fops": calibration_fops,
        "calibration_mpvs": calibration_mpvs,
        "saved_model_dirs": saved_model_dirs,
    }
