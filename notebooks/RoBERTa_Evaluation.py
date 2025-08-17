import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from nltk.corpus import wordnet
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, cohen_kappa_score, log_loss, average_precision_score, balanced_accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import os
import random
# from nltk.corpus import wordnet # Already imported above


# Download necessary NLTK data (if not already downloaded)
# try:
#     wordnet.synsets('test') # Check if wordnet is available
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet')
# except LookupError:
#     nltk.download('wordnet')


# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, fold_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Fold {fold_name}')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(test_labels, probabilities, fold_name):
    fpr, tpr, thresholds = roc_curve(test_labels, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - Fold {fold_name}')
    plt.legend(loc="lower right")
    plt.show()

# Assuming 'confusion_matrices' and 'all_predictions' and 'test_labels' from the last execution are available
# Need to re-run the previous cell to get confusion_matrices and probabilities

# Re-running the training and evaluation loop to capture fold-wise predictions and true labels for plotting
# This is necessary because the previous code only stored aggregate metrics and not fold-wise details needed for plotting.

# Re-initialize lists to store results for plotting
all_test_labels = []
all_probabilities = []
confusion_matrices_for_plotting = []


# Re-load data and configure cross-validations
csv_file_path = "/content/hip_radiology_reports_finalised_SYNTH.csv"
data = pd.read_csv(csv_file_path)
texts = data["Interpretation"].tolist()
label_map = {"No": 0, "Yes": 1}
labels = data["operated_on"].map(label_map).tolist()

# Identify minority class
class_counts = data["operated_on"].value_counts()
minority_class = class_counts.idxmin()

# Separate minority class data
minority_texts = [texts[i] for i in range(len(texts)) if labels[i] == label_map[minority_class]]
minority_labels = [labels[i] for i in range(len(labels)) if labels[i] == label_map[minority_class]]


def random_insertion(text, alpha=0.1):
    """Randomly inserts words into the text."""
    words = text.split()
    new_words = []
    n = len(words)
    n_insertions = max(1, int(alpha * n))

    for _ in range(n_insertions):
        random_word = random.choice(words) # Simple random word selection
        random_idx = random.randint(0, n)
        new_words.insert(random_idx, random_word)
        n += 1 # Update length after insertion

    return " ".join(new_words)

augmented_texts_ri = []
augmented_labels_ri = []

print(f"Applying Random Insertion to the minority class ('{minority_class}')...")
# Augment each minority class text once
for text, label in tqdm(zip(minority_texts, minority_labels), total=len(minority_texts)):
    augmented_text = random_insertion(text)
    augmented_texts_ri.append(augmented_text)
    augmented_labels_ri.append(label)


# Combine original data with augmented minority class data
augmented_texts_combined_ri = texts + augmented_texts_ri
augmented_labels_combined_ri = labels + augmented_labels_ri


kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# Initialize model_path, tokenizer, and device (ensure these are defined)
model_path = "roberta-base" # Using RoBERTa as determined in previous steps
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_length = 150
chosen_threshold = 0.3


# Use the combined original and augmented data (from random insertion)
texts_for_training = augmented_texts_combined_ri
labels_for_training = augmented_labels_combined_ri


# perform cross-validation. Split data then fine tune.
for fold, (train_index, test_index) in enumerate(kf.split(texts_for_training, labels_for_training)):

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_texts = [texts_for_training[i] for i in train_index]
    train_labels = [labels_for_training[i] for i in train_index]
    test_texts = [texts_for_training[i] for i in test_index]
    test_labels = [labels_for_training[i] for i in test_index]

    # Calculate class weights for the current fold's training data
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    tokenized_inputs = tokenizer(train_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.01) # Added weight_decay for regularization
    batch_size = 16 # Using a fixed batch size for this task
    epochs = 50 # Using a fixed number of epochs for this task
    num_samples = len(tokenized_inputs.input_ids)
    num_batches = (num_samples - 1) // batch_size + 1
    epoch_train_losses = []

    # Instantiate CrossEntropyLoss with class weights
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        optimizer.zero_grad()
        batch_losses = [] # Store losses for batches within an epoch
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}", unit = "batch"):
            optimizer.zero_grad()
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, num_samples)
            batch_inputs = {k: v[batch_start:batch_end] for k, v in tokenized_inputs.items()}
            batch_labels = torch.tensor(train_labels[batch_start:batch_end])
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            batch_labels = batch_labels.to(device)
            outputs = model(**batch_inputs, labels=batch_labels)

            # Calculate loss using the weighted loss function
            loss = loss_fn(outputs.logits, batch_labels)

            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_train_losses.append(np.mean(batch_losses)) # Average batch losses for epoch
        tqdm.write(f"Batch {batch_idx+1}/{num_batches}") # Write after epoch progress bar


    model.eval()
    test_outputs = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = test_texts[i:i + batch_size]
            tokenized_test_inputs = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
            batch_inputs2 = {k: v.to(device) for k, v in tokenized_test_inputs.items()}
            batch_outputs = model(**batch_inputs2)
            test_outputs.append(batch_outputs.logits)

    test_outputs = torch.cat(test_outputs, dim=0)
    probabilities = torch.softmax(test_outputs, dim=-1).cpu().detach().numpy()[:, 1]

    # Collect true labels and probabilities for plotting
    all_test_labels.extend(test_labels)
    all_probabilities.extend(probabilities)

    # Calculate confusion matrix for plotting
    isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    calibrated_probs = isotonic_regressor.fit_transform(probabilities, test_labels)
    predictions = (calibrated_probs >= chosen_threshold).astype(int).tolist()
    conf_matrix = confusion_matrix(test_labels, predictions)
    confusion_matrices_for_plotting.append(conf_matrix)


# Plot confusion matrices for each fold
for fold, conf_matrix in enumerate(confusion_matrices_for_plotting):
    plot_confusion_matrix(conf_matrix, fold+1) # Pass fold+1 to the plotting function

# Plot ROC curve for the combined predictions and true labels across all folds
# This gives an overall view, you could also plot individual fold ROC curves if preferred.
plot_roc_curve(all_test_labels, all_probabilities, "Overall")
