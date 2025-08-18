
import pandas as pd
from tqdm import tqdm
import random
import nltk
from nltk.corpus import wordnet  # used to ensure wordnet is available when augmenting
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# --- Data loading ---------------------------------------------------------
def load_data(csv_path: str):
    """Load the CSV, map labels {'No':0, 'Yes':1}, and return texts, labels, df."""
    data = pd.read_csv(csv_path)
    texts = data['Interpretation'].tolist()
    label_map = {'No': 0, 'Yes': 1}
    labels = data['operated_on'].map(label_map).tolist()
    return texts, labels, data

# --- Light data augmentation (Random Insertion) ---------------------------
def ensure_wordnet():
    """Best-effort ensure NLTK wordnet is available (no-op if already present)."""
    try:
        wordnet.synsets('test')
    except Exception:
        try:
            nltk.download('wordnet')
        except Exception:
            pass

def random_insertion(text: str, alpha: float = 0.1) -> str:
    """Randomly insert existing words from the sentence back into it.

    This is a very simple augmentation: choose n_insertions ≈ alpha * len(words),
    then randomly insert a random word from the original list.
    """
    words = text.split()
    if not words:
        return text
    n = len(words)
    n_insertions = max(1, int(alpha * n))
    new_words = words.copy()
    for _ in range(n_insertions):
        random_word = random.choice(words)
        random_idx = random.randint(0, len(new_words))
        new_words.insert(random_idx, random_word)
    return " ".join(new_words)

def augment_minority_with_random_insertion(texts, labels, minority_label: int):
    """Augment all minority-class samples once using random_insertion.

    Returns augmented_texts, augmented_labels (original + augmented).
    """
    ensure_wordnet()
    minority_texts = [texts[i] for i in range(len(texts)) if labels[i] == minority_label]
    augmented_texts = [random_insertion(t) for t in tqdm(minority_texts, total=len(minority_texts))]
    augmented_labels = [minority_label] * len(augmented_texts)
    return texts + augmented_texts, labels + augmented_labels

# --- Cross-validation helper ---------------------------------------------
def get_stratified_kfold(n_splits: int = 3, random_state: int = 42) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def compute_fold_class_weights(y_train):
    """Compute class weights tensor-like list for class-balanced loss."""
    classes = sorted(set(y_train))
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return weights.tolist()
