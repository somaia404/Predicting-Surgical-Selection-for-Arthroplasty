from pathlib import Path
import pandas as pd

REQUIRED_COLS = ["Interpretation", "MCID"]

def load_mcid_csv(csv_path: str):
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {p}")

    df["Interpretation"] = df["Interpretation"].fillna("").astype(str)
    df = df[df["Interpretation"].str.len() >= 10].reset_index(drop=True)
    df["label"] = df["MCID"].map({"No": 0, "Yes": 1}).astype(int)

    texts = df["Interpretation"].tolist()
    labels = df["label"].tolist()
    return df, texts, labels
