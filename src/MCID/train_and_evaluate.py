import argparse
from .data_loader import load_mcid_csv
from .model_trainer import run_cv

def main():
    ap = argparse.ArgumentParser(description="MCID 5-fold CV")
    ap.add_argument("--csv", required=True, help="Path to MCID CSV (must have Interpretation, MCID)")
    ap.add_argument("--model", default="UFNLP/gatortron-base", help="HF model id e.g. roberta-base")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df, texts, labels = load_mcid_csv(args.csv)
    metrics, table = run_cv(args.model, texts, labels, n_splits=args.folds)
    print("\nPer-fold AUC:", [round(m["auc"],3) for m in metrics])
    print("Mean AUC:", round(table["ROC-AUC"].mean(),3), "| Mean AUPRC:", round(table["AUPRC"].mean(),3))
    print("\nResults table:\n", table.to_string(index=False))

if __name__ == "__main__":
    main()
