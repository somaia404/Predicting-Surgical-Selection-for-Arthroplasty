# 📂 Source Code Overview

This folder contains all the scripts required to run the NLP pipeline for predicting surgical selection from radiology reports. Each script has a specific role in the workflow:

- **`__init__.py`**  
  Marks the folder as a Python package to allow modular imports.

- **`data_loader.py`**  
  Loads the dataset (CSV format), maps binary labels (`Yes/No` → `1/0`), applies text augmentation where required, and prepares stratified folds using `StratifiedKFold` for balanced cross-validation.

- **`model_trainer.py`**  
  Handles model initialization (e.g., GatorTron, ClinicalBERT), tokenizer loading, and training loops. It also manages cross-validation, learning rate scheduling, and checkpoint saving.

- **`evaluation_and_plotting.py`**  
  Computes and visualises performance metrics, including accuracy, precision, recall, F1 score, and AUROC. Generates confusion matrices, ROC curves, and training/validation loss plots.

- **`private_data_processing/`**  
  Contains R scripts used for initial data linkage and cleaning within the SHAIP secure environment. All scripts are de-identified and contain no private patient information.

---

✅ Together, these scripts form a **reproducible pipeline** for model training, evaluation, and result interpretation.  
flowchart LR
    A[CSV: hip_radiology_reports_finalised_SYNTH.csv] --> B[data_loader.py]
    B -->|map labels / NLTK / augmentation| C{StratifiedKFold}
    C -->|Fold 1..K| D[model_trainer.py]
    D -->|train + validate per fold| E[Checkpoints / artifacts]
    D --> F[evaluation_and_plotting.py]
    F -->|metrics + plots| G[reports/images & README]
