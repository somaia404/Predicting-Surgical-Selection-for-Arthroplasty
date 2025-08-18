

## Notebooks  

Exploratory and evaluation notebooks for the project.  

- `GatorTron_Model_Evaluation.py`: Original/modified baseline and class-weighted runs.  
- `RoBERTa_Evaluation.py`: Data augmentation & weight-decay experiments.  
- `MICD5FOLD.py`: Stratified 5-fold cross-validation for **MCID classification** with  
  class weighting, weighted random sampler, threshold tuning (Youden’s J),  
  and generation of confusion matrices & ROC curves.  

The production pipeline lives in `src/` (data loading, training, evaluation).  
Notebooks may contain ad-hoc cells; the canonical logic is in `src/`.  
