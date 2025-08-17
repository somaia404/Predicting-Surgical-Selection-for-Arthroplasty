# Models

This folder is a placeholder for trained model artifacts.  
Due to GitHub file size restrictions, **full trained model weights are not stored in this repository**.  

## 📦 Available Artifacts (if saved locally)
- `config.json` – Model configuration used during training  
- `tokenizer.json` – Tokenizer vocabulary used for input processing  
- `special_tokens_map.json` – Mapping of special tokens  

## 🔗 Accessing Trained Weights
The full trained models (RoBERTa variants and GatorTron baseline) are available **upon request**.  
Please contact: **smammari8@gmail.com**  

## 📝 Provenance
These models were **adapted from Dr. Luke Farrow’s original GatorTron-based model** developed inside the **SHAIP secure environment** using real clinical data.  
- In this repository, experiments extend that framework with **RoBERTa variants** and **synthetic (mimic) data** for reproducibility and demonstration purposes.  
- The secure dataset itself cannot be shared, but the methodology is faithfully reproduced here.  

## ⚡ Reproducing Training
You can retrain the models from scratch using the provided script:

```bash
python train.py --csv data/hip_radiology_reports_finalised_SYNTH.csv --model UFNLP/gatortron-base --folds 3
