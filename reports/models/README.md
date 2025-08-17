# Models

This folder is a placeholder for trained model artifacts.  
Due to file size restrictions, full trained model weights are **not stored in this repository**.

### Available Artifacts
- `config.json` : Model configuration used during training.
- `tokenizer.json` : Tokenizer vocabulary used for input processing.
- `special_tokens_map.json` : Mapping of special tokens.

### Accessing Trained Weights
The full trained models (RoBERTa variants, GatorTron baseline) are available upon request.  
Alternatively, the models can be retrained using the provided scripts:

```bash
python train.py --csv data/hip_radiology_reports_finalised_SYNTH.csv --model UFNLP/gatortron-base --folds 3
