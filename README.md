# Document Type Classifier (PDF + Text)

Classifies documents into:
- **EMAIL**
- **INVOICE**
- **SCIENTIFIC_PAPER**

Pipeline:
1. Build dataset from multiple sources (emails, invoices, arXiv abstracts).
2. Train a lightweight text classifier (TF‑IDF + Logistic Regression).
3. Evaluate with a held-out test split + confusion matrix.
4. Demo on PDFs via embedded text extraction (PyMuPDF / pdfplumber).

## Quickstart

### 1) Environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Data
Put the raw files into `input/` (see `input/README.md`).

### 3) Run notebooks
Open `notebooks/` and run in order:
- `00_config_and_checks.ipynb`
- `01_build_dataset.ipynb`
- `02_train_model.ipynb`
- `03_evaluate.ipynb`

### 4) PDF demo (optional)
Place PDFs into `input/pdfs/` and run:
- `04_pdf_inference.ipynb`

## Results
- Confusion matrix is saved to `outputs/figures/`.

## Repository layout
```
input/                # raw data (not committed)
notebooks/            # build → train → eval → pdf demo
data/processed/       # train/val/test splits (generated)
models/               # trained model artifacts (generated)
outputs/figures/      # evaluation plots (generated)
docs/                 # short report notes
```

## Reproducibility
Training and evaluation parameters are stored in a `*.meta.json` file next to the saved model.

## License
See `LICENSE`.


## Model organization

This repository keeps text and vision models in separate subfolders to simplify team merges:

```
models/
  text_model/
    <version_or_name>/
      model.pkl
      meta.json
  vision_model/
    <version_or_name>/
      ... (framework-specific files)
      meta.json
```

Recommended: keep large model files out of Git and publish them as release assets.
