# Notebooks

## Text model
Run order:
1. `text/00_config_and_checks.ipynb`
2. `text/01_build_dataset.ipynb`
3. `text/02_train_model.ipynb`
4. `text/03_evaluate.ipynb`
5. `text/04_pdf_inference.ipynb` (optional)

Artifacts created:
- `data/processed/{train,val,test}.csv`
- `models/<text_model_name>.pkl`
- `outputs/figures/text_model/cm_test_*.png`

## Vision model
- `vision/10_vision_model_integration.ipynb`
- `vision/11_vision_eval_report.ipynb`

Vision artifacts should live under:
- `models/vision_model/<model_name_or_version>/`
- `outputs/figures/vision_model/`
