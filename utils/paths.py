from pathlib import Path

# BASE = project root "doc_class/"
BASE = Path(__file__).resolve().parents[1]  # doc_class/


# data dirs
DATA = BASE / "datasets"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

#raw source folders
RAW_ENRON = (RAW / "enron.csv")
RAW_INVOICES = (RAW / "invoices.csv")
RAW_INVOICES_2 = (RAW / "invoices_2.csv")
RAW_ARXIV = (RAW / "arxiv.json")

# models
MODELS = BASE / "models"

# outputs (figures, logs)
OUTPUTS = BASE / "outputs"
FIGURES = OUTPUTS / "figures"
FIG_EDA = FIGURES / "eda"
FIG_EVAL = FIGURES / "eval"

LOGS = OUTPUTS / "logs"