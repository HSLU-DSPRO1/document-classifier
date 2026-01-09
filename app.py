# app.py
import os, joblib, hashlib, time
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from utils.paths import MODELS
app = FastAPI()


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/tfidf_logreg_baseline.pkl"))

def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

class Payload(BaseModel):
    text: str

model = joblib.load(MODELS / "tfidf_logreg_baseline.pkl")  

@app.post("/predict")
def predict(p: Payload):
    proba = getattr(model, "predict_proba", None)
    label = model.predict([p.text])[0]
    probs = {}
    if proba:
        classes = getattr(model, "classes_", [])
        for c, v in zip(classes, proba([p.text])[0]):
            probs[str(c)] = float(v)
    return {"label": str(label), "probs": probs}





#uvicorn app:app --host 0.0.0.0 --port 8000 --reload
