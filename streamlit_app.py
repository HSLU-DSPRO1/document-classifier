# ---- ABSOLUTE TOP: ensure project root on sys.path ----
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../doc_class
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- regular imports ----
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from utils.paths import MODELS

# Paths
TFIDF_MODEL_PATH = MODELS / "tfidf_logreg_baseline.pkl"
TRANSFORMER_DIR  = MODELS / "transformer_baseline_final"  # saved in 04_transformer_model

# =========================
# Loaders (cached)
# =========================

@st.cache_resource
def load_tfidf_model():
    """Load the trained scikit-learn pipeline (TF-IDF + LogisticRegression)."""
    model = joblib.load(TFIDF_MODEL_PATH)
    return model

@st.cache_resource
def load_transformer():
    """
    Load tokenizer + model for DistilBERT.
    Returns (tokenizer, model, id2label, device).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        st.error(f"Transformers/torch not available: {e}")
        return None, None, None, None

    if not TRANSFORMER_DIR.exists():
        st.error(f"Transformer model folder not found at {TRANSFORMER_DIR}")
        return None, None, None, None

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    id2label = model.config.id2label
    return tokenizer, model, id2label, device

# =========================
# Predictors
# =========================

def predict_tfidf(model, text: str) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[List[str]]]:
    """Predict with TF-IDF pipeline."""
    if not text or not text.strip():
        return None, None, None
    proba = model.predict_proba([text])[0]
    classes: List[str] = list(model.classes_)
    pred_idx = int(np.argmax(proba))
    pred_label = classes[pred_idx]
    prob_dict: Dict[str, float] = dict(zip(classes, proba))
    return pred_label, prob_dict, classes

def predict_transformer(tokenizer, model, id2label, device, text: str, max_length: int = 256
) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[List[str]]]:
    """Predict with DistilBERT, returning label, probs, class list."""
    if tokenizer is None or model is None:
        return None, None, None
    if not text or not text.strip():
        return None, None, None

    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    classes = [id2label[i] for i in range(len(probs))]
    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    prob_dict = {c: float(p) for c, p in zip(classes, probs)}
    return pred_label, prob_dict, classes

# =========================
# App
# =========================

def main():
    st.set_page_config(page_title="Document Type Classifier", layout="wide")
    st.title("Document Type Classifier")
    st.markdown(
        "Classifies text into: **EMAIL**, **INVOICE**,**SCIENTIFIC_PAPER**."
    )

    # Sidebar: model selector
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox(
            "Model",
            ["TF-IDF + Logistic Regression", "DistilBERT (Transformer)"],
            index=0
        )
        max_len = st.slider("Max tokens (Transformer)", 64, 512, 256, 32,
                            help="Ignored for TF-IDF.")

    # Load selected model(s)
    if model_choice == "TF-IDF + Logistic Regression":
        tfidf_model = load_tfidf_model()
        transformer_bundle = None
    else:
        tfidf_model = None
        transformer_bundle = load_transformer()

    # Layout
    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("Input")
        mode = st.radio("Input mode:", ["Paste text", "Upload file"], horizontal=True)

        text = ""
        if mode == "Paste text":
            text = st.text_area(
                "Paste your document text here:",
                height=300,
                placeholder="Dear John, as discussed in the meeting..."
            )
        else:
            uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
            if uploaded is not None:
                text = uploaded.read().decode("utf-8", errors="ignore")

        run = st.button("Classify")

    with col_output:
        st.subheader("Prediction")
        if run:
            if model_choice == "TF-IDF + Logistic Regression":
                pred_label, prob_dict, classes = predict_tfidf(tfidf_model, text)
            else:
                tokenizer, transformer, id2label, device = transformer_bundle
                pred_label, prob_dict, classes = predict_transformer(
                    tokenizer, transformer, id2label, device, text, max_length=max_len
                )

            if pred_label is None:
                st.warning("No text provided or model not available.")
            else:
                st.markdown(f"### Predicted type: **{pred_label}**")

                # Probability bar chart
                st.write("#### Class probabilities")
                proba_df = pd.DataFrame({
                    "class": classes,
                    "probability": [prob_dict[c] for c in classes]
                })
                st.bar_chart(proba_df, x="class", y="probability")

                # Show snippet of text
                st.write("#### Text snippet")
                snippet = (text[:500] + "…") if len(text) > 500 else text
                st.code(snippet)


if __name__ == "__main__":
    main()
