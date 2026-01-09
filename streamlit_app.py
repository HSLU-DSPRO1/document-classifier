import sys
from pathlib import Path
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# --- Project root ---
ROOT = Path(__file__).resolve().parent  # if streamlit_app.py is in repo root
# If your streamlit_app.py is inside /app, use: ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Model path (adjust if needed) ---
TFIDF_MODEL_PATH = ROOT / "models" / "tfidf_logreg_baseline.pkl"

SUPPORTED = ["EMAIL", "INVOICE", "SCIENTIFIC_PAPER"]

@st.cache_resource
def load_model():
    if not TFIDF_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {TFIDF_MODEL_PATH}")
    return joblib.load(TFIDF_MODEL_PATH)

def predict(model, text: str) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[List[str]]]:
    if not text or not text.strip():
        return None, None, None

    proba = model.predict_proba([text])[0]
    classes: List[str] = list(model.classes_)

    # If model contains extra classes, filter down to supported (optional safety)
    if set(SUPPORTED).issubset(set(classes)) and set(classes) != set(SUPPORTED):
        idx = [classes.index(c) for c in SUPPORTED]
        proba = np.array([proba[i] for i in idx], dtype=float)
        proba = proba / proba.sum()
        classes = SUPPORTED

    pred_idx = int(np.argmax(proba))
    pred_label = classes[pred_idx]
    prob_dict = {c: float(p) for c, p in zip(classes, proba)}
    return pred_label, prob_dict, classes

def prob_df(prob_dict: Dict[str, float], classes: List[str]) -> pd.DataFrame:
    df = pd.DataFrame({"Class": classes, "Probability": [prob_dict[c] for c in classes]})
    df = df.sort_values("Probability", ascending=False)
    df["Probability %"] = (df["Probability"] * 100).round(1)
    return df

def css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
          div[data-testid="metric-container"] {
            border: 1px solid rgba(255,255,255,0.12);
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(255,255,255,0.03);
          }
          .stButton>button { border-radius: 14px; font-weight: 650; }
          textarea { border-radius: 14px !important; }
          .small-muted { color: rgba(255,255,255,0.70); font-size: 0.95rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Document Type Classifier", layout="wide")
    css()

    st.markdown("# Document Type Classifier")
    st.markdown("<div class='small-muted'>TF-IDF + Logistic Regression • 3 classes</div>", unsafe_allow_html=True)
    st.caption("Supported: **EMAIL**, **INVOICE**, **SCIENTIFIC_PAPER**")

    try:
        model = load_model()
        st.success(f"Loaded model: {TFIDF_MODEL_PATH}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.header("⚙️ Settings")
        show_probs = st.toggle("Show probability chart", value=True)
        show_snippet = st.toggle("Show text snippet", value=True)
        st.divider()
        st.subheader("Quick samples")
        samples = {
            "EMAIL": "Hi team,\n\nCan we move the meeting to Friday 10:00? Thanks.\n\nBest,\nAlex",
            "INVOICE": "INVOICE #10492\nDate: 2024-11-01\nTotal: CHF 1,249.50\nVAT: 7.7%\nPlease pay within 30 days.",
            "SCIENTIFIC_PAPER": "In this paper, we propose a method for document type classification using TF-IDF and linear models. Results show improved macro-F1."
        }
        chosen = st.selectbox("Insert sample", list(samples.keys()), index=0)
        insert = st.button("Paste sample")

    if "doc_text" not in st.session_state:
        st.session_state["doc_text"] = ""
    if insert:
        st.session_state["doc_text"] = samples[chosen]

    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.subheader("Input")
        text = st.text_area("Document text", key="doc_text", height=320, placeholder="Paste your document text here…")
        run = st.button("Classify", type="primary", use_container_width=True)

    with col2:
        st.subheader("Output")
        if not run:
            st.info("Paste text and click **Classify**.")
        else:
            label, probs, classes = predict(model, text)
            if label is None:
                st.warning("No text provided.")
            else:
                df = prob_df(probs, classes)
                top1 = df.iloc[0]
                top2 = df.iloc[1] if len(df) > 1 else None

                k1, k2, k3 = st.columns(3)
                k1.metric("Predicted type", label)
                k2.metric("Confidence", f"{top1['Probability %']}%")
                margin = (top1["Probability"] - (top2["Probability"] if top2 is not None else 0.0)) * 100
                k3.metric("Margin vs #2", f"{margin:.1f}%")

                st.divider()
                if show_probs:
                    st.markdown("#### Probabilities")
                    st.dataframe(df[["Class", "Probability %"]], hide_index=True, use_container_width=True)
                    st.bar_chart(df.set_index("Class")["Probability"], height=240)

                if show_snippet:
                    st.markdown("#### Snippet")
                    snippet = (text[:1200] + "…") if len(text) > 1200 else text
                    st.code(snippet)

if __name__ == "__main__":
    main()
