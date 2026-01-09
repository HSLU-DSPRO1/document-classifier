import re
import pandas as pd




def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def filter_by_length(df: pd.DataFrame, text_col="text", min_chars=None, max_chars=None):
    df = df.copy()
    df["n_chars"] = df[text_col].str.len()
    if min_chars is not None:
        df = df[df["n_chars"] >= min_chars]
    if max_chars is not None:
        df = df[df["n_chars"] <= max_chars]
    return df.drop(columns=["n_chars"])

