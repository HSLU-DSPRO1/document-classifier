from pathlib import Path
import pandas as pd
import json
import re
from .text_cleaning import basic_clean, filter_by_length
import numpy as np


DATE_FORMATS = ["%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d"]


# ======================================================
# ENRON EMAILS
# ======================================================
def extract_email_body(raw_msg: str) -> str:
    """Remove headers and quoted lines from Enron messages."""
    if not isinstance(raw_msg, str):
        return ""
    parts = raw_msg.split("\n\n", 1)
    body = parts[1] if len(parts) == 2 else parts[0]
    body = re.sub(r"(?m)^>+.*$", "", body)  # quoted replies
    body = re.split(r"(?i)\b(original message|forwarded by|from:)\b", body)[0]
    return basic_clean(body)


def load_enron_emails(path: str, max_rows=None):
    print("Loading Enron emails...")
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
    df["text"] = df["message"].apply(extract_email_body)
    df = df[["text"]]
    df["doc_type"] = "EMAIL"
    df["source"] = "ENRON"
    df = df.dropna(subset=["text"])
    df = filter_by_length(df, "text", min_chars=100, max_chars=10000)
    df = df.drop_duplicates(subset=["text"])
    print("Enron after cleaning:", df.shape)
    return df



# ======================================================
# INVOICES
# ======================================================
def invoice_row_to_text(row):
    """Convert a structured invoice row into a short pseudo-document."""
    return (
        f"Document number {row['id_invoice']} issued on {row['issuedDate']} "
        f"to client {row['client']} in {row['country']} for service {row['service']}. "
        f"Total amount: {row['total']} including discount {row['discount']} and tax {row['tax']}. "
        f"Outstanding balance: {row['balance']} with payment due on {row['dueDate']}. "
        f"Payment status: {row['invoiceStatus']}."
    )




#EXTEND LATER
def invoice_variants(row):
    variants = []

    # long form
    variants.append(
        f"Document number {row['id_invoice']} issued on {row['issuedDate']} "
        f"to client {row['client']} in {row['country']} for service {row['service']}. "
        f"Total amount: {row['total']} including discount {row['discount']} and tax {row['tax']}. "
        f"Outstanding balance: {row['balance']} with payment due on {row['dueDate']}. "
        f"Payment status: {row['invoiceStatus']}."
    )

    # shorter snippet-like form
    variants.append(
        f"Invoice {row['id_invoice']} for {row['client']}, "
        f"total {row['total']} (tax {row['tax']}). Due {row['dueDate']}."
    )

    return variants




def load_invoices(path: str, max_rows=None):
    print("Loading invoices...")
    df = pd.read_csv(path)
    if max_rows:
        df = df.head(max_rows)
    required = [
        "id_invoice", "issuedDate", "country", "service", "total",
        "discount", "tax", "invoiceStatus", "balance", "dueDate", "client"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing invoice columns: {missing}")

    df["text"] = df.apply(invoice_row_to_text, axis=1)
    df["text"] = df["text"].apply(basic_clean)
    df = df[["text"]]
    df["doc_type"] = "INVOICE"
    df["source"] = "INVOICES"
    df = df.dropna(subset=["text"])
    df = filter_by_length(df, "text", min_chars=80, max_chars=2000)
    df = df.drop_duplicates(subset=["text"])
    print("Invoices after cleaning:", df.shape)
    return df



# ======================================================
# INVOICES SECOND DATASET
# ======================================================

def _parse_date(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(s, format=fmt).date().isoformat()
        except Exception:
            continue
    try:
        return pd.to_datetime(s, errors="coerce").date().isoformat()
    except Exception:
        return ""
    

def _mask_email(s: str) -> str:
    if not isinstance(s, str) or "@" not in s:
        return ""
    name, dom = s.split("@", 1)
    return name[:2] + "***@" + dom.split(".", 1)[0] + ".***"


def _num(x):
    try:
        return float(x)
    except Exception:
        return np.nan
    

KAGGLE_REQUIRED = [
    "first_name","last_name","email","product_id","qty","amount",
    "invoice_date","address","city","stock_code","job"
]

def kaggle_invoice_variants(row):
    first = str(row.get("first_name", "")).strip()
    last  = str(row.get("last_name", "")).strip()
    client = " ".join([first, last]).strip() or "Customer"
    email_masked = _mask_email(str(row.get("email", "")))
    product_id = str(row.get("product_id", "")).strip()
    stock_code = str(row.get("stock_code", "")).strip()
    qty = _num(row.get("qty"))
    amount = _num(row.get("amount"))
    line_total = qty * amount if (pd.notna(qty) and pd.notna(amount)) else np.nan
    date_iso = _parse_date(str(row.get("invoice_date", "")))
    address = str(row.get("address", "")).strip()
    city = str(row.get("city", "")).strip()
    job = str(row.get("job", "")).strip()

    qty_str = f"{int(qty)}" if pd.notna(qty) and float(qty).is_integer() else (f"{qty:.0f}" if pd.notna(qty) else "")
    amt_str = f"{amount:.2f}" if pd.notna(amount) else ""
    tot_str = f"{line_total:.2f}" if pd.notna(line_total) else ""

    v = []
    v.append(
        f"Invoice issued on {date_iso} for client {client} ({email_masked}) at {address}, {city}. "
        f"Item {product_id} / stock {stock_code}: quantity {qty_str}, unit price {amt_str}, line total {tot_str}."
    )
    v.append(
        f"INVOICE {date_iso} — Client: {client}. Product {product_id} (stock {stock_code}), "
        f"qty {qty_str} at {amt_str}. Total {tot_str}."
    )
    v.append(
        f"Bill to: {client}, {address}, {city}. Date: {date_iso}. "
        f"Product {product_id} x{qty_str} @ {amt_str} = {tot_str} (stock {stock_code})."
    )
    v.append(
        f"{date_iso} invoice for {client}: {qty_str}×{product_id} total {tot_str}."
    )
    if job:
        v.append(
            f"Attention {client}, {job}. Invoice date {date_iso}. Item {product_id} ({stock_code}) "
            f"{qty_str} pcs at {amt_str}. Amount {tot_str}."
        )

    return [re.sub(r"\s+", " ", s).strip().replace(" .", ".") for s in v]



def load_invoices_kaggle(csv_path: str | None = None, max_rows: int | None = None, augment: bool = True) -> pd.DataFrame:
    print(f"Loading Kaggle invoices from {csv_path} …")
    df = pd.read_csv(csv_path)

    missing = [c for c in KAGGLE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Kaggle invoice columns: {missing}")

    if max_rows:
        df = df.head(max_rows)

    for c in ["first_name","last_name","email","address","city","job","stock_code","product_id"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    texts = []
    for _, row in df.iterrows():
        variants = kaggle_invoice_variants(row)
        if not augment:
            variants = variants[:1]
        texts.extend(variants)

    out = pd.DataFrame({"text": texts})
    out["text"] = out["text"].apply(basic_clean)
    out = out.dropna(subset=["text"])
    out = filter_by_length(out, "text", min_chars=60, max_chars=2000)
    out = out.drop_duplicates(subset=["text"]).reset_index(drop=True)
    out["doc_type"] = "INVOICE"
    out["source"] = "KAGGLE_INVOICES"
    print("Kaggle invoices after cleaning:", out.shape)
    return out


# ======================================================
# CUAD LEGAL CONTRACTS
# ======================================================



""" 
def load_cuad_full_contracts(folder_path: str):
   \"""Load all .txt contracts from CUAD's full_contract_txt folder.\"""
    folder = Path(folder_path)
    rows = []
    for p in folder.glob("*.txt"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            txt = p.read_text(errors="ignore")
        txt = basic_clean(txt)
        if txt:
            rows.append({
                "text": txt,
                "filename": p.name,
                "doc_type": "LEGAL_DOCUMENT",
                "source": "CUAD_full_contract"
            })
    df = pd.DataFrame(rows)
    df = filter_by_length(df, text_col="text", min_chars=200, max_chars=None)
    df = df.drop_duplicates(subset=["text"])
    print("CUAD full contracts:", df.shape)
    return df
"""



# ======================================================
# ARXIV SCIENTIFIC PAPERS
# ======================================================
def load_arxiv_from_jsonl(path: str, max_rows=None):
    """Load arXiv metadata JSONL and build title+abstract text field."""
    print("Loading arXiv JSONL...")
    rows = []
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_rows and count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            title = data.get("title", "")
            abstract = data.get("abstract", "")
            
            text = f"{title}. {abstract}"
            text = basic_clean(text)
            if text:
                rows.append({
                    "text": text,
                    "doc_type": "SCIENTIFIC_PAPER",
                    "source": "ARXIV",
                    
                })
                count += 1
    df = pd.DataFrame(rows)
    df = filter_by_length(df, text_col="text", min_chars=200, max_chars=8000)
    df = df.drop_duplicates(subset=["text"])
    print("arXiv after cleaning:", df.shape)
    return df