import re
import pandas as pd
from typing import List

# --- Abbreviation mapping ---
ABBREV_MAP = {
    r"\bnsclc\b": "non small cell lung cancer",
    r"\bcrc\b": "colorectal cancer",
    r"\ber\+\b": "er positive",
    r"\bpr\+\b": "pr positive",
    r"\bmsi-h\b": "msi high",
}

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s\-\+]", " ", t)  # keep letters, numbers, hyphen, plus
    for pattern, repl in ABBREV_MAP.items():
        t = re.sub(pattern, repl, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).lower()

def combine_text(df: pd.DataFrame, text_cols: List[str]) -> pd.Series:
    combined = df[text_cols].fillna("").agg(" ".join, axis=1)
    return combined.map(clean_text)

def split_multilabel(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.split(";") if x.strip()]

def join_multilabel(labels: List[str]) -> str:
    return "; ".join(sorted([l.strip() for l in labels if l.strip()]))
