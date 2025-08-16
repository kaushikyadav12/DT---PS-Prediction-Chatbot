import pandas as pd
from collections import defaultdict
from .config import KEYWORDS_CSV, LABEL_COLS

def load_mapping(path=KEYWORDS_CSV) -> pd.DataFrame:
    """
    Load keyword-to-label mapping CSV and fill missing values.
    Uses cp1252 encoding to avoid UnicodeDecodeError on Windows.
    """
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='cp1252')
    except Exception:
        # fallback to utf-8 if cp1252 fails
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8')
    return df.fillna("")

def match_keywords(text: str, mapping_df: pd.DataFrame):
    """
    Match keywords in text to generate multilabel predictions.
    Returns:
        - dict of label -> list of matched values
        - list of matched keywords
    """
    text_low = str(text).lower()
    votes = defaultdict(set)
    matched = []

    for _, row in mapping_df.iterrows():
        kw = str(row.get("keyword", "")).lower().strip()
        if not kw:
            continue
        if kw in text_low:
            # Collect votes for each label column
            for col in LABEL_COLS:
                val = row.get(col)
                if val:
                    for v in str(val).split(";"):
                        v = v.strip()
                        if v:
                            votes[col].add(v)
            matched.append(kw)

    # Ensure all LABEL_COLS are present, even if empty
    results = {k: sorted(list(votes.get(k, []))) for k in LABEL_COLS}

    return results, sorted(set(matched))
