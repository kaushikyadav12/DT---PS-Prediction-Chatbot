import pandas as pd
from collections import defaultdict
from .config import KEYWORDS_CSV, LABEL_COLS

def load_mapping(path=KEYWORDS_CSV) -> pd.DataFrame:
    """
    Load keyword-to-label mapping CSV and fill missing values.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
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
            # disease_type
            if row.get("disease_type"):
                for v in str(row["disease_type"]).split(";"):
                    v = v.strip()
                    if v:
                        votes["disease_type"].add(v)
            # stage_subtype
            if row.get("stage_subtype"):
                for v in str(row["stage_subtype"]).split(";"):
                    v = v.strip()
                    if v:
                        votes["stage_subtype"].add(v)
            # line_of_therapy
            if row.get("line_of_therapy"):
                for v in str(row["line_of_therapy"]).split(";"):
                    v = v.strip()
                    if v:
                        votes["line_of_therapy"].add(v)
            # biomarker
            if row.get("biomarker"):
                for v in str(row["biomarker"]).split(";"):
                    v = v.strip()
                    if v:
                        votes["biomarker"].add(v)
            matched.append(kw)

    # Ensure all LABEL_COLS are present, even if empty
    results = {k: sorted(list(votes.get(k, []))) for k in LABEL_COLS}

    return results, sorted(set(matched))
