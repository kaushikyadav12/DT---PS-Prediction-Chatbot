from .ensemble import merge_predictions
from .rule_based import load_mapping, match_keywords
import numpy as np

# Load rule-based keyword mapping safely
try:
    _keyword_mapping = load_mapping()  # uses cp1252 fallback internally
except Exception as e:
    print(f"⚠️ Failed to load keyword mapping: {e}")
    _keyword_mapping = None


def _empty_outputs(models: dict) -> dict:
    """
    Return empty outputs for all labels.
    Assumes all labels are multilabel.
    """
    if not models:
        return {"final": {}, "rule_based": {}, "ml_only": {}, "provenance": {}, "explanations": {}}

    return {
        "final": {k: [] for k in models.keys()},
        "rule_based": {k: [] for k in models.keys()},
        "ml_only": {k: [] for k in models.keys()},
        "provenance": {k: {} for k in models.keys()},
        "explanations": {
            "matched_keywords": [],
            "tfidf_top_terms": {k: [] for k in models.keys()}
        }
    }


def predict(text: str, models: dict) -> dict:
    """
    Predict multilabel outputs for all labels using rule-based and ML models.
    Returns merged predictions, provenance, and explanations.
    """
    if not isinstance(text, str) or not text.strip():
        return _empty_outputs(models)

    # --- Rule-based predictions ---
    try:
        if _keyword_mapping is not None:
            rule_preds, matched_keywords = match_keywords(text, _keyword_mapping)
            # Ensure each rule prediction is a list
            rule_preds = {k: v if isinstance(v, list) else [v] for k, v in rule_preds.items()}
        else:
            rule_preds, matched_keywords = {k: [] for k in models.keys()}, []
    except Exception:
        rule_preds, matched_keywords = {k: [] for k in models.keys()}, []

    # --- ML predictions ---
    ml_preds = {}
    for key, model_info in models.items():
        if not isinstance(model_info, dict):
            model_info = {"pipeline": model_info, "mlb": None}

        pipe = model_info.get("pipeline")
        mlb = model_info.get("mlb")

        if pipe is None:
            ml_preds[key] = []
            continue

        try:
            pred = pipe.predict([text])
            if mlb is not None:
                labels = mlb.inverse_transform(pred)
                ml_preds[key] = list(labels[0]) if labels and labels[0] else []
            else:
                # Ensure always a list
                if isinstance(pred, (list, np.ndarray)):
                    ml_preds[key] = [str(p) for p in pred.flatten()]
                else:
                    ml_preds[key] = [str(pred)]
        except Exception:
            ml_preds[key] = []

    # --- Merge rule-based + ML predictions ---
    final, provenance = merge_predictions(rule_preds, ml_preds)

    # --- Explanations ---
    explanations = {
        "matched_keywords": matched_keywords,
        "tfidf_top_terms": {k: [] for k in models.keys()}  # placeholder
    }

    return {
        "final": final,
        "rule_based": rule_preds,
        "ml_only": ml_preds,
        "provenance": provenance,
        "explanations": explanations
    }
