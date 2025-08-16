from typing import Dict, Any, Optional, Callable
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from joblib import dump, load
from src.config import MODELS_DIR, TEXT_COLS, LABEL_COLS
from src.preprocessing import combine_text, split_multilabel

def _make_vectorizer():
    return TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=20000)

def _clf_multilabel():
    return OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))

def train_models(
    df: pd.DataFrame,
    load_existing: bool = False,
    save_to_disk: bool = True,
    progress_callback: Optional[Callable[[int,int,str], bool]] = None
) -> Dict[str, Dict[str, Any]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X = combine_text(df, TEXT_COLS)
    artifacts: Dict[str, Dict[str, Any]] = {}

    # Treat ALL labels as multilabel
    tasks = [
        ("disease_type", "multilabel"),
        ("line_of_therapy", "multilabel"),
        ("stage_subtype", "multilabel"),
        ("biomarker", "multilabel")
    ]
    total_steps = len(tasks) * 3
    step_counter = 0

    if progress_callback is None:
        def progress_callback(step, total, msg):
            print(f"[{step}/{total}] {msg}")
            return True

    for label, task_type in tasks:
        model_path = MODELS_DIR / f"{label}.joblib"
        task_name = f"{label} ({task_type})"

        if load_existing and model_path.exists():
            pipe, mlb = load(model_path)
        else:
            # --- Prepare multilabel target ---
            y = df[label].fillna("").map(split_multilabel)
            step_counter += 1
            if not progress_callback(step_counter, total_steps, f"Initializing {task_name}"):
                return artifacts

            mlb = MultiLabelBinarizer()
            Y = mlb.fit_transform(y)
            step_counter += 1
            if not progress_callback(step_counter, total_steps, f"Fitting ML Binarizer {task_name}"):
                return artifacts

            pipe = Pipeline([("tfidf", _make_vectorizer()), ("clf", _clf_multilabel())])
            step_counter += 1
            if not progress_callback(step_counter, total_steps, f"Fitting Classifier {task_name}"):
                return artifacts

            pipe.fit(X, Y)

            if save_to_disk:
                if not progress_callback(step_counter, total_steps, f"Saving {task_name}"):
                    return artifacts
                dump((pipe, mlb), model_path)

        artifacts[label] = {"pipeline": pipe, "mlb": mlb}

    return artifacts

def load_models() -> Dict[str, Dict[str, Any]]:
    models = {}
    for label in LABEL_COLS:
        path = MODELS_DIR / f"{label}.joblib"
        if path.exists():
            pipe, mlb = load(path)
            models[label] = {"pipeline": pipe, "mlb": mlb}
    return models
