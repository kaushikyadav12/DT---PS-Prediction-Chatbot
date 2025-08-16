from pathlib import Path
import pandas as pd

# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train" / "clinical_trials_train.csv"
TEST_CSV = DATA_DIR / "test" / "clinical_trials_test.csv"
KEYWORDS_CSV = DATA_DIR / "keywords_mapping.csv"
FEEDBACK_CSV = DATA_DIR / "feedback.csv"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Columns ---
TEXT_COLS = ["title", "summary", "inclusion_criteria"]
# All labels are now MULTILABEL (lists of values possible)
LABEL_COLS = ["disease_type", "stage_subtype", "line_of_therapy", "biomarker"]

# --- Other configs ---
RANDOM_STATE = 42

# --- Ensure folders exist ---
for folder in [DATA_DIR / "train", DATA_DIR / "test", FEEDBACK_CSV.parent, OUTPUTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- Ensure CSVs exist ---
for path, cols in [
    (TRAIN_CSV, TEXT_COLS + LABEL_COLS),
    (TEST_CSV, TEXT_COLS + LABEL_COLS),
    (FEEDBACK_CSV, ["text"] + LABEL_COLS)
]:
    if not path.exists():
        pd.DataFrame(columns=cols).to_csv(path, index=False)
