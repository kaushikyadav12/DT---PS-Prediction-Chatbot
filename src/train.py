import sys
import pandas as pd
from pathlib import Path
from src.config import TRAIN_CSV, FEEDBACK_CSV, OUTPUTS_DIR, LABEL_COLS
from src.ml_pipeline import train_models

def read_csv_safe(path: str) -> pd.DataFrame:
    """Read CSV with UTF-8, fallback to cp1252 encoding."""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='cp1252')


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load main training dataset ---
    df_train = read_csv_safe(TRAIN_CSV)

    # --- Merge text columns ---
    df_train["text"] = (
            df_train.get("title", "") + " " +
            df_train.get("summary", "") + " " +
            df_train.get("inclusion_criteria", "")
    ).str.strip()

    # --- Reorder columns: put 'text' before first label ---
    cols = df_train.columns.tolist()
    first_label_idx = min([cols.index(c) for c in LABEL_COLS if c in cols])

    # Remove duplicates before reordering
    cols = [c for c in cols if c != "text"]  # remove any existing 'text'
    cols = [c for c in cols if c not in LABEL_COLS]  # remove label columns
    cols = cols[:first_label_idx] + ["text"] + cols[first_label_idx:] + LABEL_COLS
    df_train = df_train[cols]

    # --- Prepare feedback dataset ---
    feedback_path = Path(FEEDBACK_CSV)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    if not feedback_path.exists():
        pd.DataFrame(columns=["text"] + LABEL_COLS).to_csv(feedback_path, index=False)

    df_feedback = read_csv_safe(feedback_path)

    # --- Remove duplicates: feedback rows that exactly match train rows ---
    if not df_feedback.empty:
        merged = df_feedback.merge(df_train, on=["text"] + LABEL_COLS, how="left", indicator=True)
        df_feedback_filtered = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    else:
        df_feedback_filtered = df_feedback

    # --- Ensure unique columns before concatenation ---
    df_train = df_train.loc[:, ~df_train.columns.duplicated()]
    df_feedback_filtered = df_feedback_filtered.loc[:, ~df_feedback_filtered.columns.duplicated()]

    # --- Combine train + filtered feedback ---
    df_combined = pd.concat([df_train, df_feedback_filtered], ignore_index=True)

    # --- Train models ---
    def progress_callback(step, total, msg):
        print(f"[{step}/{total}] {msg}")
        return True

    try:
        train_models(df_combined, save_to_disk=True, progress_callback=progress_callback)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

    # --- Summary ---
    report_path = OUTPUTS_DIR / "run_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Training completed. Models saved under 'models/'.\n")
        f.write(f"Rows used (train + feedback): {len(df_combined)}\n")
        f.write(f"Targets (all multilabel): {', '.join(LABEL_COLS)}\n")

    print("✅ Training completed successfully.")


if __name__ == "__main__":
    main()
