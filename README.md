# Oncology Clinical-Trials Chatbot (Python + Streamlit)

A web-based AI chatbot that predicts **Disease Type, Stage/Subtype, Line of Therapy, and Biomarker/Mutation** from user-entered clinical trial text.

It combines:

* **Rule-based layer:** keyword mapping (`keywords_mapping.csv`)
* **Machine-learning layer:** TF‑IDF + Logistic Regression (multi-class & multi-label)
* **Ensemble:** merges rule-based and ML predictions with provenance tracking

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### 2. Train models

Uses the data in `data/train/`.

```bash
python -m src.train
```

This creates:

* Model artifacts in `models/`
* Training summary in `outputs/run_summary.txt`

### 3. Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit. Enter clinical trial text and the chatbot will predict labels


## Project Structure

```
oncology_chatbot/
├─ app/
│  └─ streamlit_app.py         # Web UI (chat-style)
├─ data/
│  ├─ keywords_mapping.csv     # Rule-based mapping
│  ├─ feedback.csv             # feedback sheet
│  ├─ train/clinical_trials_train.csv
│  └─ test/clinical_trials_test.csv
├─ models/                     # Saved model artifacts (.joblib)
├─ outputs/                    # Training reports/metrics
├─ src/
│  ├─ config.py                # Global paths & constants
│  ├─ preprocessing.py         # Cleaning & normalization
│  ├─ rule_based.py            # Keyword matcher
│  ├─ ml_pipeline.py           # TF-IDF + LogisticRegression models
│  ├─ ensemble.py              # Merge rule-based + ML predictions
│  ├─ infer.py                 # Inference entry point
│  ├─ train.py                 # Training script
│  └─ utils.py                 # Helper functions (I/O, metrics)
├─ tests/
│  └─ test_preprocessing.py
├─ requirements.txt
├─ Makefile
└─ LICENSE
```

## Data Format

### `data/keywords_mapping.csv`

| keyword | disease\_type              | stage\_subtype | line\_of\_therapy | biomarker |
| ------- | -------------------------- | -------------- | ----------------- | --------- |
| nsclc   | Non-small cell lung cancer | Stage IV       | first-line        | EGFR      |
| egfr    | Non-small cell lung cancer |                |                   | EGFR      |
| her2    | Breast cancer              |                |                   | HER2      |
| kras    | Colorectal cancer          |                |                   | KRAS      |

* Blank = not applicable
* Multiple values in a column = semicolon-separated (`;`)

### `clinical_trials_[train|test].csv`

Columns:

```
trial_id, title, summary, inclusion_criteria, disease_type, stage_subtype, line_of_therapy, biomarker
```

* Multi-label columns (stage\_subtype, biomarker) can be semicolon-separated
* Must match `LABEL_COLS` in `src/config.py`

## Modeling

* **Text features:** TF‑IDF (word bi-grams, small defaults for demo)

* **Models:**
  \| Label | Type | Model |
  \|-------|------|-------|
  \| disease\_type | multi-class | LogisticRegression |
  \| line\_of\_therapy | multi-class | LogisticRegression |
  \| stage\_subtype | multi-label | OneVsRest(LogisticRegression) |
  \| biomarker | multi-label | OneVsRest(LogisticRegression) |

* **Ensemble:** merges rule-based + ML predictions

* Rule-based matches are preferred if available.
