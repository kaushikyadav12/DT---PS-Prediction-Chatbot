import sys
from pathlib import Path
import re
import streamlit as st
import pandas as pd
import threading
import time
import pyperclip

from streamlit.runtime.scriptrunner.script_runner import RerunException, RerunData

# --- Project paths ---
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.infer import predict
from src.config import MODELS_DIR, FEEDBACK_CSV, LABEL_COLS
from src.ml_pipeline import load_models

# --- Page Setup ---
st.set_page_config(page_title="DT - PS Chatbot", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
div.stButton > button {
    border-radius: 15px;
    padding: 0.3em 0.6em !important;
    font-weight: 500;
    background-color: lightgrey !important;
    color: black !important;
    border: 2px solid #ccc;
    font-size: 0.9em !important;
}
div.stButton > button:hover {
    background-color: #f0f0f0 !important;
    color: black !important;
}
.copied-msg {
    padding: 5px 10px;
    border-radius: 8px;
    background-color: #d4edda;
    color: #155724;
    font-weight: bold;
    display: inline-block;
    animation: fadeout 1.5s forwards;
}
@keyframes fadeout {
    0% {opacity:1;}
    100% {opacity:0;}
}
</style>
""", unsafe_allow_html=True)

# --- Session state initialization ---
for key, default in {
    "models": None,
    "latest_preds": {},
    "fb_values": {},
    "predicted_once": False,
    "user_text_input": "",
    "editable_boxes": {k: False for k in LABEL_COLS},
    "copied_box": None,
    "copied_time": None,
    "active_box": None,
    "editing_temp": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Normalization helpers ---
_ws_re = re.compile(r"\s+")
def _normalize_free_text(x: str) -> str:
    s = str(x if x is not None else "").strip()
    s = _ws_re.sub(" ", s).lower()
    return s

def _normalize_label_val(x: str) -> str:
    s = str(x if x is not None else "").strip()
    if "," in s:
        items = [i.strip().lower() for i in s.split(",") if i.strip()]
        items.sort()
        return ", ".join(items)
    return _ws_re.sub(" ", s).lower()

def feedback_exists(df: pd.DataFrame, user_text: str, new_vals: dict, label_cols: list[str]) -> bool:
    if df.empty:
        return False
    target_text = _normalize_free_text(user_text)
    target_labels = {k: _normalize_label_val(new_vals.get(k, "")) for k in label_cols}
    for _, row in df.iterrows():
        row_text = _normalize_free_text(row.get("text", ""))
        if row_text != target_text:
            continue
        same = True
        for k in label_cols:
            row_val = _normalize_label_val(row.get(k, ""))
            if row_val != target_labels.get(k, ""):
                same = False
                break
        if same:
            return True
    return False

def reset_app():
    st.session_state.clear()
    raise RerunException(RerunData())

# --- App Layout ---
st.title("🧬 Disease Type - Patient Segments Predictor")

disable_all = False

user_text = st.text_area(
    "Enter Clinical Trial Text:",
    value=st.session_state["user_text_input"],
    key="user_text_input",
    height=150,
    disabled=disable_all
)

btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    if st.button("🔮 Predict", key="predict_btn", disabled=disable_all):
        if not user_text.strip():
            st.error("⚠️ Please enter clinical trial text and Try again.")
        else:
            try:
                if st.session_state["models"] is None:
                    st.session_state["models"] = load_models()

                if not st.session_state["models"]:
                    st.error("⚠️ Something went wrong while preparing predictions. Please try again later.")
                else:
                    res = predict(user_text, models=st.session_state["models"])
                    if isinstance(res, dict) and "final" in res:
                        formatted_preds = {key: ", ".join(res["final"].get(key, [])) for key in LABEL_COLS}

                        # --- ✅ Reset to initial state ---
                        st.session_state["latest_preds"] = formatted_preds
                        st.session_state["fb_values"] = formatted_preds.copy()
                        st.session_state["predicted_once"] = True
                        st.session_state["editable_boxes"] = {k: False for k in LABEL_COLS}
                        st.session_state["active_box"] = None
                        st.session_state["editing_temp"] = {}
                        st.session_state["copied_box"] = None
                        st.session_state["copied_time"] = None

                    else:
                        st.error("⚠️ Something went wrong while preparing predictions. Please try again later.")
            except Exception:
                st.error("⚠️ Something went wrong while preparing predictions. Please try again later.")

with btn_col2:
    if st.session_state["predicted_once"]:
        if st.button("🔄 Reset", key="reset_btn"):
            reset_app()

# --- Predictions + Feedback ---
if st.session_state["predicted_once"] and st.session_state["latest_preds"]:
    st.subheader("📊 Predictions + Feedback (Copy/Edit)")
    st.info(
        "⚠️ If Predictions are Incorrect, Edit the predictions to submit feedback (Values should be semicolon (;) separated)."
    )

    cols = st.columns(len(LABEL_COLS))
    active_box = st.session_state["active_box"]

    for idx, key in enumerate(LABEL_COLS):
        with cols[idx]:
            editable = st.session_state["editable_boxes"].get(key, False)
            text_val = st.session_state["fb_values"].get(key, "")

            # --- Editable text area ---
            if editable:
                temp_val = st.session_state["editing_temp"].get(key, text_val)
                text_area_val = st.text_area(
                    label=key,
                    value=temp_val,
                    height=120,
                    key=f"box_{key}",
                    disabled=disable_all
                )
                st.session_state["editing_temp"][key] = text_area_val
            else:
                st.text_area(
                    label=key,
                    value=text_val,
                    height=120,
                    key=f"box_{key}_disabled",
                    disabled=True
                )

            col_copy, col_edit, col_done = st.columns([1, 1, 1], gap="small")

            # --- Dynamic button disabled logic ---
            copy_disabled = disable_all or (active_box == key)
            edit_disabled = disable_all or (active_box is not None)
            done_disabled = disable_all or not editable

            # --- Copy button ---
            if col_copy.button("📋 Copy", key=f"copy_{key}", disabled=copy_disabled):
                pyperclip.copy(st.session_state["fb_values"].get(key, ""))
                st.session_state["copied_box"] = key
                st.session_state["copied_time"] = time.time()
                placeholder = col_copy.empty()
                placeholder.markdown("<div class='copied-msg'>Copied!</div>", unsafe_allow_html=True)
                threading.Thread(target=lambda p=placeholder: (time.sleep(1.5), p.empty()), daemon=True).start()
                raise RerunException(RerunData())

            # --- Edit button ---
            if col_edit.button("✏️ Edit", key=f"edit_{key}", disabled=edit_disabled):
                st.session_state["editable_boxes"] = {k: k == key for k in LABEL_COLS}
                st.session_state["active_box"] = key
                raise RerunException(RerunData())

            # --- Done button ---
            if editable and col_done.button("✅ Done", key=f"done_{key}", disabled=done_disabled):
                st.session_state["fb_values"][key] = st.session_state["editing_temp"].get(key, text_val)
                st.session_state["editable_boxes"][key] = False
                st.session_state["active_box"] = None
                st.session_state["editing_temp"].pop(key, None)
                raise RerunException(RerunData())

    # --- Feedback submission ---
    edited = any(
        st.session_state["fb_values"].get(k, "").strip()
        != st.session_state["latest_preds"].get(k, "").strip()
        for k in LABEL_COLS
    )

    if edited:
        if st.button("💾 Submit Feedback", key="feedback_btn", disabled=disable_all):
            new_vals = {k: st.session_state["fb_values"].get(k, "").strip() for k in LABEL_COLS}
            user_text_val = st.session_state.get("user_text_input", "").strip()

            if not any(new_vals.values()):
                st.error("⚠️ Feedback cannot be empty. Please enter at least one value in the boxes.")
            else:
                if Path(FEEDBACK_CSV).exists():
                    # ✅ Use cp1252 encoding
                    df_feedback = pd.read_csv(FEEDBACK_CSV, dtype=str, keep_default_na=False, encoding='cp1252')
                else:
                    df_feedback = pd.DataFrame(columns=["text"] + LABEL_COLS)

                if feedback_exists(df_feedback, user_text_val, new_vals, LABEL_COLS):
                    st.warning("⚠️ This clinical trial info and feedback is already submitted.")
                else:
                    new_row = pd.DataFrame([{"text": user_text_val, **new_vals}], dtype=str)
                    df_feedback = pd.concat([df_feedback, new_row], ignore_index=True)
                    df_feedback.to_csv(FEEDBACK_CSV, index=False, encoding='cp1252')
                    st.success("✅ Feedback submitted successfully!")
