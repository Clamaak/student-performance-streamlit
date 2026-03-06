import sqlite3
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Student G3 Predictor", layout="centered")

@st.cache_resource
def load_bundle():
    return joblib.load("gbr_student_model.joblib")

bundle = load_bundle()
model = bundle["model"]
abs_cap = bundle["abs_cap"]
fail_cap = bundle["fail_cap"]
columns = bundle["columns"]

# ---- Storage ----
DB_PATH = "predictions.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT    NOT NULL,
            source    TEXT    NOT NULL,
            G1        REAL,
            G2        REAL,
            absences  REAL,
            failures  REAL,
            school    TEXT,
            sex       TEXT,
            address   TEXT,
            internet  TEXT,
            predicted_g3 REAL NOT NULL
        )
    """)
    con.commit()
    con.close()

def save_prediction(user_input: dict, pred: float, source: str):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO predictions
           (timestamp, source, G1, G2, absences, failures, school, sex, address, internet, predicted_g3)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
            source,
            user_input.get("G1"),
            user_input.get("G2"),
            user_input.get("absences"),
            user_input.get("failures"),
            user_input.get("school"),
            user_input.get("sex"),
            user_input.get("address"),
            user_input.get("internet"),
            pred,
        ),
    )
    con.commit()
    con.close()

init_db()

# ---- Helper: build an input row with correct columns ----
def make_input_df(user_dict: dict) -> pd.DataFrame:
    row = pd.DataFrame([user_dict])

    for col in columns:
        if col not in row.columns:
            row[col] = None
    row = row[columns]

    if "absences" in row.columns and row.loc[0, "absences"] is not None:
        row.loc[0, "absences"] = min(float(row.loc[0, "absences"]), abs_cap)
    if "failures" in row.columns and row.loc[0, "failures"] is not None:
        row.loc[0, "failures"] = min(float(row.loc[0, "failures"]), fail_cap)

    return row

# ---- Read URL query params ----
qp = st.query_params
url_source = len(qp) > 0

def qp_int(key, default):
    try:
        return int(qp[key])
    except (KeyError, ValueError):
        return default

def qp_str(key, options, default):
    val = qp.get(key, default)
    return val if val in options else default

# ---- UI ----
st.title("Student Final Grade (G3) Predictor")
st.write("Predicts the final grade (G3) using a Gradient Boosting model.")

if url_source:
    st.info("Form pre-filled from URL parameters. Prediction saved automatically.")

with st.form("student_form"):
    st.subheader("Enter student information")

    G1       = st.number_input("G1 (first period grade)",  min_value=0, max_value=20,  value=qp_int("G1", 10))
    G2       = st.number_input("G2 (second period grade)", min_value=0, max_value=20,  value=qp_int("G2", 10))
    absences = st.number_input("Absences",                 min_value=0, max_value=200, value=qp_int("absences", 2))
    failures = st.number_input("Failures",                 min_value=0, max_value=10,  value=qp_int("failures", 0))

    school   = st.selectbox("School",   ["GP", "MS"],     index=["GP", "MS"].index(qp_str("school", ["GP", "MS"], "GP")))
    sex      = st.selectbox("Sex",      ["F", "M"],       index=["F", "M"].index(qp_str("sex", ["F", "M"], "F")))
    address  = st.selectbox("Address",  ["U", "R"],       index=["U", "R"].index(qp_str("address", ["U", "R"], "U")))
    internet = st.selectbox("Internet", ["yes", "no"],    index=["yes", "no"].index(qp_str("internet", ["yes", "no"], "yes")))

    submitted = st.form_submit_button("Predict G3")

# Auto-predict when URL params are present; manual predict on form submit
if submitted or url_source:
    user_input = {
        "G1": G1, "G2": G2, "absences": absences, "failures": failures,
        "school": school, "sex": sex, "address": address, "internet": internet,
    }

    X_user = make_input_df(user_input)
    pred = model.predict(X_user)[0]
    source = "url" if (url_source and not submitted) else "form"

    save_prediction(user_input, pred, source)

    st.success(f"Predicted G3: {pred:.2f}")
    st.caption(
        f"Note: absences capped at {abs_cap:.0f}, failures capped at {fail_cap:.0f} "
        "(same as training winsorization)."
    )

# ---- Prediction history ----
with st.expander("Prediction history"):
    con = sqlite3.connect(DB_PATH)
    df_log = pd.read_sql_query(
        "SELECT timestamp, source, G1, G2, absences, failures, school, sex, address, internet, predicted_g3 "
        "FROM predictions ORDER BY id DESC LIMIT 100",
        con,
    )
    con.close()
    if df_log.empty:
        st.write("No predictions saved yet.")
    else:
        st.dataframe(df_log, use_container_width=True)
