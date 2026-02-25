import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student G3 Predictor", layout="centered")
st.title("✅ App started")
st.write("If you see this, Streamlit is running.")

@st.cache_resource
def load_bundle():
    return joblib.load("gbr_student_model.joblib")

bundle = load_bundle()
model = bundle["model"]
abs_cap = bundle["abs_cap"]
fail_cap = bundle["fail_cap"]
columns = bundle["columns"]

st.title("🎓 Student Final Grade (G3) Predictor")
st.write("This app predicts the final grade (G3) using a Gradient Boosting model.")

# ---- Helper: build an input row with correct columns ----
def make_input_df(user_dict: dict) -> pd.DataFrame:
    row = pd.DataFrame([user_dict])

    # Ensure all expected columns exist (in correct order)
    for col in columns:
        if col not in row.columns:
            row[col] = None
    row = row[columns]

    # Apply the same capping (winsorization) used in training
    if "absences" in row.columns and row.loc[0, "absences"] is not None:
        row.loc[0, "absences"] = min(float(row.loc[0, "absences"]), abs_cap)
    if "failures" in row.columns and row.loc[0, "failures"] is not None:
        row.loc[0, "failures"] = min(float(row.loc[0, "failures"]), fail_cap)

    return row

# ---- Build UI form ----
with st.form("student_form"):
    st.subheader("Enter student information")

    # Common numeric inputs (adjust min/max if needed)
    G1 = st.number_input("G1 (first period grade)", min_value=0, max_value=20, value=10)
    G2 = st.number_input("G2 (second period grade)", min_value=0, max_value=20, value=10)
    absences = st.number_input("Absences", min_value=0, max_value=200, value=2)
    failures = st.number_input("Failures", min_value=0, max_value=10, value=0)

    # A few example categorical inputs (you can add more)
    school = st.selectbox("School", ["GP", "MS"])
    sex = st.selectbox("Sex", ["F", "M"])
    address = st.selectbox("Address", ["U", "R"])
    internet = st.selectbox("Internet", ["yes", "no"])

    submitted = st.form_submit_button("Predict G3")

if submitted:
    # Build input dict. Add any other features your dataset has.
    user_input = {
        "G1": G1,
        "G2": G2,
        "absences": absences,
        "failures": failures,
        "school": school,
        "sex": sex,
        "address": address,
        "internet": internet,
    }

    # Create a one-row dataframe, aligned to training columns
    X_user = make_input_df(user_input)

    # Predict
    pred = model.predict(X_user)[0]

    st.success(f"Predicted G3: {pred:.2f}")

    st.caption(
        f"Note: absences capped at {abs_cap:.0f}, failures capped at {fail_cap:.0f} "
        "(same as training winsorization)."
    )
