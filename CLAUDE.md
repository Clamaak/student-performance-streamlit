# CLAUDE.md — Student G3 Predictor

## Project overview

A Streamlit web app that predicts a student's final grade (G3) using a Gradient Boosting Regressor trained on the Student Performance dataset. Predictions are stored in a local SQLite database for logging and review.

## Environment

Always run the app using the **Beroepsprofiel** conda environment:

```bash
conda activate Beroepsprofiel
streamlit run app.py
```

The model file (`gbr_student_model.joblib`) was saved with:
- scikit-learn 1.7.2
- numpy 2.3.4

These versions are installed in the `Beroepsprofiel` environment. Using any other environment (e.g. Anaconda base or `py3_11_indatad`) will cause a `ModuleNotFoundError` or `ValueError` when loading the model.

## Project structure

```
Regression/
├── app.py                              # Streamlit app with prediction UI and SQLite storage
├── gbr_student_model.joblib            # Trained model bundle (model + preprocessing metadata)
├── student_performance.csv             # Training dataset
├── Student_Performance_regression.ipynb # Training notebook
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container definition for deployment
├── predictions.db                      # SQLite database (auto-created on first run)
└── streamlit_demo.py                   # Earlier demo version of the app
```

## What was added: SQLite storage layer

### Why

Every time a user submits the prediction form (or hits the app via URL parameters), the input data and resulting prediction were not being saved anywhere. The SQLite layer was added to log every prediction for later review.

### What changed in `app.py`

Three things were added:

**1. Database initialisation (`init_db`)**

```python
DB_PATH = "predictions.db"

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            source       TEXT    NOT NULL,
            G1           REAL,
            G2           REAL,
            absences     REAL,
            failures     REAL,
            school       TEXT,
            sex          TEXT,
            address      TEXT,
            internet     TEXT,
            predicted_g3 REAL NOT NULL
        )
    """)
    con.commit()
    con.close()

init_db()  # called once at app startup
```

`CREATE TABLE IF NOT EXISTS` means this is safe to call on every page load — it only creates the table the first time.

**2. Save function (`save_prediction`)**

```python
def save_prediction(user_input: dict, pred: float, source: str):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO predictions
           (timestamp, source, G1, G2, absences, failures, school, sex, address, internet, predicted_g3)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (datetime.utcnow().isoformat(), source, ...)
    )
    con.commit()
    con.close()
```

Uses parameterised queries (`?` placeholders) to prevent SQL injection. The `source` field records whether the prediction came from the web form (`"form"`) or from URL query parameters (`"url"`).

**3. Prediction history expander**

```python
with st.expander("Prediction history"):
    con = sqlite3.connect(DB_PATH)
    df_log = pd.read_sql_query(
        "SELECT ... FROM predictions ORDER BY id DESC LIMIT 100", con
    )
    con.close()
    st.dataframe(df_log, use_container_width=True)
```

Shows the last 100 predictions in a collapsible section at the bottom of the page.

### How it fits into the prediction flow

```
User submits form  ──►  make_input_df()  ──►  model.predict()
                                                     │
                                               save_prediction()
                                                     │
                                               predictions.db
                                                     │
                                          shown in history expander
```

### The `source` field

| Value  | Meaning                                      |
|--------|----------------------------------------------|
| `form` | User clicked the Predict button manually      |
| `url`  | App was called with URL query parameters      |

URL parameter example:
```
http://localhost:8501/?G1=14&G2=15&absences=3&failures=0&school=GP&sex=F&address=U&internet=yes
```

## Dependencies

```
streamlit
pandas
numpy
scikit-learn==1.7.2
joblib
```

SQLite is part of Python's standard library — no extra install needed.

## Deployment

The project includes a `Dockerfile` for containerised deployment (e.g. Render). Note that `predictions.db` is written to the local filesystem inside the container, so predictions are lost on container restart unless a persistent volume is mounted.
