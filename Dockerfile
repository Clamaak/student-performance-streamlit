FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files (and model file if you have one)
COPY app.py .
COPY gbr_student_model.joblib .


# Render sets $PORT for web services;
CMD ["sh","-c","streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]