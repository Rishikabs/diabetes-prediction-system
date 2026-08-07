# ---------------------------------------------------------------------------
# Dockerfile - Diabetes Prediction System (Streamlit app)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Prevents Python from writing .pyc files & buffers stdout (good for logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install system dependencies needed by matplotlib/scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Train the model at image build time so the container starts instantly.
# (Safe to re-run: train_model.py regenerates data/model if missing.)
RUN python train_model.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
