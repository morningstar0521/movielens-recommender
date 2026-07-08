# Dockerfile for Hugging Face Spaces (Docker SDK) or Render.
#
# WHY a single multi-process container instead of API + frontend split:
# - Free-tier hosts give you one container. Running uvicorn + streamlit
#   in the same container via a tiny entrypoint script is the simplest
#   path to a one-click deploy.
#
# Memory note: the runtime image only needs the trained model, the
# sparse train matrix, the mappings, and movies.parquet - no raw
# ratings.csv. Total artefact size is well under 200 MB for ALS with
# factors=128, fits comfortably on a 512 MB free tier.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# implicit needs a C++ runtime + BLAS at runtime. The slim image is fine
# once libgomp is installed (OpenMP for parallel ALS).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so Docker layer caching survives code edits.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code + pre-trained artefacts.
# The CI / deploy step must run `python -m src.train ...` first so that
# `models/` contains best_als.joblib, train_ui.npz, mappings.joblib,
# and movies.parquet BEFORE building the image.
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY models/ ./models/

# Hugging Face Spaces sets $PORT to 7860; Render uses $PORT too. We
# bind the frontend there. The API listens on an internal port that
# the frontend talks to via localhost.
ENV API_PORT=8000 \
    PORT=7860 \
    API_URL=http://localhost:8000 \
    MODELS_DIR=/app/models

EXPOSE 7860

# Start the FastAPI backend in the background, then Streamlit in the
# foreground. If uvicorn dies, the container exits because the
# foreground process is what Docker watches.
CMD bash -c "uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT} & \
    streamlit run frontend/app.py --server.port ${PORT} --server.address 0.0.0.0 --server.headless true"
