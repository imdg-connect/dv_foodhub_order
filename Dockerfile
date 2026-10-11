# ---- stage 1: build the React UI ----
FROM node:20-slim AS ui
WORKDIR /ui
COPY foodhub-ui/package*.json ./
RUN npm ci
COPY foodhub-ui/ ./
RUN npm run build

# ---- stage 2: Python app (FastAPI serves the API + the built UI) ----
FROM python:3.11-slim

# Hugging Face runs containers as user 1000: the app folder must be writable (live feed edits the CSV, EDA writes processed/)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    USE_EMBEDDINGS=0 \
    PORT=7860
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
COPY --chown=user --from=ui /ui/dist ./foodhub-ui/dist

# make sure the folder the EDA writes into exists and is writable
RUN mkdir -p processed

EXPOSE 7860
# listens on $PORT (Render sets it; 7860 elsewhere). ONE process on purpose: the live feed (a thread) and its status live inside it
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]