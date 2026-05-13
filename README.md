# Customer Churn Prediction API

[![CI](https://github.com/GabrielMacGregor/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrielMacGregor/churn-prediction/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)](https://mlflow.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)

Production-grade ML system that predicts customer churn for a telecom company. The goal
is to demonstrate the full ML lifecycle: downloading data, preprocessing, training,
tracking, versioning, serving, monitoring, and testing.

Project progress is tracked in [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Architecture

```text
Raw Kaggle CSV
  -> Preprocessing with scikit-learn
  -> XGBoost training with cross-validation
  -> MLflow Tracking
  -> MLflow Model Registry
  -> Exported local model artifact
  -> FastAPI /predict
  -> SQLite prediction audit log
  -> Docker / Railway
```

## Features

| Feature | Detail |
|---|---|
| Data ingestion | Kaggle API download via `python -m src.download_data` |
| Training | XGBoost with 5-fold stratified cross-validation |
| Tracking | MLflow logs params, metrics, reports, and model artifacts |
| Versioning | MLflow Model Registry with ROC-AUC gated `production` alias promotion |
| REST API | FastAPI with Swagger docs at `/docs` |
| Validation | Pydantic schema for the customer payload |
| Monitoring | `/logs` and `/logs/summary` backed by SQLite prediction audit logs |
| Security | Optional `X-API-Key` authentication through `API_KEY` |
| Infra | Docker Compose for data download, MLflow, training, and API |
| Deployment artifact | Exported local `model_artifact/` for self-contained Docker images |
| CI | GitHub Actions runs Ruff and Pytest |

## Dataset

This project uses the public Telco Customer Churn dataset from Kaggle:

```text
blastchar/telco-customer-churn
```

Expected local file:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The raw CSV is excluded from Git.

## Kaggle API Setup

Create a Kaggle API token from your Kaggle account settings and configure either:

1. A local `.env` file:

```bash
copy .env.example .env
```

Then fill:

```text
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

2. Or the standard Kaggle credentials file:

```text
%USERPROFILE%\.kaggle\kaggle.json
```

Never commit Kaggle credentials.

## Quick Start With Docker

```bash
# 1. Configure Kaggle credentials
copy .env.example .env

# 2. Download the dataset
docker compose --profile data run --rm download-data

# 3. Start MLflow
docker compose up -d mlflow

# 4. Train and register the model
docker compose --profile training run --rm train

# 5. Export the promoted production model to a local artifact
docker compose --profile export run --rm export-model

# 6. Start the API
docker compose up -d api
```

Open:

- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`
- MLflow UI: `http://localhost:5000`

The API attempts a lazy model load on `/predict` only when no model was loaded at
startup. New model versions require an API restart until an admin reload endpoint is
added.

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Download the dataset (needs Kaggle credentials in .env)
copy .env.example .env
python -m src.download_data

python -m src.preprocess
python -m src.train

# Optional for local-path serving and required before production image builds
python -m src.export_model

uvicorn api.main:app --reload
```

To open the MLflow UI locally:

```bash
mlflow ui --port 5000
```

For notebook work:

```bash
pip install -r requirements-dev.txt
jupyter notebook notebooks/eda.ipynb
```

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_uri": "/app/model_artifact"
}
```

### `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

Example response:

```json
{
  "prediction": 1,
  "churn": true,
  "churn_probability": 0.82,
  "threshold": 0.5,
  "model_uri": "/app/model_artifact",
  "model_version": null
}
```

### `GET /logs`

Returns recent prediction audit records. Supports `limit`, `offset`, and optional
`prediction=0|1` filters.

```bash
curl "http://localhost:8000/logs?limit=10"
```

### `GET /logs/summary`

Returns aggregate monitoring counters.

Use `window_hours` to summarize only recent predictions:

```bash
curl "http://localhost:8000/logs/summary?window_hours=24"
```

```json
{
  "total_predictions": 42,
  "positive_predictions": 18,
  "churn_rate": 0.4286,
  "positive_prediction_rate": 0.4286,
  "average_churn_probability": 0.57,
  "average_churn_probability_prediction_0": 0.31,
  "average_churn_probability_prediction_1": 0.79,
  "latest_prediction_at": "2026-05-13T16:42:00Z",
  "window_hours": 24,
  "window_start_at": "2026-05-12T16:42:00Z"
}
```

If `API_KEY` is configured, send it on protected endpoints:

```bash
curl "http://localhost:8000/logs/summary" \
  -H "X-API-Key: your_api_key"
```

Authenticated recent log example:

```bash
curl "http://localhost:8000/logs?limit=20&prediction=1" \
  -H "X-API-Key: your_api_key"
```

## Project Structure

```text
churn-prediction-api/
|-- api/
|   |-- main.py
|   `-- schema.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- examples/
|   `-- predict_request.json
|-- notebooks/
|   `-- eda.ipynb
|-- reports/
|-- src/
|   |-- config.py
|   |-- download_data.py
|   |-- evaluate.py
|   |-- export_model.py
|   |-- preprocess.py
|   `-- train.py
|-- tests/
|   |-- test_api.py
|   `-- test_preprocess.py
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
`-- requirements-dev.txt
```

## Model Promotion

Every training run registers a candidate model. The candidate is promoted to the
`production` alias only when its `test_roc_auc` is greater than or equal to the model
currently in production. If no production model exists, the first registered candidate
is promoted.

The promotion decision is saved in `reports/training_metrics.json` and tagged on the
MLflow model version.

## Export Production Model Artifact

Deployment does not require a live MLflow tracking server. After training promotes a
model to the `production` alias, export that model to a generated local artifact:

```bash
python -m src.export_model
```

With Docker Compose:

```bash
docker compose --profile export run --rm export-model
```

This writes an MLflow model directory to `model_artifact/`. The directory is ignored by
Git because it is generated binary output, but it is intentionally included in Docker
build contexts so production images can load `MODEL_URI=/app/model_artifact`.

Re-run the export command after promoting a new production model, then rebuild/restart
the API image to serve the new version.

## Current Model Performance

First registered production model:

| Metric | CV mean | CV std | Test |
|---|---:|---:|---:|
| ROC-AUC | 0.8433 | 0.0099 | 0.8418 |
| F1 | 0.5834 | 0.0293 | 0.5859 |
| Precision | 0.6647 | 0.0384 | 0.6644 |
| Recall | 0.5204 | 0.0285 | 0.5241 |
| Accuracy | 0.8028 | 0.0143 | 0.8034 |

The full report is generated at `reports/training_metrics.json`.

## Train/Test Splits

`python -m src.preprocess` writes deterministic train/test CSVs for inspection and
debugging. `src.train` rebuilds the same split from raw data using the same seed so
training stays reproducible without depending on generated artifacts.

## Prediction Logs

Each successful prediction is written to SQLite with:

- `created_at`
- request `payload`
- `churn_probability`
- `threshold`
- `prediction`
- `model_uri`

Local default:

```text
data/processed/predictions.db
```

Read logs through:

- `GET /logs`
- `GET /logs/summary`
- `GET /logs/summary?window_hours=24`

## Tests And Lint

```bash
ruff check .
pytest -q
```

Ruff intentionally excludes `notebooks/`, because notebooks are exploratory artifacts
and should not block API or training CI.

## Railway Notes

For Railway, the preferred production path is a self-contained image that includes a
generated local model artifact. Train and track with MLflow locally, export the
production model to `model_artifact/`, copy it into the image, and load it with a local
`MODEL_URI`.

Set these variables in Railway:

| Variable | Value |
|---|---|
| `MODEL_URI` | `/app/model_artifact` |
| `LOG_DB_PATH` | `/data/predictions.db` |
| `API_KEY` | Secret value required in `X-API-Key` when enabled |

For production, use a persistent database or storage volume for logs. SQLite is
intentionally simple for local development and portfolio demos.

## Next Steps

- Add drift checks comparing recent prediction payloads with the training data profile.
- Add a scheduled retraining workflow.
