# Customer Churn Prediction API

[![CI](https://github.com/GabrielMacGregor/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/GabrielMacGregor/churn-prediction/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3110/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-blue)](https://mlflow.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)

Production-grade ML system that predicts customer churn for a telecom company. The goal
is to demonstrate the full ML lifecycle: downloading data, preprocessing, training,
tracking, versioning, serving, monitoring, and testing.

## Architecture

```text
Raw Kaggle CSV
  -> Preprocessing with scikit-learn
  -> XGBoost training with cross-validation
  -> MLflow Tracking
  -> MLflow Model Registry
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
| Monitoring | SQLite prediction log with timestamp, payload, threshold, and score |
| Infra | Docker Compose for data download, MLflow, training, and API |
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

# 5. Start the API
docker compose up -d api
```

Open:

- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`
- MLflow UI: `http://localhost:5000`

The API also attempts a lazy model reload on `/predict`, so if the API started before
the model was registered, the first prediction after training can load the new model.

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Download the dataset (needs Kaggle credentials in .env)
pip install -r requirements-data.txt
copy .env.example .env
python -m src.download_data

python -m src.preprocess
python -m src.train
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
  "model_uri": "models:/telco-churn-xgb@production"
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
  "model_uri": "models:/telco-churn-xgb@production",
  "model_version": "production"
}
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
|   |-- preprocess.py
|   `-- train.py
|-- tests/
|   |-- test_api.py
|   `-- test_preprocess.py
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- requirements.txt
|-- requirements-data.txt
`-- requirements-dev.txt
```

## Model Promotion

Every training run registers a candidate model. The candidate is promoted to the
`production` alias only when its `test_roc_auc` is greater than or equal to the model
currently in production. If no production model exists, the first registered candidate
is promoted.

The promotion decision is saved in `reports/training_metrics.json` and tagged on the
MLflow model version.

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

## Tests And Lint

```bash
ruff check .
pytest -q
```

Ruff intentionally excludes `notebooks/`, because notebooks are exploratory artifacts
and should not block API or training CI.

## Railway Notes

Set these variables in Railway:

| Variable | Value |
|---|---|
| `MLFLOW_TRACKING_URI` | Your MLflow server URL |
| `MODEL_URI` | `models:/telco-churn-xgb@production` |
| `LOG_DB_PATH` | `/data/predictions.db` |

For production, use a persistent database or storage volume for logs and MLflow
metadata. SQLite is intentionally simple for local development and portfolio demos.

## Next Steps

- Add an authenticated `/logs` endpoint for prediction monitoring.
