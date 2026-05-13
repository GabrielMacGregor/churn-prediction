# Project Plan

Living plan for the Customer Churn Prediction API. Update this file whenever a
milestone is completed, a technical decision changes, or the next work block becomes
clearer.

## Current Status

| Area | Status | Notes |
|---|---|---|
| Repository | Done | Published at `GabrielMacGregor/churn-prediction` |
| Data ingestion | Done | Kaggle API downloader implemented in `src/download_data.py` |
| Preprocessing | Done | Cleaning, target encoding, deterministic train/test split |
| Training | Done | XGBoost pipeline with cross-validation |
| Tracking | Done | MLflow experiment, metrics, artifacts, model registry |
| Promotion | Done | Production alias gated by `test_roc_auc` |
| Serving | Done | FastAPI `/health` and `/predict` |
| Monitoring | Done | `/logs` and `/logs/summary` backed by SQLite |
| Security | Done | Optional `X-API-Key` auth through `API_KEY` |
| Tests | Done | API and preprocessing tests passing |
| Docker | Partial | Compose exists; local Docker validation still pending |
| Deployment | Next | Railway deployment not started |

## Completed Milestones

- [x] Create project scaffold with `src/`, `api/`, `tests/`, `data/`, `reports/`.
- [x] Add Kaggle API data download flow.
- [x] Add `.env.example` and keep local `.env` ignored.
- [x] Implement Telco data cleaning and feature/target split.
- [x] Build sklearn preprocessing pipeline with numeric and categorical handling.
- [x] Train XGBoost model with stratified cross-validation.
- [x] Log params, metrics, reports, and model artifact to MLflow.
- [x] Register model in MLflow Model Registry.
- [x] Promote model to `production` only when ROC-AUC is not worse than current production.
- [x] Add FastAPI `/health` endpoint.
- [x] Add FastAPI `/predict` endpoint with Pydantic validation.
- [x] Add prediction logging with timestamp, payload, probability, threshold, and model URI.
- [x] Add `/logs` endpoint for recent prediction audit records.
- [x] Add `/logs/summary` endpoint for aggregate monitoring counters.
- [x] Add optional API key authentication.
- [x] Add Dockerfile and Docker Compose services for data, MLflow, train, and API.
- [x] Add GitHub Actions CI for lint and tests.
- [x] Publish initial project to GitHub.

## Active Work Queue

### 1. Validate Docker Flow

Goal: prove the README Docker workflow works end to end.

- [ ] Run `docker compose --profile data run --rm download-data`.
- [ ] Run `docker compose up -d mlflow`.
- [ ] Run `docker compose --profile training run --rm train`.
- [ ] Run `docker compose up -d api`.
- [ ] Verify `GET /health`.
- [ ] Verify `POST /predict`.
- [ ] Verify `GET /logs/summary`.

Notes:
- Docker was not available in the local PATH during the first implementation pass.

### 2. Check GitHub CI

Goal: make sure the published repo passes CI on GitHub.

- [ ] Open the Actions tab.
- [ ] Confirm dependency install succeeds.
- [ ] Confirm `ruff check .` passes.
- [ ] Confirm `pytest -q` passes.
- [ ] Fix workflow paths if GitHub treats the repo differently than local.

### 3. Add Monitoring Polish

Goal: make monitoring more useful for a product team story.

- [ ] Add `GET /logs/summary?window_hours=24`.
- [ ] Add positive prediction rate over a recent time window.
- [ ] Add average churn probability by prediction label.
- [ ] Add README examples for authenticated `/logs` calls.

### 4. Add Model Explainability

Goal: make the model easier to explain to recruiters and stakeholders.

- [ ] Export feature importance after training.
- [ ] Save feature importance to `reports/feature_importance.csv`.
- [ ] Add feature importance plot to MLflow artifacts.
- [ ] Document top drivers in the README after the next training run.

### 5. Prepare Deployment

Goal: make the app deployable without local assumptions.

- [ ] Decide how MLflow artifacts are hosted for Railway.
- [ ] Add Railway-specific deployment notes.
- [ ] Validate environment variables for deployed API.
- [ ] Decide whether logs stay SQLite or move to Postgres.
- [ ] Add a production health check note for missing model behavior.

## Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Model | XGBoost | Strong baseline for tabular churn data |
| Pipeline | sklearn `Pipeline` + `ColumnTransformer` | Keeps preprocessing and model versioned together |
| Tracking | MLflow | Shows experiment tracking and registry workflow |
| Promotion | ROC-AUC gated alias | Avoids blindly replacing production with a worse model |
| API | FastAPI + Pydantic | Clear contract, validation, and Swagger docs |
| Logs | SQLite | Simple local audit trail for portfolio and development |
| Auth | Optional API key | Easy to demo locally, stricter when deployed |
| Data | Kaggle API | Reproducible data ingestion without committing raw CSV |

## Verification Log

| Date | Check | Result |
|---|---|---|
| 2026-05-13 | `python -m src.download_data` | Passed with Kaggle credentials |
| 2026-05-13 | `python -m src.preprocess` | Passed |
| 2026-05-13 | `python -m src.train` | Passed, model version 1 promoted |
| 2026-05-13 | `python -m src.evaluate` | Passed |
| 2026-05-13 | API `/health` with trained model | Passed |
| 2026-05-13 | API `/predict` with trained model | Passed |
| 2026-05-13 | `ruff check .` | Passed |
| 2026-05-13 | `pytest -q` | Passed, 11 tests |

## Next Session Checklist

Start here next time:

1. Check `git status --short`.
2. Check GitHub Actions for the latest push.
3. Validate Docker locally if Docker is available.
4. Choose the next feature from the Active Work Queue.
5. Update this file before committing.

