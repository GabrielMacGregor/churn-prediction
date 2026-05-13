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
| EDA | Done | Full exploratory notebook at `notebooks/eda.ipynb` |
| Tests | Done | API and preprocessing tests passing |
| Docker | Partial | Compose includes data, MLflow, training, export, and API; validation pending |
| Deployment | Next | Railway deployment not started |

## Performance Target

| Metric | Target | Current (v1) |
|---|---:|---:|
| Test ROC-AUC | >= 0.82 | **0.8418** achieved |
| Test F1 | n/a | 0.5859 |
| Test Precision | n/a | 0.6644 |
| Test Recall | n/a | 0.5241 |
| Test Accuracy | n/a | 0.8034 |

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
- [x] Add production model export CLI and Docker Compose export service.
- [x] Add `.dockerignore` so local secrets and training artifacts are not copied into images.
- [x] Add GitHub Actions CI for lint and tests.
- [x] Publish initial project to GitHub.
- [x] Build full EDA notebook with class imbalance, numeric distributions, categorical churn rates, tenure cohorts, and correlation analysis.

## Active Work Queue

### 1. Validate Docker Flow

Goal: prove the README Docker workflow works end to end.

- [ ] Run `docker compose --profile data run --rm download-data`.
- [ ] Run `docker compose up -d mlflow`.
- [ ] Run `docker compose --profile training run --rm train`.
- [ ] Run `docker compose --profile export run --rm export-model`.
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

### 3. Export Production Model Artifact

Goal: create a production-ready model directory that can be copied into a Docker image.

Model loading strategy: train and register with MLflow locally; for deployment, bundle
the trained model artifact directly into the Docker image at build time using
`mlflow.sklearn.save_model`. The API loads from a local path instead of the MLflow
registry, removing the runtime dependency on a live tracking server.

- [x] Add an export CLI: `python -m src.export_model`.
- [x] Load the current `models:/telco-churn-xgb@production` model from MLflow.
- [x] Save it to `model_artifact/` with `mlflow.sklearn.save_model`.
- [x] Keep `model_artifact/` ignored in Git because it is generated binary output.
- [x] Document the export command in README.

### 4. Prepare Railway Deployment

Goal: deploy API to Railway with a self-contained image.

- [x] Confirm `load_model()` accepts local MLflow model directories.
- [x] Ensure Docker image builds can include `model_artifact/` when it exists locally.
- [x] Document `MODEL_URI=/app/model_artifact` in Railway environment variables.
- [x] Do not require `MLFLOW_TRACKING_URI` in Railway runtime.
- [ ] Validate Railway deployment end to end.
- [x] Add Railway environment variable table to README.

### 5. Monitoring Polish

Goal: make monitoring more useful for a product team story.

- [ ] Add `GET /logs/summary?window_hours=24`.
- [ ] Add positive prediction rate over a recent time window.
- [ ] Add average churn probability by prediction label.
- [ ] Add README examples for authenticated `/logs` calls.

### 6. Explainability with SHAP

Goal: make the model interpretable and link it back to the `/predict` endpoint story.

- [ ] Compute permutation importance on the test set after training.
- [ ] Save permutation importance to `reports/permutation_importance.csv`.
- [ ] Compute SHAP values using `shap.TreeExplainer` where compatible with the XGBoost pipeline.
- [ ] Generate and save SHAP summary plot to `reports/shap_summary.png`.
- [ ] Generate and save SHAP waterfall plot for a representative high-risk prediction.
- [ ] Log explainability artifacts to the MLflow run.
- [ ] Document top SHAP drivers in README after the next training run.

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
| Data dependency | `kaggle` stays in `requirements.txt` for now | Keeps setup simple until runtime and development dependencies are split later |
| Notebook dependencies | `requirements-dev.txt` | Keeps notebook-only tooling separate from the core runtime install |
| Deployment model loading | Train with MLflow, deploy with bundled local artifact | Removes runtime dependency on a live MLflow server |
| Production model artifact | Generated locally, ignored by Git | `model_artifact/` is binary output produced from MLflow, not source code |

## Model Reload Policy

The API loads the model once at startup via the lifespan hook. It lazy-loads on the
first `/predict` call only when no model was found at startup. New production versions
registered in MLflow require an API restart to take effect. A `/reload` admin endpoint
is a known future improvement if live reloading becomes necessary.

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
| 2026-05-13 | `pytest -q` | Passed, 13 tests |
| 2026-05-13 | `python -m src.export_model` | Passed, wrote `model_artifact/` |
| 2026-05-13 | `python -m src.evaluate --model-uri model_artifact` | Passed, ROC-AUC 0.8418 |
| 2026-05-13 | API `load_model()` with `MODEL_URI=model_artifact` | Passed |

## Next Session Checklist

Start here next time:

1. Check `git status --short`.
2. Check GitHub Actions for the latest push.
3. Validate Docker locally if Docker is available.
4. Choose the next feature from the Active Work Queue.
5. Update this file before committing.
