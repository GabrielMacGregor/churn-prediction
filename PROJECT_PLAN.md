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
| Monitoring | Done | `/logs` and windowed `/logs/summary` backed by SQLite |
| Explainability | Done | Permutation importance and SHAP artifacts generated and logged to MLflow |
| Security | Done | Optional `X-API-Key` auth through `API_KEY` |
| EDA | Done | Full exploratory notebook at `notebooks/eda.ipynb` |
| Tests | Done | API and preprocessing tests passing |
| Docker | Done | Compose validated end to end with data download, MLflow, training, export, and API smoke test |
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
- [x] Add `window_hours` filtering and per-label probability averages to `/logs/summary`.
- [x] Add optional API key authentication.
- [x] Add Dockerfile and Docker Compose services for data, MLflow, train, and API.
- [x] Add production model export CLI and Docker Compose export service.
- [x] Add `.dockerignore` so local secrets and training artifacts are not copied into images.
- [x] Add explainability CLI with permutation importance and SHAP artifact generation.
- [x] Add GitHub Actions CI for lint and tests.
- [x] Publish initial project to GitHub.
- [x] Build full EDA notebook with class imbalance, numeric distributions, categorical churn rates, tenure cohorts, and correlation analysis.

## Active Work Queue

### 1. Check GitHub CI

Goal: make sure the published repo passes CI on GitHub.

- [ ] Open the Actions tab.
- [ ] Confirm dependency install succeeds.
- [ ] Confirm `ruff check .` passes.
- [ ] Confirm `pytest -q` passes.
- [ ] Fix workflow paths if GitHub treats the repo differently than local.

### 2. Validate Auth In Docker

Goal: prove the optional API key protection works in the containerized flow.

- [x] Set `API_KEY` in `.env`.
- [x] Recreate the API container.
- [x] Verify protected endpoints return `401` without `X-API-Key`.
- [x] Verify protected endpoints return `200` with `X-API-Key`.

### 3. Prepare Railway Deployment

Goal: deploy API to Railway with a self-contained image.

- [x] Confirm `load_model()` accepts local MLflow model directories.
- [x] Ensure Docker image builds can include `model_artifact/` when it exists locally.
- [x] Document `MODEL_URI=/app/model_artifact` in Railway environment variables.
- [x] Do not require `MLFLOW_TRACKING_URI` in Railway runtime.
- [ ] Validate Railway deployment end to end.
- [x] Add Railway environment variable table to README.

### 4. Export Production Model Artifact

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

### 5. Monitoring Polish

Goal: make monitoring more useful for a product team story.

- [x] Add `GET /logs/summary?window_hours=24`.
- [x] Add positive prediction rate over a recent time window.
- [x] Add average churn probability by prediction label.
- [x] Add README examples for authenticated `/logs` calls.

### 6. Explainability with SHAP

Goal: make the model interpretable and link it back to the `/predict` endpoint story.

- [x] Compute permutation importance on the test set after training.
- [x] Save permutation importance to `reports/permutation_importance.csv`.
- [x] Add SHAP generation with `shap.TreeExplainer` for the fitted XGBoost pipeline.
- [x] Install/validate `shap` in the local environment.
- [x] Generate and save SHAP summary plot to `reports/shap_summary.png`.
- [x] Generate and save SHAP waterfall plot for a representative high-risk prediction.
- [x] Log generated explainability artifacts to the MLflow run.
- [x] Document top permutation drivers in README.
- [x] Document top SHAP drivers in README after the full SHAP run.

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
| Explainability artifacts | Generated locally, ignored by Git, logged to MLflow | Keeps reports reproducible without committing binary outputs |
| MLflow in Docker | Explicit `--allowed-hosts` for Compose networking and localhost | MLflow 3 rejects internal container host headers unless allowed explicitly |

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
| 2026-05-13 | `docker --version` | Blocked, Docker not installed/in PATH |
| 2026-05-13 | GitHub Actions status via connector/`gh` | Blocked, connector 404 and `gh` not installed |
| 2026-05-13 | `python -m src.explain --skip-shap` | Passed, logged permutation importance to MLflow |
| 2026-05-13 | `python -m src.explain` | Passed, logged permutation and SHAP artifacts to MLflow |
| 2026-05-14 | `docker compose build` | Passed, built `churnprediction-api` and `churnprediction-mlflow` |
| 2026-05-14 | `docker compose --profile data run --rm download-data` | Passed, downloaded `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| 2026-05-14 | `docker compose up -d mlflow` | Passed, service healthy on `localhost:5000` |
| 2026-05-14 | `docker compose --profile training run --rm train` | Passed, model version 1 promoted with test ROC-AUC `0.8397` |
| 2026-05-14 | `docker compose --profile export run --rm export-model` | Passed, wrote `model_artifact/` from `models:/telco-churn-xgb@production` |
| 2026-05-14 | `docker compose up -d api` | Passed, API container started on `localhost:8000` |
| 2026-05-14 | Docker API `GET /health` | Passed, `model_loaded=true` and `model_uri=/app/model_artifact` |
| 2026-05-14 | Docker API `POST /predict` | Passed, returned churn prediction `1` with probability `0.6404` |
| 2026-05-14 | Docker API `GET /logs` and `/logs/summary?window_hours=24` | Passed, recorded and aggregated the smoke-test prediction |
| 2026-05-14 | Docker API auth on `GET /logs/summary` | Passed, returned `401` without `X-API-Key` and `200` with the configured key |

## Next Session Checklist

1. Check `git status --short`.
2. Check GitHub Actions for the latest push.
3. Decide whether to start the Railway deployment track.
4. Update Git author config and commit the Docker validation docs if still pending.
5. Update this file before committing.
