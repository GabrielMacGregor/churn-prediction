# Next Session Handoff

Use this file when continuing the project on a machine with Docker Desktop available.
The current local blocker is environment-only: Docker is not installed/in PATH on the
previous machine, so the Docker and Railway paths still need end-to-end validation.

## Goal

Validate the Docker flow first, then decide whether to deploy to Railway.

The project is already pushed to:

```text
https://github.com/GabrielMacGregor/churn-prediction.git
```

## Starting From A New Machine

```powershell
git clone https://github.com/GabrielMacGregor/churn-prediction.git
cd churn-prediction
git status --short
```

Expected: clean worktree.

Install/check tools:

```powershell
docker --version
docker compose version
git --version
```

Docker Desktop should be running before the Compose commands.

## Environment Setup

Create a local `.env` file:

```powershell
copy .env.example .env
```

Fill these values in `.env`:

```text
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
KAGGLE_DATASET=blastchar/telco-customer-churn
API_KEY=
```

Keep `API_KEY` empty for the first local Docker smoke test. Add it later when testing
authenticated endpoints.

Never commit `.env`, Kaggle credentials, raw CSV files, `mlruns/`, `mlflow.db`, reports,
or `model_artifact/`.

## Docker Validation Flow

Build the images:

```powershell
docker compose build
```

Download the dataset through Kaggle:

```powershell
docker compose --profile data run --rm download-data
```

Confirm the CSV exists:

```powershell
dir data\raw
```

Start MLflow:

```powershell
docker compose up -d mlflow
docker compose ps
```

Open MLflow UI:

```text
http://localhost:5000
```

Train and register the model:

```powershell
docker compose --profile training run --rm train
```

Expected:

- A model is registered in MLflow as `telco-churn-xgb`.
- The `production` alias is assigned if the candidate is not worse than current production.
- `reports/training_metrics.json` is generated locally.

Export the production model into a local Docker-ready artifact:

```powershell
docker compose --profile export run --rm export-model
dir model_artifact
```

Expected files:

```text
MLmodel
model.pkl
conda.yaml
python_env.yaml
requirements.txt
export_metadata.json
```

Start the API:

```powershell
docker compose up -d api
docker compose ps
```

Check health:

```powershell
curl.exe http://localhost:8000/health
```

Expected:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_uri": "/app/model_artifact"
}
```

Run a prediction:

```powershell
curl.exe -X POST "http://localhost:8000/predict?threshold=0.5" `
  -H "Content-Type: application/json" `
  --data-binary "@examples/predict_request.json"
```

Check monitoring:

```powershell
curl.exe "http://localhost:8000/logs?limit=5"
curl.exe "http://localhost:8000/logs/summary?window_hours=24"
```

Open Swagger:

```text
http://localhost:8000/docs
```

## Explainability Validation

The explainability CLI was already validated locally outside Docker. On the Docker
machine, validate it again after training:

```powershell
docker compose run --rm `
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 `
  -e MODEL_URI=models:/telco-churn-xgb@production `
  api python -m src.explain
dir reports
```

Expected generated files:

```text
permutation_importance.csv
shap_summary.png
shap_waterfall.png
explainability_report.json
```

If this command is slow, that is normal. SHAP loads XGBoost and renders plots.

## Auth Smoke Test

After the unauthenticated smoke test passes, set an `API_KEY` in `.env`, then restart
the API:

```powershell
docker compose up -d --force-recreate api
```

Without the header, protected endpoints should return `401`:

```powershell
curl.exe "http://localhost:8000/logs/summary"
```

With the header, they should pass:

```powershell
curl.exe "http://localhost:8000/logs/summary" -H "X-API-Key: your_api_key"
```

## Troubleshooting

Check service status:

```powershell
docker compose ps
```

Read logs:

```powershell
docker compose logs mlflow --tail=100
docker compose logs train --tail=100
docker compose logs export-model --tail=100
docker compose logs api --tail=100
```

Common issues:

- Kaggle download fails: check `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env`.
- API health says `model_loaded=false`: run the train step, then the export step, then recreate the API.
- API loads old model: rerun export and recreate the API container.
- Port conflict: stop the conflicting local service or change ports in `docker-compose.yml`.

Clean up containers when done:

```powershell
docker compose down
```

Clean generated local artifacts only if you want a fresh run:

```powershell
Remove-Item -Recurse -Force model_artifact
Remove-Item -Recurse -Force mlruns
Remove-Item -Force mlflow.db
Remove-Item -Recurse -Force data\processed
```

Do not remove `data\raw` unless you want to download the dataset again.

## GitHub Actions Check

Open:

```text
https://github.com/GabrielMacGregor/churn-prediction/actions
```

Confirm the latest workflow on `main` passes:

- dependency install
- `ruff check .`
- `pytest -q`

If CI fails because of dependency resolution, compare the failing Python version with
the local Docker image. The project is designed around Python 3.11 in Docker and CI.

## Railway Decision

Only start Railway after the Docker flow passes locally.

Preferred production strategy:

- train and track with MLflow locally
- export the production model to `model_artifact/`
- build a self-contained image with that artifact included
- set `MODEL_URI=/app/model_artifact`
- do not run MLflow in Railway runtime

Railway variables:

```text
MODEL_URI=/app/model_artifact
LOG_DB_PATH=/data/predictions.db
API_KEY=secret-value
```

Before deploying, decide whether Railway should build from GitHub directly or from a
locally built/pushed Docker image. GitHub builds will not include ignored/generated
`model_artifact/` unless the artifact is produced during the build or provided through
another release process.

## Update The Plan

After Docker validation, update `PROJECT_PLAN.md`:

- mark Docker validation items as complete
- add the Docker verification commands/results to the verification log
- mark GitHub CI status based on the Actions tab
- record any Railway deployment decision

Then commit:

```powershell
git status --short
git add -A
git commit -m "docs: record docker validation"
git push
```
