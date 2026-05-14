# Next Session Handoff

Docker validation is now complete on this machine. The next session should build on
that success instead of redoing the environment setup.

## What Was Validated

- Docker Desktop installed and engine running.
- `docker compose build` completed successfully.
- `docker compose --profile data run --rm download-data` downloaded the Kaggle CSV.
- `docker compose up -d mlflow` started MLflow successfully on `http://localhost:5000`.
- `docker compose --profile training run --rm train` registered `telco-churn-xgb`
  and promoted version `1` to `production`.
- `docker compose --profile export run --rm export-model` wrote `model_artifact/`.
- `docker compose up -d api` started the API successfully on `http://localhost:8000`.
- `GET /health`, `POST /predict`, `GET /logs`, and
  `GET /logs/summary?window_hours=24` all passed.

## Important Docker Note

MLflow 3 rejected internal Docker requests until the server was started with explicit
allowed hosts. That is now captured in `docker-compose.yml`:

```text
--allowed-hosts "mlflow,mlflow:5000,localhost:*,127.0.0.1:*"
```

Keep that flag unless the Compose networking strategy changes.

## Current Status

- Docker flow: validated end to end.
- GitHub Actions: still needs a manual check on GitHub.
- Auth smoke test: validated with `401` without header and `200` with `X-API-Key`.
- Railway deployment: not started.

## Recommended Next Step

Check GitHub Actions, then decide whether to continue with Railway deployment.

## After Auth Validation

1. Check GitHub Actions:

```text
https://github.com/GabrielMacGregor/churn-prediction/actions
```

2. Decide whether to continue with Railway.
3. Update `PROJECT_PLAN.md` with the result.
4. If documentation changes are not committed yet, configure Git author identity and
   create the pending commit.

## Cleanup Commands

Stop containers when finished:

```powershell
docker compose down
```

If you want a fresh rerun later:

```powershell
Remove-Item -Recurse -Force model_artifact
Remove-Item -Recurse -Force mlruns
Remove-Item -Force mlflow.db
Remove-Item -Recurse -Force data\processed
```

Do not remove `data\raw` unless you want to download the dataset again.
