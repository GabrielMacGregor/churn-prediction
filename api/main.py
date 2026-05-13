from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Security
from fastapi.security import APIKeyHeader

from api.schema import (
    CustomerFeatures,
    HealthResponse,
    PredictionLogEntry,
    PredictionLogSummary,
    PredictionResponse,
)
from src.config import API_KEY, DEFAULT_LOG_DB_PATH, DEFAULT_MLFLOW_TRACKING_URI, DEFAULT_MODEL_URI

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_URI = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
LOG_DB_PATH = Path(os.getenv("LOG_DB_PATH", str(DEFAULT_LOG_DB_PATH)))
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

model: Any | None = None
model_metadata: dict[str, str | None] = {"uri": MODEL_URI, "version": None}


def require_api_key(api_key: str | None = Security(API_KEY_HEADER)) -> None:
    if not API_KEY:
        return

    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def init_db() -> None:
    LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(LOG_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                threshold REAL NOT NULL DEFAULT 0.5,
                prediction INTEGER NOT NULL,
                model_uri TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(prediction_logs)").fetchall()
        }
        if "threshold" not in columns:
            connection.execute(
                "ALTER TABLE prediction_logs ADD COLUMN threshold REAL NOT NULL DEFAULT 0.5"
            )


def load_model() -> None:
    global model, model_metadata

    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        model_metadata = {"uri": MODEL_URI, "version": _extract_model_version(MODEL_URI)}
        logger.info("Loaded model from %s", MODEL_URI)
    except Exception as exc:  # pragma: no cover - covered by health behavior in tests
        model = None
        model_metadata = {"uri": MODEL_URI, "version": None}
        logger.warning("Model could not be loaded from %s: %s", MODEL_URI, exc)


def _extract_model_version(model_uri: str) -> str | None:
    if "@" in model_uri:
        return model_uri.rsplit("@", 1)[-1]
    if "/Production" in model_uri:
        return "Production"
    return None


def log_prediction(
    payload: dict[str, Any],
    probability: float,
    prediction: int,
    threshold: float,
) -> None:
    created_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(LOG_DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO prediction_logs (
                created_at,
                payload,
                churn_probability,
                threshold,
                prediction,
                model_uri
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                json.dumps(payload, sort_keys=True),
                probability,
                threshold,
                prediction,
                model_metadata["uri"],
            ),
        )


def fetch_prediction_logs(
    limit: int = 50,
    offset: int = 0,
    prediction: int | None = None,
) -> list[PredictionLogEntry]:
    query = """
        SELECT id, created_at, payload, churn_probability, threshold, prediction, model_uri
        FROM prediction_logs
    """
    params: list[Any] = []

    if prediction is not None:
        query += " WHERE prediction = ?"
        params.append(prediction)

    query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with sqlite3.connect(LOG_DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    return [
        PredictionLogEntry(
            id=row["id"],
            created_at=row["created_at"],
            payload=json.loads(row["payload"]),
            churn_probability=row["churn_probability"],
            threshold=row["threshold"],
            prediction=row["prediction"],
            churn=bool(row["prediction"]),
            model_uri=row["model_uri"],
        )
        for row in rows
    ]


def summarize_prediction_logs() -> PredictionLogSummary:
    with sqlite3.connect(LOG_DB_PATH) as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                COALESCE(SUM(prediction), 0) AS positive_predictions,
                AVG(churn_probability) AS average_churn_probability,
                MAX(created_at) AS latest_prediction_at
            FROM prediction_logs
            """
        ).fetchone()

    total_predictions = int(row[0])
    positive_predictions = int(row[1])
    return PredictionLogSummary(
        total_predictions=total_predictions,
        positive_predictions=positive_predictions,
        churn_rate=positive_predictions / total_predictions if total_predictions else 0.0,
        average_churn_probability=row[2],
        latest_prediction_at=row[3],
    )


def predict_probability(features: CustomerFeatures) -> float:
    if model is None:
        load_model()

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Train/register a model before calling /predict.",
        )

    frame = features.to_dataframe()
    probabilities = model.predict_proba(frame)
    return float(np.asarray(probabilities)[0, 1])


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    version="0.1.0",
    description="Predicts customer churn risk for a telecom customer profile.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model is not None, model_uri=MODEL_URI)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: CustomerFeatures,
    threshold: float = Query(default=0.5, ge=0.0, le=1.0),
    _auth: None = Depends(require_api_key),
) -> PredictionResponse:
    probability = predict_probability(payload)
    prediction = int(probability >= threshold)
    log_prediction(payload.model_dump(), probability, prediction, threshold)

    return PredictionResponse(
        prediction=prediction,
        churn=bool(prediction),
        churn_probability=probability,
        threshold=threshold,
        model_uri=model_metadata["uri"] or MODEL_URI,
        model_version=model_metadata["version"],
    )


@app.get("/logs", response_model=list[PredictionLogEntry])
def prediction_logs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    prediction: int | None = Query(default=None, ge=0, le=1),
    _auth: None = Depends(require_api_key),
) -> list[PredictionLogEntry]:
    return fetch_prediction_logs(limit=limit, offset=offset, prediction=prediction)


@app.get("/logs/summary", response_model=PredictionLogSummary)
def prediction_log_summary(_auth: None = Depends(require_api_key)) -> PredictionLogSummary:
    return summarize_prediction_logs()
