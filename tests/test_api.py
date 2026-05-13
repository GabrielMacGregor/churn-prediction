from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

import api.main as api_main

SAMPLE_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 947.45,
}

TEST_DB_DIR = Path(__file__).parent / ".runtime"


class FakeModel:
    def predict_proba(self, _frame):
        return np.array([[0.2, 0.8]])


def prepare_test_db(monkeypatch):
    TEST_DB_DIR.mkdir(exist_ok=True)
    db_path = TEST_DB_DIR / f"{uuid4().hex}.db"
    monkeypatch.setattr(api_main, "LOG_DB_PATH", db_path)
    api_main.init_db()
    return db_path


def test_health_endpoint_returns_status() -> None:
    with TestClient(api_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_churn_probability(monkeypatch) -> None:
    captured_log = {}

    def fake_log_prediction(payload, probability, prediction, threshold):
        captured_log.update(
            {
                "payload": payload,
                "probability": probability,
                "prediction": prediction,
                "threshold": threshold,
            }
        )

    with TestClient(api_main.app) as client:
        monkeypatch.setattr(api_main, "model", FakeModel())
        monkeypatch.setattr(
            api_main,
            "model_metadata",
            {"uri": "models:/telco-churn-xgb@production", "version": "production"},
        )
        monkeypatch.setattr(api_main, "log_prediction", fake_log_prediction)

        response = client.post("/predict?threshold=0.9", json=SAMPLE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0
    assert body["churn"] is False
    assert body["churn_probability"] == 0.8
    assert body["threshold"] == 0.9
    assert captured_log["threshold"] == 0.9


def test_predict_requires_api_key_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(api_main, "API_KEY", "secret-key")

    with TestClient(api_main.app) as client:
        monkeypatch.setattr(api_main, "model", FakeModel())
        monkeypatch.setattr(api_main, "log_prediction", lambda *_args, **_kwargs: None)

        unauthorized = client.post("/predict", json=SAMPLE_PAYLOAD)
        authorized = client.post(
            "/predict",
            json=SAMPLE_PAYLOAD,
            headers={"X-API-Key": "secret-key"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_predict_returns_503_when_model_is_not_loaded(monkeypatch) -> None:
    with TestClient(api_main.app) as client:
        monkeypatch.setattr(api_main, "model", None)
        monkeypatch.setattr(api_main, "load_model", lambda: None)

        response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]


def test_logs_endpoint_returns_recent_predictions(monkeypatch) -> None:
    prepare_test_db(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "model_metadata",
        {"uri": "models:/test@production", "version": "1"},
    )

    api_main.log_prediction(SAMPLE_PAYLOAD, probability=0.8, prediction=1, threshold=0.5)
    api_main.log_prediction(SAMPLE_PAYLOAD, probability=0.2, prediction=0, threshold=0.7)

    with TestClient(api_main.app) as client:
        response = client.get("/logs?limit=10")
        filtered = client.get("/logs?prediction=1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["threshold"] == 0.7
    assert body[0]["churn"] is False
    assert body[1]["churn_probability"] == 0.8

    assert filtered.status_code == 200
    filtered_body = filtered.json()
    assert len(filtered_body) == 1
    assert filtered_body[0]["prediction"] == 1


def test_logs_summary_returns_aggregate_metrics(monkeypatch) -> None:
    prepare_test_db(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "model_metadata",
        {"uri": "models:/test@production", "version": "1"},
    )

    api_main.log_prediction(SAMPLE_PAYLOAD, probability=0.8, prediction=1, threshold=0.5)
    api_main.log_prediction(SAMPLE_PAYLOAD, probability=0.2, prediction=0, threshold=0.5)

    with TestClient(api_main.app) as client:
        response = client.get("/logs/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 2
    assert body["positive_predictions"] == 1
    assert body["churn_rate"] == 0.5
    assert body["average_churn_probability"] == 0.5
    assert body["latest_prediction_at"] is not None


def test_logs_require_api_key_when_configured(monkeypatch) -> None:
    prepare_test_db(monkeypatch)
    monkeypatch.setattr(api_main, "API_KEY", "secret-key")

    with TestClient(api_main.app) as client:
        unauthorized = client.get("/logs")
        authorized = client.get("/logs", headers={"X-API-Key": "secret-key"})

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_init_db_migrates_existing_log_table(monkeypatch) -> None:
    TEST_DB_DIR.mkdir(exist_ok=True)
    db_path = TEST_DB_DIR / f"legacy-{uuid4().hex}.db"
    monkeypatch.setattr(api_main, "LOG_DB_PATH", db_path)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                model_uri TEXT NOT NULL
            )
            """
        )

    api_main.init_db()

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(prediction_logs)")}

    assert "threshold" in columns
