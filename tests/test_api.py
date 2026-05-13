from __future__ import annotations

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


class FakeModel:
    def predict_proba(self, _frame):
        return np.array([[0.2, 0.8]])


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


def test_predict_returns_503_when_model_is_not_loaded(monkeypatch) -> None:
    with TestClient(api_main.app) as client:
        monkeypatch.setattr(api_main, "model", None)
        monkeypatch.setattr(api_main, "load_model", lambda: None)

        response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 503
    assert "Model is not loaded" in response.json()["detail"]
