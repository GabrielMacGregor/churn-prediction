from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()])


class PredictionResponse(BaseModel):
    prediction: int
    churn: bool
    churn_probability: float
    threshold: float
    model_uri: str
    model_version: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_uri: str


class PredictionLogEntry(BaseModel):
    id: int
    created_at: datetime
    payload: dict[str, Any]
    churn_probability: float
    threshold: float
    prediction: int
    churn: bool
    model_uri: str


class PredictionLogSummary(BaseModel):
    total_predictions: int
    positive_predictions: int
    churn_rate: float
    average_churn_probability: float | None
    latest_prediction_at: datetime | None
