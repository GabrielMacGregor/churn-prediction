from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "telco-churn")
REGISTERED_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "telco-churn-xgb")
DEFAULT_MODEL_URI = os.getenv("MODEL_URI", f"models:/{REGISTERED_MODEL_NAME}@production")

DEFAULT_MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}",
)

DEFAULT_LOG_DB_PATH = Path(
    os.getenv("LOG_DB_PATH", str(PROCESSED_DIR / "predictions.db"))
).resolve()
API_KEY = os.getenv("API_KEY", "")

KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "blastchar/telco-customer-churn")
KAGGLE_CONFIG_DIR = Path(os.getenv("KAGGLE_CONFIG_DIR", str(BASE_DIR / ".kaggle"))).resolve()
