from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_MLFLOW_TRACKING_URI, DEFAULT_MODEL_URI, RAW_DATA_PATH, REPORTS_DIR
from src.preprocess import load_raw_data, split_features_target


def evaluate_model(
    raw_path: str | Path = RAW_DATA_PATH,
    model_uri: str = DEFAULT_MODEL_URI,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    test_size: float = 0.2,
) -> dict[str, float]:
    mlflow.set_tracking_uri(tracking_uri)

    raw_df = load_raw_data(raw_path)
    features, target = split_features_target(raw_df)
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=target,
    )

    model = mlflow.sklearn.load_model(model_uri)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, predictions)),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a registered churn model.")
    parser.add_argument("--raw-path", default=str(RAW_DATA_PATH))
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = evaluate_model(
        raw_path=args.raw_path,
        model_uri=args.model_uri,
        tracking_uri=args.tracking_uri,
        test_size=args.test_size,
    )
    print(json.dumps(report, indent=2))
