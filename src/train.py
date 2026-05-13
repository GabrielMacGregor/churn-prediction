from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    DEFAULT_MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    RAW_DATA_PATH,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
)
from src.preprocess import build_preprocessor, load_raw_data, split_features_target

DEFAULT_XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
}


def build_model_pipeline(**xgb_params: object) -> Pipeline:
    params = DEFAULT_XGB_PARAMS | xgb_params
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", XGBClassifier(**params)),
        ]
    )


def _classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
    }


def _cross_validation_metrics(
    model: Pipeline,
    features,
    target,
    cv_splits: int,
) -> dict[str, float]:
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_validate(model, features, target, cv=cv, scoring=scoring, n_jobs=None)

    metrics: dict[str, float] = {}
    for name, values in scores.items():
        if not name.startswith("test_"):
            continue
        metric_name = name.replace("test_", "cv_")
        metrics[f"{metric_name}_mean"] = float(np.mean(values))
        metrics[f"{metric_name}_std"] = float(np.std(values))
    return metrics


def _find_registered_version_for_run(client: MlflowClient, model_name: str, run_id: str):
    versions = client.search_model_versions(f"name = '{model_name}'")
    matching_versions = [version for version in versions if version.run_id == run_id]
    if not matching_versions:
        return None
    return max(matching_versions, key=lambda version: int(version.version))


def _get_model_version_by_alias(client: MlflowClient, model_name: str, alias: str):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None


def _get_run_metric(client: MlflowClient, run_id: str | None, metric_name: str) -> float | None:
    if not run_id:
        return None

    run = client.get_run(run_id)
    metric = run.data.metrics.get(metric_name)
    return float(metric) if metric is not None else None


def _promote_candidate_if_better(
    model_name: str,
    candidate_run_id: str,
    candidate_roc_auc: float,
    alias: str = "production",
    metric_name: str = "test_roc_auc",
) -> dict[str, object]:
    client = MlflowClient()
    candidate_version = _find_registered_version_for_run(client, model_name, candidate_run_id)
    if candidate_version is None:
        return {
            "promoted": False,
            "reason": "candidate_version_not_found",
            "candidate_version": None,
            "current_version": None,
            "candidate_roc_auc": candidate_roc_auc,
            "current_roc_auc": None,
        }

    current_version = _get_model_version_by_alias(client, model_name, alias)
    current_roc_auc = _get_run_metric(
        client,
        current_version.run_id if current_version is not None else None,
        metric_name,
    )

    should_promote = current_roc_auc is None or candidate_roc_auc >= current_roc_auc
    if should_promote:
        client.set_registered_model_alias(model_name, alias, candidate_version.version)
        reason = (
            "promoted_first_candidate"
            if current_roc_auc is None
            else "promoted_metric_improved"
        )
    else:
        reason = "skipped_candidate_worse_than_production"

    client.set_model_version_tag(model_name, candidate_version.version, "promotion_status", reason)
    client.set_model_version_tag(
        model_name,
        candidate_version.version,
        "candidate_test_roc_auc",
        str(candidate_roc_auc),
    )
    if current_roc_auc is not None:
        client.set_model_version_tag(
            model_name,
            candidate_version.version,
            "previous_production_test_roc_auc",
            str(current_roc_auc),
        )

    return {
        "promoted": should_promote,
        "reason": reason,
        "candidate_version": str(candidate_version.version),
        "current_version": str(current_version.version) if current_version is not None else None,
        "candidate_roc_auc": candidate_roc_auc,
        "current_roc_auc": current_roc_auc,
    }


def train_model(
    raw_path: str | Path = RAW_DATA_PATH,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    experiment_name: str = EXPERIMENT_NAME,
    model_name: str = REGISTERED_MODEL_NAME,
    cv_splits: int = 5,
    test_size: float = 0.2,
) -> dict[str, object]:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    raw_df = load_raw_data(raw_path)
    features, target = split_features_target(raw_df)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=42,
        stratify=target,
    )

    model = build_model_pipeline()
    cv_metrics = _cross_validation_metrics(model, x_train, y_train, cv_splits=cv_splits)

    run_name = f"xgboost-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(DEFAULT_XGB_PARAMS)
        mlflow.log_param("cv_splits", cv_splits)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("features_count", features.shape[1])

        for metric_name, metric_value in cv_metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        model.fit(x_train, y_train)
        test_probabilities = model.predict_proba(x_test)[:, 1]
        test_metrics = _classification_metrics(y_test, test_probabilities)

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=x_test.head(3),
        )

        promotion = _promote_candidate_if_better(
            model_name=model_name,
            candidate_run_id=run.info.run_id,
            candidate_roc_auc=test_metrics["roc_auc"],
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / "training_metrics.json"
        report = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "model_uri": model_info.model_uri,
            "registered_model_name": model_name,
            "promotion": promotion,
            "cv": cv_metrics,
            "test": test_metrics,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register the churn model.")
    parser.add_argument("--raw-path", default=str(RAW_DATA_PATH))
    parser.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--model-name", default=REGISTERED_MODEL_NAME)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    training_report = train_model(
        raw_path=args.raw_path,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        cv_splits=args.cv_splits,
        test_size=args.test_size,
    )
    print(json.dumps(training_report, indent=2))
