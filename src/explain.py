from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.config import (
    BASE_DIR,
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_URI,
    RAW_DATA_PATH,
    REPORTS_DIR,
)
from src.preprocess import load_raw_data, split_features_target


def build_permutation_importance_frame(
    feature_names: list[str],
    importances_mean: np.ndarray,
    importances_std: np.ndarray,
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": importances_mean,
                "importance_std": importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _to_dense(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def _feature_names_from_model(model: Any, fallback_features: pd.DataFrame) -> list[str]:
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        return fallback_features.columns.tolist()
    return preprocessor.get_feature_names_out().tolist()


def _infer_run_id_from_model_uri(model_uri: str) -> str | None:
    if model_uri.startswith("runs:/"):
        return model_uri.removeprefix("runs:/").split("/", 1)[0]

    if not model_uri.startswith("models:/"):
        return None

    client = MlflowClient()
    model_ref = model_uri.removeprefix("models:/")

    try:
        if "@" in model_ref:
            model_name, alias = model_ref.rsplit("@", 1)
            return client.get_model_version_by_alias(model_name, alias).run_id

        parts = model_ref.split("/")
        if len(parts) >= 2 and parts[1].isdigit():
            return client.get_model_version(parts[0], parts[1]).run_id
    except MlflowException:
        return None

    return None


def _log_artifacts_to_run(run_id: str, artifact_paths: list[Path]) -> None:
    client = MlflowClient()
    for artifact_path in artifact_paths:
        client.log_artifact(run_id, str(artifact_path), artifact_path="reports")


def _save_shap_artifacts(
    model: Any,
    x_test: pd.DataFrame,
    output_dir: Path,
    max_rows: int,
) -> list[Path]:
    matplotlib_config_dir = BASE_DIR / ".tmp" / "matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP explainability requires shap and matplotlib. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    if not hasattr(model, "named_steps"):
        raise TypeError("SHAP export expects the trained sklearn Pipeline.")

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    x_sample = x_test.head(max_rows)
    transformed = _to_dense(preprocessor.transform(x_sample))
    transformed_feature_names = _feature_names_from_model(model, x_sample)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    summary_path = output_dir / "shap_summary.png"
    shap.summary_plot(
        shap_values,
        features=transformed,
        feature_names=transformed_feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()

    probabilities = model.predict_proba(x_sample)[:, 1]
    high_risk_position = int(np.argmax(probabilities))
    explanation = shap.Explanation(
        values=shap_values[high_risk_position],
        base_values=explainer.expected_value,
        data=transformed[high_risk_position],
        feature_names=transformed_feature_names,
    )
    waterfall_path = output_dir / "shap_waterfall.png"
    shap.plots.waterfall(explanation, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()

    return [summary_path, waterfall_path]


def explain_model(
    raw_path: str | Path = RAW_DATA_PATH,
    model_uri: str = DEFAULT_MODEL_URI,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    output_dir: str | Path = REPORTS_DIR,
    test_size: float = 0.2,
    permutation_repeats: int = 5,
    shap_max_rows: int = 500,
    include_shap: bool = True,
    log_to_mlflow: bool = True,
) -> dict[str, Any]:
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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    permutation = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=permutation_repeats,
        random_state=42,
        n_jobs=1,
    )
    permutation_frame = build_permutation_importance_frame(
        feature_names=x_test.columns.tolist(),
        importances_mean=permutation.importances_mean,
        importances_std=permutation.importances_std,
    )
    permutation_path = output_path / "permutation_importance.csv"
    permutation_frame.to_csv(permutation_path, index=False)

    artifact_paths = [permutation_path]
    if include_shap:
        artifact_paths.extend(
            _save_shap_artifacts(
                model=model,
                x_test=x_test,
                output_dir=output_path,
                max_rows=shap_max_rows,
            )
        )

    run_id = _infer_run_id_from_model_uri(model_uri)
    if log_to_mlflow and run_id is not None:
        _log_artifacts_to_run(run_id, artifact_paths)

    report = {
        "model_uri": model_uri,
        "tracking_uri": tracking_uri,
        "run_id": run_id,
        "permutation_importance_path": str(permutation_path),
        "shap_artifact_paths": [str(path) for path in artifact_paths if path.suffix == ".png"],
        "logged_to_mlflow": bool(log_to_mlflow and run_id is not None),
    }
    report_path = output_path / "explainability_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explainability artifacts.")
    parser.add_argument("--raw-path", default=str(RAW_DATA_PATH))
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--output-dir", default=str(REPORTS_DIR))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--shap-max-rows", type=int, default=500)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--no-log-to-mlflow", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explanation_report = explain_model(
        raw_path=args.raw_path,
        model_uri=args.model_uri,
        tracking_uri=args.tracking_uri,
        output_dir=args.output_dir,
        test_size=args.test_size,
        permutation_repeats=args.permutation_repeats,
        shap_max_rows=args.shap_max_rows,
        include_shap=not args.skip_shap,
        log_to_mlflow=not args.no_log_to_mlflow,
    )
    print(json.dumps(explanation_report, indent=2))
