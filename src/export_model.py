from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn

from src.config import (
    BASE_DIR,
    DEFAULT_MLFLOW_TRACKING_URI,
    DEFAULT_MODEL_URI,
    MODEL_ARTIFACT_DIR,
)


def _remove_existing_artifact(output_dir: Path) -> None:
    resolved_output = output_dir.resolve()
    protected_paths = {BASE_DIR.resolve(), BASE_DIR.parent.resolve(), Path.cwd().resolve()}

    if resolved_output in protected_paths:
        raise ValueError(f"Refusing to overwrite protected directory: {resolved_output}")

    if not output_dir.exists():
        return

    if not output_dir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {resolved_output}")

    shutil.rmtree(output_dir)


def export_model_artifact(
    model_uri: str = DEFAULT_MODEL_URI,
    output_dir: str | Path = MODEL_ARTIFACT_DIR,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    overwrite: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    mlflow.set_tracking_uri(tracking_uri)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_path}")
        _remove_existing_artifact(output_path)

    model = mlflow.sklearn.load_model(model_uri)
    mlflow.sklearn.save_model(sk_model=model, path=str(output_path))

    metadata = {
        "model_uri": model_uri,
        "tracking_uri": tracking_uri,
        "output_dir": str(output_path),
        "exported_at": datetime.now(UTC).isoformat(),
    }
    (output_path / "export_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the production MLflow model to a local Docker-ready artifact.",
    )
    parser.add_argument("--model-uri", default=DEFAULT_MODEL_URI)
    parser.add_argument("--output-dir", default=str(MODEL_ARTIFACT_DIR))
    parser.add_argument("--tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail if the output directory already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_report = export_model_artifact(
        model_uri=args.model_uri,
        output_dir=args.output_dir,
        tracking_uri=args.tracking_uri,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(export_report, indent=2))
