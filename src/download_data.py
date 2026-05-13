from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import KAGGLE_CONFIG_DIR, KAGGLE_DATASET, RAW_DATA_PATH


def download_telco_churn_dataset(
    dataset: str = KAGGLE_DATASET,
    output_dir: str | Path = RAW_DATA_PATH.parent,
    expected_file: str = RAW_DATA_PATH.name,
    force: bool = False,
) -> Path:
    """Download the Telco churn dataset with the Kaggle API."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    expected_path = output_path / expected_file

    if expected_path.exists() and not force:
        return expected_path

    KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(KAGGLE_CONFIG_DIR))

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle package is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Set KAGGLE_USERNAME and KAGGLE_KEY in `.env`, "
            "or create `%USERPROFILE%\\.kaggle\\kaggle.json` on Windows."
        ) from exc

    api.dataset_download_files(dataset, path=str(output_path), unzip=True, quiet=False)

    if expected_path.exists():
        return expected_path

    csv_files = sorted(output_path.glob("*.csv"))
    if len(csv_files) == 1:
        csv_files[0].replace(expected_path)
        return expected_path

    raise FileNotFoundError(
        f"Dataset downloaded, but `{expected_file}` was not found in {output_path}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Telco churn dataset from Kaggle.")
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--output-dir", default=str(RAW_DATA_PATH.parent))
    parser.add_argument("--expected-file", default=RAW_DATA_PATH.name)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_path = download_telco_churn_dataset(
        dataset=args.dataset,
        output_dir=args.output_dir,
        expected_file=args.expected_file,
        force=args.force,
    )
    print(f"Dataset ready at {data_path}")
