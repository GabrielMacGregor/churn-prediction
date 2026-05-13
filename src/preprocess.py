from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import ID_COLUMN, PROCESSED_DIR, RAW_DATA_PATH, TARGET_COLUMN


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the Telco Customer Churn CSV."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {data_path}. "
            "Download the Kaggle Telco Customer Churn CSV and place it there."
        )
    return pd.read_csv(data_path)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and encode the target column when present."""
    cleaned = df.copy()

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = cleaned["TotalCharges"].replace(" ", pd.NA)
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")

        if "tenure" in cleaned.columns:
            new_customers = cleaned["TotalCharges"].isna() & (cleaned["tenure"].fillna(0) == 0)
            cleaned.loc[new_customers, "TotalCharges"] = 0.0

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = encode_target(cleaned[TARGET_COLUMN])

    return cleaned


def encode_target(series: pd.Series) -> pd.Series:
    """Encode churn labels as 1 for churn and 0 for retained customers."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    encoded = normalized.map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})

    if encoded.isna().any():
        invalid_values = sorted(series[encoded.isna()].dropna().unique().tolist())
        raise ValueError(f"Unexpected churn labels: {invalid_values}")

    return encoded.astype(int)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and binary target."""
    cleaned = clean_telco_data(df)

    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in input data.")

    drop_columns = [TARGET_COLUMN]
    if ID_COLUMN in cleaned.columns:
        drop_columns.append(ID_COLUMN)

    features = cleaned.drop(columns=drop_columns)
    target = cleaned[TARGET_COLUMN]
    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Create the sklearn preprocessing graph used by train and inference."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
    )


def create_train_test_split(
    raw_path: str | Path = RAW_DATA_PATH,
    output_dir: str | Path = PROCESSED_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    """Persist deterministic train/test CSVs for inspection.

    Training rebuilds this same split from raw data instead of depending on generated files.
    """
    df = load_raw_data(raw_path)
    features, target = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train.csv"
    test_path = output_path / "test.csv"

    pd.concat([x_train, y_train.rename(TARGET_COLUMN)], axis=1).to_csv(train_path, index=False)
    pd.concat([x_test, y_test.rename(TARGET_COLUMN)], axis=1).to_csv(test_path, index=False)

    return {"train": train_path, "test": test_path}


if __name__ == "__main__":
    paths = create_train_test_split()
    for split_name, split_path in paths.items():
        print(f"{split_name}: {split_path}")
