from __future__ import annotations

import pandas as pd

from src.preprocess import build_preprocessor, clean_telco_data, split_features_target


def sample_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001-A", "0002-B", "0003-C"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "tenure": [0, 12, 24],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "MonthlyCharges": [29.85, 70.35, 89.1],
            "TotalCharges": [" ", "845.50", "2150.30"],
            "Churn": ["No", "Yes", "No"],
        }
    )


def test_clean_telco_data_converts_total_charges_and_target() -> None:
    cleaned = clean_telco_data(sample_training_frame())

    assert cleaned["TotalCharges"].tolist() == [0.0, 845.50, 2150.30]
    assert cleaned["Churn"].tolist() == [0, 1, 0]


def test_split_features_target_drops_identifier_and_target() -> None:
    features, target = split_features_target(sample_training_frame())

    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert target.tolist() == [0, 1, 0]


def test_build_preprocessor_handles_mixed_feature_types() -> None:
    features, _ = split_features_target(sample_training_frame())
    transformed = build_preprocessor().fit_transform(features)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] >= features.shape[1]

