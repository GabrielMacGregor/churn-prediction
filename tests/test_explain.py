from __future__ import annotations

import numpy as np

from src.explain import build_permutation_importance_frame


def test_build_permutation_importance_frame_sorts_by_importance() -> None:
    frame = build_permutation_importance_frame(
        feature_names=["low", "high", "middle"],
        importances_mean=np.array([0.01, 0.20, 0.05]),
        importances_std=np.array([0.001, 0.020, 0.005]),
    )

    assert frame["feature"].tolist() == ["high", "middle", "low"]
    assert frame.loc[0, "importance_mean"] == 0.20
    assert frame.loc[2, "importance_std"] == 0.001
