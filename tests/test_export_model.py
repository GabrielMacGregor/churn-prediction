from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import src.export_model as export_model

TEST_EXPORT_DIR = Path(__file__).parent / ".runtime" / "export_models"


class FakeModel:
    pass


def make_output_dir() -> Path:
    TEST_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_EXPORT_DIR / uuid4().hex


def test_export_model_artifact_saves_model_and_metadata(monkeypatch) -> None:
    output_dir = make_output_dir()
    calls = {}

    def fake_load_model(model_uri):
        calls["model_uri"] = model_uri
        return FakeModel()

    def fake_save_model(sk_model, path):
        calls["saved_model"] = sk_model
        calls["path"] = path
        output_dir.mkdir()
        (output_dir / "MLmodel").write_text("fake mlflow model", encoding="utf-8")

    monkeypatch.setattr(export_model.mlflow.sklearn, "load_model", fake_load_model)
    monkeypatch.setattr(export_model.mlflow.sklearn, "save_model", fake_save_model)

    metadata = export_model.export_model_artifact(
        model_uri="models:/telco-churn-xgb@production",
        output_dir=output_dir,
        tracking_uri="sqlite:///mlflow.db",
    )

    saved_metadata = json.loads((output_dir / "export_metadata.json").read_text())

    assert calls["model_uri"] == "models:/telco-churn-xgb@production"
    assert calls["path"] == str(output_dir)
    assert isinstance(calls["saved_model"], FakeModel)
    assert saved_metadata["model_uri"] == "models:/telco-churn-xgb@production"
    assert saved_metadata["tracking_uri"] == "sqlite:///mlflow.db"
    assert metadata["output_dir"] == str(output_dir)


def test_export_model_artifact_overwrites_existing_generated_directory(monkeypatch) -> None:
    output_dir = make_output_dir()
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(export_model.mlflow.sklearn, "load_model", lambda _uri: FakeModel())

    def fake_save_model(sk_model, path):
        assert isinstance(sk_model, FakeModel)
        assert path == str(output_dir)
        assert not (output_dir / "stale.txt").exists()
        output_dir.mkdir()
        (output_dir / "MLmodel").write_text("new", encoding="utf-8")

    monkeypatch.setattr(export_model.mlflow.sklearn, "save_model", fake_save_model)

    export_model.export_model_artifact(output_dir=output_dir)

    assert (output_dir / "MLmodel").read_text(encoding="utf-8") == "new"
