import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction import FeatureHasher


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

model_loader = SourceFileLoader("model_loader", "src/serving/model_loader.py").load_module()
predictor = SourceFileLoader("predictor", "src/serving/predictor.py").load_module()
feature_utils = SourceFileLoader("feature_utils", "src/feature_utils.py").load_module()


def _build_artifact(tmp_path):
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "click": 0,
                "site_id": "s1",
                "app_id": "a1",
                "site_domain": "sd1",
                "app_domain": "ad1",
                "device_type": "d1",
                "device_conn_type": "dc1",
            },
            {
                "id": 2,
                "click": 1,
                "site_id": "s2",
                "app_id": "a2",
                "site_domain": "sd2",
                "app_domain": "ad2",
                "device_type": "d2",
                "device_conn_type": "dc2",
            },
        ]
    )
    X_raw = df.drop(columns=["click"])
    y = df["click"].astype(int)

    hasher = FeatureHasher(n_features=2**6, input_type="dict")
    tokens = feature_utils.to_feature_dict(X_raw, add_feature_cross=False)
    X_h = hasher.transform(tokens)

    model = DummyClassifier(strategy="prior")
    model.fit(X_h, y)

    artifact_path = tmp_path / "ctr_baseline_hashing.joblib"
    joblib.dump({"model": model, "hasher": hasher}, artifact_path)
    return artifact_path


def test_serving_loads_artifact_and_predicts(tmp_path, monkeypatch):
    artifact_path = _build_artifact(tmp_path)
    monkeypatch.setenv("MODEL_PATH", str(artifact_path))

    artifact = model_loader.load_artifact()
    predict_one = predictor.build_predictor(artifact)

    features = {
        "site_id": "s1",
        "app_id": "a1",
        "site_domain": "sd1",
        "app_domain": "ad1",
        "device_type": "d1",
        "device_conn_type": "dc1",
    }
    proba = predict_one(features)
    assert 0.0 <= proba <= 1.0
