"""Create a tiny model artifact for serving in CI containers."""
import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import joblib
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction import FeatureHasher

    from src.feature_utils import to_feature_dict

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

    hasher = FeatureHasher(n_features=2**8, input_type="dict")
    tokens = to_feature_dict(X_raw, add_feature_cross=False)
    X_h = hasher.transform(tokens)

    model = DummyClassifier(strategy="prior")
    model.fit(X_h, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {"model": model, "hasher": hasher},
        "models/ctr_baseline_hashing.joblib",
    )
    print("Wrote models/ctr_baseline_hashing.joblib")


if __name__ == "__main__":
    main()
