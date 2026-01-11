from importlib.machinery import SourceFileLoader

import pandas as pd


feature_utils = SourceFileLoader("feature_utils", "src/feature_utils.py").load_module()


def test_escape_token_part():
    assert feature_utils._escape_token_part("a|b=c") == "a%7Cb%3Dc"


def test_to_feature_dict_base_and_cross():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "click": 0,
                "site_id": "s1",
                "app_id": "a1",
                "site_domain": "sd|1",
                "app_domain": "ad=1",
                "device_type": "d1",
                "device_conn_type": None,
            }
        ]
    )

    dicts = feature_utils.to_feature_dict(df, add_feature_cross=True)
    assert len(dicts) == 1
    tokens = dicts[0]

    assert tokens["site_id=s1"] == 1
    assert tokens["app_id=a1"] == 1
    assert tokens["site_domain=sd%7C1"] == 1
    assert tokens["app_domain=ad%3D1"] == 1
    assert tokens["device_type=d1"] == 1
    assert "device_conn_type" not in tokens

    assert tokens["cross:site_id=s1|app_id=a1"] == 1
    assert tokens["cross:site_domain=sd%7C1|app_domain=ad%3D1"] == 1
    assert "cross:device_type=d1|device_conn_type=None" not in tokens


def test_to_feature_dict_include_missing_tokens():
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
                "device_conn_type": None,
            }
        ]
    )

    dicts = feature_utils.to_feature_dict(df, add_feature_cross=False, skip_missing=False)
    tokens = dicts[0]
    assert tokens["device_conn_type=None"] == 1
