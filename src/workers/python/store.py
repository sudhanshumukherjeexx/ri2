import json
import math

import numpy as np
import pandas as pd

# In-memory registry of named dataframes for this browser tab's session.
# Mirrors the current app's session_state keys: df, df_cleaned, df_processed,
# df_encoded, df_scaled -- except each transform writes its own key instead
# of ever overwriting an upstream one.
DATA = {}


def _sanitize(obj):
    """Recursively replace NaN/Infinity/pd.NA/numpy scalars with JSON-safe
    values *before* dumping. json.dumps' default= callback is NOT enough for
    this: it only fires for types the encoder doesn't natively recognize, and
    both Python float and its subclass np.float64 ARE natively recognized --
    so a NaN/Infinity float sails straight through as a bare `NaN`/`Infinity`
    token, which is not valid JSON and fails JSON.parse() in the browser."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def to_json(obj):
    return json.dumps(_sanitize(obj), default=str)


def get_df(key):
    if key not in DATA:
        raise KeyError(f"No dataset loaded for key '{key}'")
    return DATA[key]


def set_df(key, df):
    DATA[key] = df


def list_keys():
    return list(DATA.keys())


def preview(key, n=10):
    df = get_df(key)
    describe = json.loads(df.describe(include="all").to_json(default_handler=str))
    return {
        "key": key,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
        "head": json.loads(df.head(n).to_json(orient="records", date_format="iso")),
        "describe": describe,
    }
