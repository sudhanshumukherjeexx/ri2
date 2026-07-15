import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ENCODED_KEY = "df_encoded"

ENCODING_METHODS = ["label", "onehot", "ordinal"]


def _normalize(series):
    # Only string-cast non-numeric columns before encoding. Casting numeric
    # columns to string first would make the encoder sort "10" before "2"
    # lexicographically instead of preserving numeric order.
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype(str)


def _label_encode(df, columns):
    for col in columns:
        df[col] = LabelEncoder().fit_transform(_normalize(df[col]))
    return df


def _ordinal_encode(df, columns):
    normalized = pd.concat([_normalize(df[c]) for c in columns], axis=1)
    df[columns] = OrdinalEncoder().fit_transform(normalized)
    return df


def _onehot_encode(df, columns):
    return pd.get_dummies(df, columns=columns, dtype=float)


@register("encoding_apply")
def _encoding_apply(args):
    source_key = args["source_key"]
    columns = args["columns"]
    method = args["method"]

    if not columns:
        raise ValueError("Select at least one column to encode.")
    if method not in ENCODING_METHODS:
        raise ValueError(f"Unknown encoding method: {method}")

    source = get_df(source_key)
    df = source.copy()

    if method == "label":
        df = _label_encode(df, columns)
    elif method == "ordinal":
        df = _ordinal_encode(df, columns)
    else:
        df = _onehot_encode(df, columns)

    set_df(ENCODED_KEY, df)
    return {
        "preview": preview(ENCODED_KEY),
        "original_columns": list(source.columns),
        "encoded_columns": list(df.columns),
    }
