import numpy as np
import pandas as pd

CLEAN_KEY = "df_cleaned"


def _dtype_summary(df):
    return {c: str(t) for c, t in df.dtypes.items()}


def _duplicate_count(df):
    # keep=False marks every row that belongs to a duplicate group (matching
    # the original Polars is_duplicated() semantics), not just the repeats
    # after the first occurrence -- drop_duplicates() below still only
    # removes the repeats, this is purely the diagnostic count shown to the
    # user.
    return int(df.duplicated(keep=False).sum())


@register("cleanup_start")
def _cleanup_start(args):
    source_key = args["source_key"]
    source = get_df(source_key)
    df = source.copy()
    set_df(CLEAN_KEY, df)
    return {
        "preview": preview(CLEAN_KEY),
        "duplicate_count": _duplicate_count(df),
        "original_dtypes": _dtype_summary(source),
    }


@register("cleanup_remove_duplicates")
def _cleanup_remove_duplicates(args):
    df = get_df(CLEAN_KEY)
    df = df.drop_duplicates().reset_index(drop=True)
    set_df(CLEAN_KEY, df)
    return {
        "preview": preview(CLEAN_KEY),
        "duplicate_count": _duplicate_count(df),
    }


def _convert_column(series, new_type):
    if new_type == "STRING":
        # astype(str) turns NaN into the literal text "nan" -- mask original
        # missing positions back to None so they stay missing.
        return series.astype(str).mask(series.isna(), None)

    if new_type == "INT":
        # Truncate toward zero (22.9 -> 22, -22.9 -> -22), matching the
        # original Polars cast's permissive (non-strict) truncating
        # behavior -- NOT round-half-to-even, which would turn 22.9 into 23.
        if pd.api.types.is_numeric_dtype(series):
            numeric = series
        else:
            cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            numeric = pd.to_numeric(cleaned, errors="coerce")
        return np.trunc(numeric).astype("Int64")

    if new_type == "FLOAT":
        if pd.api.types.is_numeric_dtype(series):
            return series.astype("float64")
        cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        return pd.to_numeric(cleaned, errors="coerce")

    if new_type == "DATETIME":
        return pd.to_datetime(series, errors="coerce")

    if new_type == "BOOLEAN":
        if pd.api.types.is_numeric_dtype(series):
            # series != 0 on a NaN entry evaluates to True (NaN is "not
            # equal" to everything), so missing values must be masked back
            # to NA explicitly instead of being left as a stray True.
            result = (series != 0).astype("boolean")
            return result.mask(series.isna(), pd.NA)
        return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])

    raise ValueError(f"Unknown target type: {new_type}")


@register("cleanup_convert_dtype")
def _cleanup_convert_dtype(args):
    column = args["column"]
    new_type = args["new_type"]
    df = get_df(CLEAN_KEY)

    try:
        df = df.copy()
        df[column] = _convert_column(df[column], new_type)
        set_df(CLEAN_KEY, df)
        return {
            "preview": preview(CLEAN_KEY),
            "success": True,
            "message": f"Converted '{column}' to {new_type}.",
        }
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
        return {
            "preview": preview(CLEAN_KEY),
            "success": False,
            "message": f"Could not convert '{column}' to {new_type}: {e}",
        }
