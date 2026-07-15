import numpy as np
import pandas as pd

PROCESSED_KEY = "df_processed"

IMPUTATION_METHODS = {
    "drop": "Remove rows where the selected column is missing",
    "specific_value": "Replace missing values with a user-specified value",
    "ffill": "Fill missing values using the last known value (forward fill)",
    "bfill": "Fill missing values using the next known value (backward fill)",
    "distribution": "Fill missing values by sampling the column's own distribution",
    "mean": "Replace missing values with the column's mean",
    "median": "Replace missing values with the column's median",
    "nearest_neighbor": "Fill missing values with the closest existing value",
}


@register("impute_methods")
def _impute_methods(args):
    return IMPUTATION_METHODS


@register("impute_start")
def _impute_start(args):
    source_key = args["source_key"]
    df = get_df(source_key).copy()
    set_df(PROCESSED_KEY, df)
    return preview(PROCESSED_KEY)


def _distribution_impute(series):
    missing = series.isna()
    n_missing = int(missing.sum())
    if n_missing == 0 or not pd.api.types.is_numeric_dtype(series):
        return series
    mean, std = series.mean(), series.std()
    std = 0.0 if pd.isna(std) else std
    sampled = np.random.normal(loc=mean, scale=std, size=n_missing)
    sampled = np.clip(sampled, 0, None)
    result = series.copy()
    result.loc[missing] = sampled
    return result


def _nearest_neighbor_impute(series):
    missing = series.isna()
    if not missing.any() or not pd.api.types.is_numeric_dtype(series):
        return series
    known = series[~missing]
    if known.empty:
        return series
    result = series.copy()
    missing_vals = series[missing].to_numpy().reshape(-1, 1)
    known_vals = known.to_numpy()
    closest_idx = np.abs(missing_vals - known_vals).argmin(axis=1)
    result.loc[missing] = known.iloc[closest_idx].to_numpy()
    return result


@register("impute_apply")
def _impute_apply(args):
    column = args["column"]
    method = args["method"]
    value = args.get("value")

    df = get_df(PROCESSED_KEY)

    if method not in IMPUTATION_METHODS:
        raise ValueError(f"Unknown imputation method: {method}")

    try:
        df = df.copy()
        if method == "drop":
            df = df.dropna(subset=[column]).reset_index(drop=True)
        elif method == "specific_value":
            if value is None:
                raise ValueError("A replacement value is required for this method.")
            df[column] = df[column].fillna(value)
        elif method == "ffill":
            df[column] = df[column].ffill()
        elif method == "bfill":
            df[column] = df[column].bfill()
        elif method == "distribution":
            df[column] = _distribution_impute(df[column])
        elif method == "mean":
            df[column] = df[column].fillna(df[column].mean())
        elif method == "median":
            df[column] = df[column].fillna(df[column].median())
        elif method == "nearest_neighbor":
            df[column] = _nearest_neighbor_impute(df[column])

        set_df(PROCESSED_KEY, df)
        return {
            "preview": preview(PROCESSED_KEY),
            "success": True,
            "message": f"Applied '{IMPUTATION_METHODS[method]}' to '{column}'.",
        }
    except Exception as e:  # noqa: BLE001 - surfaced to the UI, not swallowed
        return {
            "preview": preview(PROCESSED_KEY),
            "success": False,
            "message": f"Could not impute '{column}': {e}",
        }
