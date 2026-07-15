import numpy as np
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

SCALED_KEY = "df_scaled"

SCALING_METHODS = {
    "minmax": "Min-Max Scaling (rescales to [0, 1])",
    "standard": "Standardization / Z-score (mean 0, std 1)",
    "robust": "Robust Scaling (uses IQR, ignores outliers)",
    "maxabs": "Max Abs Scaling (scales by max absolute value)",
    "quantile": "Quantile Transformer (uniform distribution)",
    "log": "Log Transform (log(1 + x))",
    "boxcox": "Power Transform - Box-Cox (positive values only)",
    "yeojohnson": "Power Transform - Yeo-Johnson (handles negatives)",
}


def _scale_column(values, method):
    col = values.to_numpy().reshape(-1, 1).astype(float)
    if method == "minmax":
        return MinMaxScaler().fit_transform(col).ravel()
    if method == "standard":
        return StandardScaler().fit_transform(col).ravel()
    if method == "robust":
        return RobustScaler().fit_transform(col).ravel()
    if method == "maxabs":
        return MaxAbsScaler().fit_transform(col).ravel()
    if method == "quantile":
        n = max(2, min(1000, col.shape[0]))
        return QuantileTransformer(n_quantiles=n).fit_transform(col).ravel()
    if method == "log":
        return np.log1p(col).ravel()
    if method == "boxcox":
        return PowerTransformer(method="box-cox").fit_transform(col).ravel()
    if method == "yeojohnson":
        return PowerTransformer(method="yeo-johnson").fit_transform(col).ravel()
    raise ValueError(f"Unknown scaling method: {method}")


@register("scaling_methods")
def _scaling_methods(args):
    return SCALING_METHODS


@register("scaling_apply")
def _scaling_apply(args):
    source_key = args["source_key"]
    columns = args["columns"]
    method = args["method"]

    if not columns:
        raise ValueError("Select at least one numeric column to scale.")
    if method not in SCALING_METHODS:
        raise ValueError(f"Unknown scaling method: {method}")

    source = get_df(source_key)
    df = source.copy()

    comparisons = []
    for col in columns:
        before = df[col].dropna()
        scaled = _scale_column(before, method)
        df.loc[before.index, col] = scaled
        comparisons.append(
            {
                "column": col,
                "before": before.to_numpy().tolist(),
                "after": scaled.tolist(),
            }
        )

    set_df(SCALED_KEY, df)
    return {
        "preview": preview(SCALED_KEY),
        "comparisons": comparisons,
    }
