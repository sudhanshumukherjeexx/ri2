import numpy as np
from scipy import stats

MAX_SHAPIRO_N = 5000


def _numeric_columns(df):
    return [c for c in df.columns if str(df[c].dtype).lower().startswith(("int", "float"))]


@register("diagnostics_summary")
def _diagnostics_summary(args):
    source_key = args["source_key"]
    df = get_df(source_key)
    rows = []
    for col in _numeric_columns(df):
        series = df[col].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "column": col,
                "skewness": float(stats.skew(series)),
                "kurtosis": float(stats.kurtosis(series)),
            }
        )
    return {"columns": _numeric_columns(df), "rows": rows}


@register("diagnostics_column")
def _diagnostics_column(args):
    source_key = args["source_key"]
    column = args["column"]
    df = get_df(source_key)
    series = df[column].dropna().astype(float)

    if series.empty:
        raise ValueError(f"Column '{column}' has no numeric data to analyze.")

    values = series.to_numpy()

    # Q-Q plot data (sample quantiles vs a theoretical normal distribution).
    (osm, osr), (slope, intercept, r) = stats.probplot(values, dist="norm", fit=True)

    shapiro_sample = values if len(values) <= MAX_SHAPIRO_N else np.random.choice(
        values, MAX_SHAPIRO_N, replace=False
    )
    shapiro_stat, shapiro_p = stats.shapiro(shapiro_sample)

    mean, std = float(np.mean(values)), float(np.std(values, ddof=1))
    # kstest(values, "norm", args=(mean, std)) trips a positional-arg bug in
    # this build's compiled ndtr(); standardizing first and testing against
    # the standard normal sidesteps it (mathematically equivalent).
    standardized = (values - mean) / std if std > 0 else values - mean
    ks_stat, ks_p = stats.kstest(standardized, "norm")

    anderson = stats.anderson(values, dist="norm")

    return {
        "column": column,
        "n": int(len(values)),
        "mean": mean,
        "std": std,
        "skewness": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
        "histogram_values": values.tolist(),
        "qq": {
            "theoretical": osm.tolist(),
            "sample": osr.tolist(),
            "slope": float(slope),
            "intercept": float(intercept),
            "r": float(r),
        },
        "normality_tests": {
            "shapiro": {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)},
            "kolmogorov_smirnov": {"statistic": float(ks_stat), "p_value": float(ks_p)},
            "anderson_darling": {
                "statistic": float(anderson.statistic),
                "critical_values": anderson.critical_values.tolist(),
                "significance_levels": anderson.significance_level.tolist(),
            },
        },
    }


@register("diagnostics_outliers")
def _diagnostics_outliers(args):
    source_key = args["source_key"]
    column = args["column"]
    k = float(args.get("k", 1.5))

    df = get_df(source_key)
    series = df[column].dropna().astype(float)
    if series.empty:
        raise ValueError(f"Column '{column}' has no numeric data to analyze.")

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    outliers = series[(series < lower) | (series > upper)]

    return {
        "column": column,
        "k": k,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "outlier_count": int(outliers.shape[0]),
        "total_count": int(series.shape[0]),
        "outlier_values": outliers.head(50).tolist(),
    }
