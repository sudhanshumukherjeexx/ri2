@register("correlation")
def _correlation(args):
    df = get_df(args["key"])
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr()
    return {
        "columns": list(corr.columns),
        "matrix": corr.values.tolist(),
    }
