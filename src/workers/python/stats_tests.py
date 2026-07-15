import numpy as np
import pandas as pd
from scipy import stats

TEST_SPECS = {
    "ttest": {"label": "2-Sample T-Test", "kind": "group", "min_groups": 2, "max_groups": 2},
    "anova": {"label": "ANOVA", "kind": "group", "min_groups": 3, "max_groups": None},
    "mannwhitney": {"label": "Mann-Whitney U Test", "kind": "group", "min_groups": 2, "max_groups": 2},
    "kruskal": {"label": "Kruskal-Wallis Test", "kind": "group", "min_groups": 3, "max_groups": None},
    "chisquare": {"label": "Chi-Square Test", "kind": "two_columns"},
    "wilcoxon": {"label": "Wilcoxon Signed-Rank Test", "kind": "two_columns"},
}


@register("stats_test_specs")
def _stats_test_specs(args):
    return TEST_SPECS


def _run_group_test(df, test, group_col, value_col):
    spec = TEST_SPECS[test]
    groups = {str(k): v[value_col].dropna().to_numpy() for k, v in df.groupby(group_col)}
    group_names = [g for g in groups if groups[g].size > 0]
    arrays = [groups[g] for g in group_names]

    n_groups = len(group_names)
    if n_groups < spec["min_groups"] or (spec["max_groups"] and n_groups > spec["max_groups"]):
        expected = (
            f"exactly {spec['min_groups']}"
            if spec["max_groups"] == spec["min_groups"]
            else f"at least {spec['min_groups']}"
        )
        raise ValueError(
            f"{spec['label']} requires {expected} groups in '{group_col}', found {n_groups}."
        )

    used_welch = False
    if test == "ttest":
        # Levene's test checks the equal-variance assumption; if it's
        # violated (p < 0.05), fall back to Welch's t-test instead of
        # silently running the pooled-variance Student's t-test regardless.
        _, levene_p = stats.levene(arrays[0], arrays[1])
        used_welch = levene_p < 0.05
        stat, p = stats.ttest_ind(arrays[0], arrays[1], equal_var=not used_welch)
    elif test == "mannwhitney":
        stat, p = stats.mannwhitneyu(arrays[0], arrays[1])
    elif test == "anova":
        stat, p = stats.f_oneway(*arrays)
    elif test == "kruskal":
        stat, p = stats.kruskal(*arrays)
    else:
        raise ValueError(f"Unknown group test: {test}")

    group_summary = [
        {"group": g, "n": int(arr.size), "mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}
        for g, arr in zip(group_names, arrays)
    ]
    result = {"statistic": float(stat), "p_value": float(p), "groups": group_summary}
    if test == "ttest":
        result["used_welch"] = used_welch
    return result


def _run_chisquare(df, col_a, col_b):
    table = pd.crosstab(df[col_a], df[col_b])
    stat, p, dof, expected = stats.chi2_contingency(table)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "dof": int(dof),
        "contingency_table": {
            "index": [str(v) for v in table.index.tolist()],
            "columns": [str(v) for v in table.columns.tolist()],
            "values": table.values.tolist(),
        },
    }


def _run_wilcoxon(df, col_a, col_b):
    paired = df[[col_a, col_b]].dropna()
    if paired.empty:
        raise ValueError(f"No overlapping non-missing rows between '{col_a}' and '{col_b}'.")
    stat, p = stats.wilcoxon(paired[col_a], paired[col_b])
    return {"statistic": float(stat), "p_value": float(p), "n_pairs": int(len(paired))}


@register("stats_test_run")
def _stats_test_run(args):
    source_key = args["source_key"]
    test = args["test"]
    if test not in TEST_SPECS:
        raise ValueError(f"Unknown test: {test}")

    df = get_df(source_key)
    spec = TEST_SPECS[test]

    if spec["kind"] == "group":
        return _run_group_test(df, test, args["group_col"], args["value_col"])
    if test == "chisquare":
        return _run_chisquare(df, args["col_a"], args["col_b"])
    if test == "wilcoxon":
        return _run_wilcoxon(df, args["col_a"], args["col_b"])
    raise ValueError(f"Unhandled test kind for {test}")
