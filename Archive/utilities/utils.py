def sample_dataframe_with_outliers(df, target_rows=1000000):
    if df.shape[0] > target_rows:
        df_sampled = df.sample(n=target_rows, seed=42)  # Ensure reproducibility
        return df_sampled
    else:
        return df
    



def sample_dataframe_with_outliers_for_ml(df, target_rows=200000):
    if df.shape[0] > target_rows:
        df_sampled = df.sample(n=target_rows, seed=42)  # Ensure reproducibility
        return df_sampled
    else:
        return df