import pandas as pd
import numpy as np
import polars as pl



def handle_target_missing_values(target_series, is_regression=True):
    """
    Handle missing values in the target variable using the same strategy as the preprocessing pipeline.
    
    Parameters:
    -----------
    target_series : pd.Series or np.ndarray
        The target variable that may contain missing values
    is_regression : bool, default=True
        Whether this is a regression task (numeric target) or classification (categorical target)
        
    Returns:
    --------
    pd.Series : The target series with missing values handled
    bool : Whether the target had any missing values
    str : Info message about what was done
    """
    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer
    
    # Convert to pandas Series if it's a numpy array
    if isinstance(target_series, np.ndarray):
        target_series = pd.Series(target_series)
        
    # Check if there are any missing values
    missing_count = target_series.isna().sum()
    had_missing = missing_count > 0
    
    if not had_missing:
        return target_series, False, "No missing values found in the target variable."
    
    # Handle missing values based on variable type
    if is_regression:
        # For numeric targets (regression), use mean imputation
        imputer = SimpleImputer(strategy="mean")
        # Reshape for SimpleImputer which expects 2D array
        imputed_values = imputer.fit_transform(target_series.values.reshape(-1, 1))
        # Convert back to Series
        imputed_series = pd.Series(imputed_values.flatten(), index=target_series.index)
        message = f"Filled {missing_count} missing values in the target variable with the mean ({imputed_series.mean():.4f})."
    else:
        # For categorical targets (classification), use constant imputation with "missing"
        # First check if the data is already string type
        if pd.api.types.is_numeric_dtype(target_series):
            # If numeric, fill with the mode (most common value)
            imputer = SimpleImputer(strategy="most_frequent")
            imputed_values = imputer.fit_transform(target_series.values.reshape(-1, 1))
            imputed_series = pd.Series(imputed_values.flatten(), index=target_series.index)
            mode_value = target_series.mode()[0]
            message = f"Filled {missing_count} missing values in the numeric categorical target with the most frequent value ({mode_value})."
        else:
            # If string/categorical, fill with "missing"
            imputed_series = target_series.fillna("missing")
            message = f"Filled {missing_count} missing values in the categorical target with 'missing'."
    
    return imputed_series, had_missing, message


def check_and_handle_target(df, target_column, analysis_type="Regression Analysis"):
    """
    Check if the target variable has missing values and handle them appropriately.
    
    Parameters:
    -----------
    df : pd.DataFrame or pl.DataFrame
        The dataframe containing the target column
    target_column : str
        The name of the target column
    analysis_type : str, default="Regression Analysis"
        The type of analysis being performed, either "Regression Analysis" or "Classification Analysis"
        
    Returns:
    --------
    tuple : (updated_df, had_missing, message)
        - updated_df: DataFrame with handled target column
        - had_missing: Boolean indicating if there were missing values
        - message: Information message about what was done
    """
    
    # Convert to pandas if it's a polars DataFrame
    is_polars = isinstance(df, pl.DataFrame)
    if is_polars:
        # Store column order for later
        column_order = df.columns
        # Convert to pandas
        pandas_df = df.to_pandas()
    else:
        pandas_df = df.copy()
    
    # Determine if this is regression or classification
    is_regression = analysis_type == "Regression Analysis"
    
    # Extract the target
    target_series = pandas_df[target_column]
    
    # Handle missing values in the target
    updated_target, had_missing, message = handle_target_missing_values(
        target_series, 
        is_regression=is_regression
    )
    
    # Update the target in the dataframe
    pandas_df[target_column] = updated_target
    
    # Convert back to polars if original was polars
    if is_polars:
        # Convert back to polars using the same column order
        updated_df = pl.from_pandas(pandas_df)[column_order]
    else:
        updated_df = pandas_df
    
    return updated_df, had_missing, message