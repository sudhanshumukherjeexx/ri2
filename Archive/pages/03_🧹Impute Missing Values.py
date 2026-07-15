import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import time
cache_buster = int(time.time())



def initialize_page():
    """Initialize the page and session state"""
    if 'df_processed' not in st.session_state:
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("📤 Please upload a dataset to get started.")
            st.stop()
        st.session_state.df_processed = st.session_state.df
    
    return st.session_state.df_processed

def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/missing_h.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>🔄 Missing Value Imputation</h2>
            <p style='font-size: 1.1em;'>Choose from eight powerful methods to handle missing data and maintain data integrity. 
            Select the most appropriate strategy for your analysis needs.</p>
        </div>
        """, unsafe_allow_html=True)

def display_missing_values_info(df):
    """Display missing values information"""
    missing_df = df.to_pandas().isnull().sum()
    missing_df = missing_df[missing_df > 0].to_frame(name='Missing Values Count')
    
    if not missing_df.empty:
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 10px; border-radius: 5px; border: 1px solid #ffeeba;'>
            <h4 style='color: #856404;'>⚠️ Missing Values Detected</h4>
            <p>The following columns contain missing values:</p>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(missing_df, use_container_width=True)
    else:
        st.success("✅ No missing values detected in the dataset!")

IMPUTATION_METHODS = {
    "1. Drop Missing Values": {
        "description": "Remove rows containing missing values in the selected column",
        "function": lambda df, col, **kwargs: pl.from_pandas(df.to_pandas().dropna())
    },
    "2. Replace Missing Values with Specific Value": {
        "description": "Replace missing values with a user-specified value",
        "function": lambda df, col, value: df.with_columns(pl.col(col).fill_null(value))
    },
    "3. Impute Missing Data with Fill Forward Strategy": {
        "description": "Fill missing values using the last known value",
        "function": lambda df, col, **kwargs: df.with_columns(pl.col(col).forward_fill())
    },
    "4. Impute Missing Data with Backward Fill Strategy": {
        "description": "Fill missing values using the next known value",
        "function": lambda df, col, **kwargs: df.with_columns(pl.col(col).backward_fill())
    },
    "5. Impute Missing Data based on Distribution of Feature": {
        "description": "Fill missing values using random values from the column's distribution",
        "function": lambda df, col, **kwargs: distribution_imputation(df, col)
    },
    "6. Impute Missing Data with Mean": {
        "description": "Replace missing values with the column's mean value",
        "function": lambda df, col, **kwargs: df.with_columns(pl.col(col).fill_null(df.select(pl.mean(col))))
    },
    "7. Impute Missing Data with Median": {
        "description": "Replace missing values with the column's median value",
        "function": lambda df, col, **kwargs: df.with_columns(pl.col(col).fill_null(df.select(pl.median(col))))
    },
    "8. Impute Missing Data with Nearest Neighbours": {
        "description": "Fill missing values using nearest neighbor values",
        "function": lambda df, col, **kwargs: nearest_neighbor_imputation(df, col)
    },
    # "8. Impute Missing Data with Interpolation": {
    #     "description": "Fill missing values using interpolation",
    #     "function": lambda df, col, **kwargs: df.with_columns(pl.col(col).interpolate())
    # }
}

def distribution_imputation(df, column):
    """Impute missing values based on column distribution"""
    df_pandas = df.to_pandas()
    if df_pandas[column].dtype != object:
        mean = df_pandas[column].mean()
        std = df_pandas[column].std()
        random_values = np.random.normal(loc=mean, scale=std, size=df_pandas[column].isnull().sum())
        random_values = np.where(random_values < 0, 0, random_values)
        df_pandas[column] = df_pandas[column].fillna(
            pd.Series(random_values, index=df_pandas[column][df_pandas[column].isnull()].index)
        )
        return pl.from_pandas(df_pandas)
    return df

def nearest_neighbor_imputation(df, column):
    """Impute missing values using nearest neighbors"""
    df_pandas = df.to_pandas()
    if df_pandas[column].dtype != object:
        missing_inds = df_pandas[column].isnull()
        non_missing_inds = ~missing_inds
        non_missing_vals = df_pandas[column][non_missing_inds]
        closest_inds = np.abs(
            df_pandas[column][missing_inds].values - non_missing_vals.values.reshape(-1, 1)
        ).argmin(axis=0)
        df_pandas.loc[missing_inds, column] = non_missing_vals.iloc[closest_inds].values
        return pl.from_pandas(df_pandas)
    return df

def create_imputation_interface(df):
    """Create the imputation interface"""
    col1, col2 = st.columns(2)
    
    with col1:
        column = st.selectbox(
            "Select Column for Imputation",
            df.columns,
            help="Choose the column containing missing values that you want to impute"
        )
    
    with col2:
        method = st.selectbox(
            "Select Imputation Method",
            list(IMPUTATION_METHODS.keys()),
            help="Choose the method to handle missing values"
        )
    
    # Show description
    st.info(IMPUTATION_METHODS[method]["description"])
    
    # input for specific value replacement
    placeholder_value = None
    if method == "2. Replace Missing Values with Specific Value":
        placeholder_value = st.number_input(
            "Enter replacement value",
            help="Enter the value to replace missing values with"
        )
    
    return column, method, placeholder_value

def process_imputation(df, column, method, placeholder_value=None):
    """Process the imputation request"""
    try:
        df_copy = df.clone()
        imputation_function = IMPUTATION_METHODS[method]["function"]
        
        if method == "2. Replace Missing Values with Specific Value":
            if placeholder_value is None:
                st.error("Please enter a specific value for imputation.")
                return None
            df_copy = imputation_function(df_copy, column, value=placeholder_value)
        else:
            df_copy = imputation_function(df_copy, column)
        
        st.success(f"✅ Successfully applied {method.split('.')[1]} imputation!")
        return df_copy
    
    except Exception as e:
        st.error(f"❌ Error during imputation: {str(e)}")
        return None

def display_download_options(df):
    """Display download and view options for processed data"""
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👁️ View Processed Data", use_container_width=True):
            st.dataframe(df, use_container_width=True)
    
    with col2:
        csv_data = df.write_csv()
        st.download_button(
            label="⬇️ Download Processed Data",
            data=csv_data,
            file_name="processed_data.csv",
            use_container_width=True
        )


def main():
    # initialize
    df = initialize_page()
    

    create_header()
    
    # display current data
    st.markdown("### Current Dataset")
    st.dataframe(df, use_container_width=True)
    
    # display missing values information
    st.markdown("### Missing Values Summary")
    display_missing_values_info(df)
    
    # imputation interface
    st.markdown("### Imputation Settings")
    column, method, placeholder_value = create_imputation_interface(df)
    
    # process imputation
    if st.button("🔄 Apply Imputation", use_container_width=True):
        processed_df = process_imputation(df, column, method, placeholder_value)
        if processed_df is not None:
            st.session_state.df_processed = processed_df
            display_download_options(processed_df)
    
    # always show download options if processed data exists
    elif 'df_processed' in st.session_state:
        display_download_options(st.session_state.df_processed)

if __name__ == "__main__":
    main()