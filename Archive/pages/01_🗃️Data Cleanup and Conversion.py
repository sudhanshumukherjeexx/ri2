import streamlit as st
import pandas as pd
import polars as pl
import time
import dateutil.parser as dlp

# initialize
def initialize_page():
    """Initialize the page and check for dataset"""
    df = st.session_state.get('df')
    if df is None:
        st.info("📤 Please upload a dataset to get started.")
        st.stop()
    # session state, if not exists
    if 'df_dtype' not in st.session_state:
        st.session_state.df_dtype = df.clone()
    return df

# panel gif and info
def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/datatype_c.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>🔄 Data Cleanup and Conversion</h2>
            <p style='font-size: 1.1em;'>Transform your data with ease! Clean up duplicates and convert data types 
            to ensure your dataset is properly formatted for analysis. Select from various operations below to begin.</p>
        </div>
        """, unsafe_allow_html=True)

# data types
def display_data_type_info():
    """Display information about available data types"""
    with st.expander("ℹ️ Available Data Types"):
        st.markdown("""
        | Data Type | Description |
        |-----------|-------------|
        | `INT64` | 64-bit signed integer type |
        | `FLOAT64` | 64-bit floating point type |
        | `DATETIME` | Calendar date and time representation |
        | `BOOLEAN` | True/False values |
        | `STRING` | Text data type |
        """)

# duplicates
def handle_duplicates(df):
    """Handle duplicate rows in the dataset"""
    duplicate_count = df.is_duplicated().sum()
    
    if duplicate_count > 0:
        st.warning(
            f"🔍 Found {duplicate_count} duplicate rows in the dataset",
            icon="⚠️"
        )
        if st.button('🗑️ Remove Duplicates', use_container_width=True):
            try:
                st.session_state.df_dtype = df.unique()
                st.success("✅ Duplicates removed successfully!")
                return st.session_state.df_dtype
            except Exception as e:
                st.error(f"❌ Error removing duplicates: {str(e)}")
    else:
        st.success("✅ No duplicate rows found in the dataset", icon="✨")
    return df

# datetime

def try_parse_datetime(df, column):
    """
    Attempt to parse datetime in a column using dateutil.parser
    
    This provides more flexible datetime parsing than format strings
    and can handle ISO formats with timezones and fractional seconds.
    
    Args:
        df: polars DataFrame
        column: column name containing datetime strings
    
    Returns:
        DataFrame with the column converted to datetime
    
    Raises:
        ValueError: If datetime parsing fails
    """
    import dateutil.parser
    
    def parse_with_dateutil(date_str):
        try:
            return dateutil.parser.parse(date_str)
        except (ValueError, TypeError):
            return None
    
    try:
        return df.with_columns(
            pl.col(column).map_elements(parse_with_dateutil)
        )
    except Exception as e:
        raise ValueError(f"Could not parse datetime in column '{column}'. Error: {str(e)}")

def try_parse_datetime_(df, column):
    """Attempt to parse datetime using various formats"""
    common_formats = [
        # European/International date formats (DD-MM-YYYY)
        "%d-%m-%Y",                 # 04-09-2012
        "%d/%m/%Y",                 # 04/09/2012
        
        # US date formats with AM/PM
        "%m/%d/%Y %I:%M:%S %p",     # 02/24/2022 02:57:00 AM
        "%m/%d/%Y %I:%M %p",        # 02/24/2022 02:57 AM
        
        # ISO and common formats
        "%Y-%m-%d %H:%M:%S%.f %z",  # Using %.f instead of .%f
        "%Y-%m-%d %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S%.f",     # Using %.f instead of .%f
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    
    last_error = None
    for fmt in common_formats:
        try:
            return df.with_columns(pl.col(column).str.to_datetime(fmt))
        except Exception as e:
            last_error = e
            continue
    
    raise ValueError(f"Could not parse datetime in column '{column}'. Last error: {str(last_error)}")


def change_data_type(df, column, new_type):
    """Change the data type of a specified column"""
    try:
        # Copy dataframe 
        modified_df = df.clone()
        
        # current column type
        current_type = modified_df.schema[column]
        
        if new_type == "STRING":
            modified_df = modified_df.with_columns(pl.col(column).cast(pl.Utf8))
        
        elif new_type == "INT":
            # Check if already numeric type
            if current_type.is_numeric():
                modified_df = modified_df.with_columns(pl.col(column).cast(pl.Int64))
            else:
                # Only apply string operations if not already numeric
                modified_df = modified_df.with_columns(
                    pl.col(column).str.replace(r'[^0-9.-]', '').cast(pl.Int64)
                )
        
        elif new_type == "FLOAT":
            # Check if already numeric type
            if current_type.is_numeric():
                modified_df = modified_df.with_columns(pl.col(column).cast(pl.Float64))
            else:
                # Only apply string operations if not already numeric
                modified_df = modified_df.with_columns(
                    pl.col(column).str.replace(r'[^0-9.-]', '').cast(pl.Float64)
                )
        
        elif new_type == "DATETIME":
            modified_df = try_parse_datetime(modified_df, column)
        
        else:  # BOOLEAN
            if isinstance(current_type, pl.Utf8):
                modified_df = modified_df.with_columns(
                    pl.col(column).map_elements(
                        lambda x: str(x).lower() in ['true', '1', 'yes', 'y']
                    ).cast(pl.Boolean)
                )
            else:
                # For numeric types, treat 0 as False and non-zero as True
                modified_df = modified_df.with_columns(
                    pl.col(column).cast(pl.Boolean)
                )
        
        st.success(f"✅ Successfully converted '{column}' to {new_type}")
        return modified_df
    
    except Exception as e:
        st.error(f"❌ Error converting data type: {str(e)}")
        # Return original dataframe if conversion fails
        return df

def display_datatype_comparison(original_df, modified_df):
    """Display a comparison of original and modified data types"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Original Data Types")
        original_types = original_df.to_pandas().dtypes
        for col, dtype in original_types.items():
            st.code(f"{col}: {dtype}")
            
    with col2:
        st.markdown("#### Updated Data Types")
        modified_types = modified_df.to_pandas().dtypes
        for col, dtype in modified_types.items():
            st.code(f"{col}: {dtype}")



def main():
    df = initialize_page()
    create_header()
    display_data_type_info()
    
    # tabs
    tab1, tab2 = st.tabs(["🔍 Handle Duplicates", "🔄 Convert Data Types"])
    
    # Initialize df_dtype if not exists
    if 'df_dtype' not in st.session_state:
        st.session_state.df_dtype = df.clone()
    
    with tab1:
        st.session_state.df_dtype = handle_duplicates(st.session_state.df_dtype)
        st.dataframe(
            st.session_state.df_dtype.to_pandas(),
            use_container_width=True,
            height=400
        )
    
    with tab2:
        st.dataframe(
            st.session_state.df_dtype.to_pandas(),
            use_container_width=True,
            height=300
        )
        
        col1, col2 = st.columns(2)
        with col1:
            column = st.selectbox(
                "Select column",
                df.columns,
                key="column_select"
            )
        with col2:
            new_type = st.selectbox(
                "Select new data type",
                ["INT", "FLOAT", "DATETIME", "BOOLEAN", "STRING"],
                key="type_select"
            )
        
        if st.button('🔄 Convert Data Type', use_container_width=True):
            # Update the session state with the modified dataframe
            st.session_state.df_dtype = change_data_type(
                st.session_state.df_dtype,  # Use the current state
                column,
                new_type
            )
        
        display_datatype_comparison(df, st.session_state.df_dtype)
    
    # session state updated
    st.session_state.df = st.session_state.df_dtype.clone() 



if __name__ == "__main__":
    main()