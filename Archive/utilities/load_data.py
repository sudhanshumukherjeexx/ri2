import polars as pl
import pandas as pd
import time
import streamlit as st
from utilities.utils import *

# def LoadData(file, file_type):
#     start_time = time.time()
#     try:
#         # Load the data
#         if file_type == 'csv':
#             df = pl.read_csv(file,has_header=True ,ignore_errors=True)
#         elif file_type == 'xlsx':
#             df = pl.read_excel(file)
#         elif file_type == 'parquet':
#             df = pl.read_parquet(file)
#         else:
#             raise ValueError("Unsupported file format. Supported formats are 'csv', 'xlsx', and 'parquet'.")
        
#         df = sample_dataframe_with_outliers(df)
#         load_time = time.time() - start_time
#         return df, load_time
#     except FileNotFoundError:
#         st.info(f"The file '{file}' was not found. Please check the file path.")
#     except pl.exceptions.ComputeError as e:
#         st.info(f"An error occurred while processing the file: {e}")
#     except Exception as e:
#         st.info(f"An unexpected error occurred: {e}")

def LoadData(file, file_type):
    start_time = time.time()
    try:
        # Load the data
        if file_type == 'csv':
            # Explicitly set has_header=True to ensure column names are recognized
            df = pl.read_csv(file, has_header=True, ignore_errors=True)
            
            # Check if column names are single letters (A, B, C...)
            single_letter_cols = all(
                len(str(col)) == 1 and str(col).isalpha() 
                for col in df.columns
            )
            
            if single_letter_cols:
                st.warning("Warning: Column names appear to be single letters (A, B, C...), which may indicate a header issue.")
                # Try to infer header from first row of data
                try:
                    # Re-read with infer_schema_length set higher to better detect column types
                    df = pl.read_csv(file, has_header=True, infer_schema_length=10000, ignore_errors=True)
                    
                    # If still single letter columns, try reading the first row
                    if all(len(str(col)) == 1 and str(col).isalpha() for col in df.columns):
                        # Get first row to use as column names
                        first_row = df.row(0)
                        
                        # Check if first row values would make reasonable column names
                        if all(isinstance(val, (str, int, float)) for val in first_row):
                            # Convert first row to strings for column names
                            new_headers = [str(val) for val in first_row]
                            
                            # Re-read file, skip first row, and set new column names
                            df = pl.read_csv(file, has_header=True, skip_rows=1, ignore_errors=True)
                            df.columns = new_headers
                            
                            st.success("Successfully used the first row as column headers.")
                except Exception as header_error:
                    st.warning(f"Attempted to fix headers but encountered an error: {header_error}")
                
        elif file_type == 'xlsx':
            # For Excel files, specify the sheet_name and header row
            df = pl.read_excel(file, sheet_name=0, read_header=True)
            
        elif file_type == 'parquet':
            df = pl.read_parquet(file)
            
        else:
            raise ValueError("Unsupported file format. Supported formats are 'csv', 'xlsx', and 'parquet'.")
        
        # Make dataframe compatible with Arrow by ensuring proper types
        # Convert the Polars DataFrame to pandas for Streamlit compatibility
        pandas_df = df.to_pandas()
        
        # Fix Arrow compatibility issues by converting problematic object columns to strings
        for col in pandas_df.select_dtypes(include=['object']).columns:
            try:
                # First try to convert to more specific types based on content
                # Try numeric conversion
                pandas_df[col] = pd.to_numeric(pandas_df[col], errors='ignore')
                
                # If still object type, convert to string
                if pandas_df[col].dtype == 'object':
                    pandas_df[col] = pandas_df[col].astype(str)
            except Exception as e:
                # If conversion fails, force to string
                st.warning(f"Converting column '{col}' to string due to Arrow compatibility issues.")
                pandas_df[col] = pandas_df[col].astype(str)
        
        # Convert back to Polars
        try:
            df = pl.from_pandas(pandas_df)
        except Exception as e:
            st.warning(f"Error converting back to Polars: {e}. Using pandas DataFrame instead.")
            # Keep as pandas DataFrame if conversion back fails
            df = pandas_df
        
        # Apply the sample_dataframe_with_outliers function
        # Check if we're using a Polars or pandas DataFrame and handle accordingly
        if isinstance(df, pl.DataFrame):
            df = sample_dataframe_with_outliers(df)
        else:
            # Assuming sample_dataframe_with_outliers can work with pandas DataFrames
            # If not, you'll need to adapt this part
            df = sample_dataframe_with_outliers(df)
        
        load_time = time.time() - start_time
        
        return df, load_time
        
    except FileNotFoundError:
        st.info(f"The file '{file}' was not found. Please check the file path.")
        return None, 0
    except pl.exceptions.ComputeError as e:
        st.info(f"An error occurred while processing the file: {e}")
        return None, 0
    except Exception as e:
        st.info(f"An unexpected error occurred: {e}")
        # Print detailed error for debugging
        import traceback
        st.error(traceback.format_exc())
        return None, 0
