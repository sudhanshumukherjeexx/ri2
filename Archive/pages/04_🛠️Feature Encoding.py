import streamlit as st
import polars as pl
from utilities.select_data_types import SelectDataTypes
from utilities.encoding import *
import openai
import time
cache_buster = int(time.time())



def initialize_page():
    """Initialize page and check requirements"""
    if 'openai_api_key' not in st.session_state:
        st.error("⚠️ Please add your OpenAI API key on the Home Page to continue.")
        st.stop()
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("📤 Please upload a dataset to get started.")
        st.stop()
    
    return st.session_state.openai_api_key, st.session_state.df, st.session_state.get('df_processed')

def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/page_6.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>🔄 Feature Encoding</h2>
            <p style='font-size: 1.1em;'>Transform categorical data into numerical format for machine learning models. 
            Choose from multiple encoding techniques to optimize your data representation.</p>
        </div>
        """, unsafe_allow_html=True)

def display_encoding_info():
    """Display information about encoding techniques"""
    encoding_methods = {
        "Label Encoding": {
            "icon": "🏷️",
            "description": "Converts categorical values into numerical labels (0 to n-1)",
            "best_for": "Ordinal data where order matters",
            "example": "Red → 0, Blue → 1, Green → 2"
        },
        "One Hot Encoding": {
            "icon": "📊",
            "description": "Creates binary columns for each category",
            "best_for": "Nominal categorical data with no inherent order",
            "example": "Color → [is_red, is_blue, is_green]"
        },
        "Ordinal Encoding": {
            "icon": "📈",
            "description": "Assigns numbers based on ordered categories",
            "best_for": "Ordinal data with clear ranking",
            "example": "Low → 0, Medium → 1, High → 2"
        }
    }

    for method, info in encoding_methods.items():
        st.markdown(f"""
        <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #e1e4e8; margin: 5px;'>
            <h4>{info['icon']} {method}</h4>
            <p><strong>Description:</strong> {info['description']}</p>
            <p><strong>Best for:</strong> {info['best_for']}</p>
            <p><strong>Example:</strong> {info['example']}</p>
        </div>
        """, unsafe_allow_html=True)

def get_session_state_data(selected_state, df, df_processed):
    """Get data based on selected session state"""
    if selected_state == "Intial DataFrame":
        return df.to_pandas()
    elif selected_state == "DataFrame after Missing value Imputation":
        if df_processed is None:
            st.warning("⚠️ This option is only available after missing value imputation. Please impute missing values first.")
            st.stop()
        return df_processed.to_pandas()
    return None

def encode_and_display(df, encoding_type, selected_columns):
    """Encode data and display results"""
    encoding_functions = {
        "Label Encoding": PerformLabelEncoding,
        "One Hot Encoding": PerformOneHotEncoding,
        "Ordinal Encoding": PerformOrdinalEncoding
    }
    
    try:
        with st.spinner(f"Applying {encoding_type}..."):
            # Apply encoding
            encoded_df = encoding_functions[encoding_type](df, selected_columns)
            
            # Display results
            st.success(f"✅ {encoding_type} applied successfully!")
            
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Encoded Data", "📈 Changes Summary"])
            
            with tab1:
                st.dataframe(encoded_df.head(), use_container_width=True)
            
            with tab2:
                # Show changes in data structure
                st.markdown("#### Data Structure Changes")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Columns:**")
                    st.write(f"Total columns: {len(df.columns)}")
                    st.write(list(df.columns))
                with col2:
                    st.markdown("**New Columns:**")
                    st.write(f"Total columns: {len(encoded_df.columns)}")
                    st.write(list(encoded_df.columns))
            
            # Save to session state and enable download
            df_encoded = pl.from_pandas(encoded_df)
            st.session_state.df_encoded = df_encoded
            
            # Download button
            csv_data = df_encoded.write_csv()
            st.download_button(
                label="⬇️ Download Encoded Data",
                data=csv_data,
                file_name="encoded_data.csv",
                use_container_width=True
            )
            
            return df_encoded
            
    except Exception as e:
        st.error(f"❌ Error during encoding: {str(e)}")
        return None


def main():
    # initialize
    api_key, df, df_processed = initialize_page()
    openai.api_key = api_key
    
    create_header()
    
    # display current data
    st.markdown("### Current Dataset")
    st.dataframe(df.head(), use_container_width=True)
    
    # show encoding information
    with st.expander("ℹ️ About Encoding Methods"):
        display_encoding_info()
    
    # encoding interface
    st.markdown("### Encoding Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_state = st.selectbox(
            "📁 Select Data Source",
            ["Intial DataFrame", "DataFrame after Missing value Imputation"],
            help="Choose which version of your data to encode"
        )
    
    with col2:
        selected_encoding = st.selectbox(
            "🔄 Select Encoding Method",
            ["Label Encoding", "One Hot Encoding", "Ordinal Encoding"],
            help="Choose how to encode your categorical variables"
        )
    
    # data based on selected state
    current_df = get_session_state_data(selected_state, df, df_processed)
    
    if current_df is not None:
        # column selection
        selected_columns = st.multiselect(
            '📊 Select Columns to Encode',
            current_df.columns,
            help="Choose which columns you want to encode"
        )
        
        # encoding
        if selected_columns and st.button('🔄 Apply Encoding', use_container_width=True):
            start = time.time() #testing
            encode_and_display(current_df, selected_encoding, selected_columns)
            end = time.time()
            st.info(f"Time Taken: {end - start} seconds")


if __name__ == "__main__":
    main()