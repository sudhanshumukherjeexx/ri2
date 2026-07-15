import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
from utilities.select_data_types import SelectDataTypes
from utilities.scaling import *
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
    
    return (st.session_state.openai_api_key, 
            st.session_state.df,
            st.session_state.get('df_processed'),
            st.session_state.get('df_encoded'))

def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/n_op.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>📊 Feature Scaling & Transformation</h2>
            <p style='font-size: 1.1em;'>Prepare your data for machine learning algorithms using various scaling 
            and transformation techniques. Choose the most appropriate method for your analysis needs.</p>
        </div>
        """, unsafe_allow_html=True)

SCALING_METHODS = {
    "Feature Scaling": {
        "Min-Max Scaling (Normalization)": {
            "description": "Rescales features to a range between 0 and 1",
            "function": MinMaxScaling,
            "icon": "📏",
            "best_for": "When you need bounded values"
        },
        "Standardization (Z-score Normalization)": {
            "description": "Transforms data to have a mean of 0 and standard deviation of 1",
            "function": StandardScaling,
            "icon": "📊",
            "best_for": "When you need normalized distribution"
        },
        "Robust Scaler": {
            "description": "Scales features using the interquartile range to handle outliers",
            "function": RobustScaling,
            "icon": "🛡️",
            "best_for": "When your data has outliers"
        },
        "Max AbsScaler": {
            "description": "Scales each feature by its maximum absolute value",
            "function": MaxAbsScaling,
            "icon": "📈",
            "best_for": "When dealing with sparse data"
        }
    },
    "Feature Transformation": {
        "Quantile Transformer Scaler": {
            "description": "Maps values to desired output distribution using quantile function",
            "function": QuantileTransformerScaling,
            "icon": "📉",
            "best_for": "When you need uniform distribution"
        },
        "Log Transformer": {
            "description": "Calculates log(1 + x) to handle skewed data",
            "function": LogTransformer,
            "icon": "📐",
            "best_for": "For right-skewed distributions"
        },
        "Power Transformer: Box-Cox": {
            "description": "Applies Box-Cox transformation for normal distribution",
            "function": PowerTransformerBoxCox,
            "icon": "🔄",
            "best_for": "For positive data only"
        },
        "Power Transformer: Yeo-Johnson": {
            "description": "Applies Yeo-Johnson transformation for normal distribution",
            "function": PowerTransformerYeoJohnson,
            "icon": "🔁",
            "best_for": "For both positive and negative data"
        }
    }
}

def display_methods_info():
    """Display information about scaling and transformation methods"""
    tab1, tab2 = st.tabs(["Feature Scaling", "Feature Transformation"])
    
    with tab1:
        for method, info in SCALING_METHODS["Feature Scaling"].items():
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #e1e4e8; margin: 5px;'>
                <h4>{info['icon']} {method}</h4>
                <p><strong>Description:</strong> {info['description']}</p>
                <p><strong>Best for:</strong> {info['best_for']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        for method, info in SCALING_METHODS["Feature Transformation"].items():
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #e1e4e8; margin: 5px;'>
                <h4>{info['icon']} {method}</h4>
                <p><strong>Description:</strong> {info['description']}</p>
                <p><strong>Best for:</strong> {info['best_for']}</p>
            </div>
            """, unsafe_allow_html=True)

def get_session_state_data(selected_state, df, df_processed, df_encoded):
    """Get data based on selected session state"""
    if selected_state == "Intial DataFrame":
        return df
    elif selected_state == "DataFrame after Missing value Imputation":
        if df_processed is None:
            st.warning("⚠️ This option is only available after missing value imputation.")
            st.stop()
        return df_processed
    else:
        if df_encoded is None:
            st.warning("⚠️ This option is only available after feature encoding.")
            st.stop()
        return df_encoded


def apply_scaling(df, method, selected_features):
    """Apply selected scaling method to selected features and display results"""
    try:
        # method info and function
        method_info = (SCALING_METHODS["Feature Scaling"].get(method) or 
                      SCALING_METHODS["Feature Transformation"].get(method))
        
        with st.spinner(f"Applying {method}..."):
            # scale only selected features
            df_selected = df[selected_features].copy()
            scaled_data = method_info["function"](df_selected)
            
            # final dataframe with scaled features
            df_final = df.copy()
            for col in selected_features:
                df_final[col] = scaled_data[col]
            
            # polars conversion for storage
            df_scaled = pl.from_pandas(df_final)
            st.session_state.df_scaled = df_scaled
            
            # tabs 
            tab1, tab2, tab3 = st.tabs(["📊 Scaled Data", "📈 Transformation Summary", "📉 Distribution Comparison"])
            
            with tab1:
                st.dataframe(df_final, use_container_width=True)
            
            with tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Before Scaling")
                    st.write(df[selected_features].describe())
                with col2:
                    st.markdown("#### After Scaling")
                    st.write(df_final[selected_features].describe())
            
            with tab3:
                if st.button("📊 Show Distribution Comparison", use_container_width=True):
                    st.session_state.show_comparison = True
                
                if st.session_state.get('show_comparison', False):
                    CompareHistograms(df[selected_features], df_final[selected_features])
            
            st.success(f"✅ {method} applied successfully to selected features!")
            
            # Add download button
            csv_data = df_scaled.write_csv()
            st.download_button(
                label="⬇️ Download Scaled Data",
                data=csv_data,
                file_name="scaled_data.csv",
                use_container_width=True
            )
            
            return df_scaled
            
    except Exception as e:
        st.error(f"❌ Error during scaling: {str(e)}")
        return None


def main():
    # Initialize page
    api_key, df, df_processed, df_encoded = initialize_page()
    openai.api_key = api_key
    
    # Create header
    create_header()
    
    # Show methods information
    with st.expander("ℹ️ About Scaling Methods"):
        display_methods_info()
    
    # Create scaling interface
    st.markdown("### Scaling Settings")
    
    # Initialize session state for selections
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = "Intial DataFrame"
    if 'selected_method' not in st.session_state:
        st.session_state.selected_method = list(SCALING_METHODS["Feature Scaling"].keys())[0]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_state = st.selectbox(
            "📁 Select Data Source",
            ["Intial DataFrame", "DataFrame after Missing value Imputation", "DataFrame after Feature Encoding"],
            key='data_source',
            help="Choose which version of your data to scale"
        )
    
    with col2:
        all_methods = list(SCALING_METHODS["Feature Scaling"].keys()) + list(SCALING_METHODS["Feature Transformation"].keys())
        selected_method = st.selectbox(
            "🔄 Select Scaling Method",
            all_methods,
            key='scaling_method',
            help="Choose how to scale your features"
        )
    
    # Get data based on selected state
    current_df = get_session_state_data(selected_state, df, df_processed, df_encoded)
    
    if current_df is not None:
        # Display current data
        st.markdown("### Current Data Preview")
        st.dataframe(current_df.head(), use_container_width=True)
        
        # Get numeric columns using SelectDataTypes
        df_numeric = SelectDataTypes(current_df)
        numeric_columns = df_numeric.columns.tolist()
        
        st.markdown("### Select Features to Scale")
        selected_features = st.multiselect(
            "Choose features to apply scaling",
            options=numeric_columns,
            default=numeric_columns,
            help="Select one or more numeric features to scale"
        )
        
        if not selected_features:
            st.warning("⚠️ Please select at least one feature to scale")
            st.stop()
        
        # Scale button
        if st.button('📐 Apply Scaling', key='apply_scaling', use_container_width=True):
            # Save selections to session state
            st.session_state.selected_state = selected_state
            st.session_state.selected_method = selected_method
            # Pass the pandas DataFrame from SelectDataTypes
            apply_scaling(df_numeric, selected_method, selected_features)
        
        # Show distribution comparison if scaled data exists
        if 'df_scaled' in st.session_state:
            st.markdown("### Distribution Comparison")
            if st.button("📊 Compare Original vs Scaled Distribution", use_container_width=True):
                CompareHistograms(
                    df_numeric[selected_features], 
                    st.session_state.df_scaled.to_pandas()[selected_features]
                )


if __name__ == "__main__":
    main()