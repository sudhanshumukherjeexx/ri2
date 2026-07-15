import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from utilities.select_data_types import SelectDataTypes
from utilities.image_explanation import *
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os
import openai
import gc
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats
import gc
import io
import base64
try:
    import seaborn as sns
except AttributeError:
    # if issue with seaborn and cm.register_cmap, the workaround
    import matplotlib as mpl
    if not hasattr(mpl.cm, 'register_cmap'):
        # Add a dummy function to prevent the error
        mpl.cm.register_cmap = lambda *args, **kwargs: None
    import seaborn as sns


import time
cache_buster = int(time.time())


def initialize_page():
    """Initialize page requirements"""
    os.makedirs("plot_images", exist_ok=True)
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("📤 Please upload a dataset to get started.")
        st.stop()
        
    return (st.session_state.get('openai_api_key'),
            st.session_state.df,
            st.session_state.get('df_processed'),
            st.session_state.get('df_encoded'),
            st.session_state.get('df_scaled'))

def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/stats_page.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>📊 Statistical Data Exploration</h2>
            <p style='font-size: 1.1em;'>Explore <strong>skewness</strong> and <strong>kurtosis</strong> for distribution insights, 
            assess <strong>normality</strong> with Q-Q plots, and detect <strong>outliers</strong> for data cleansing.</p>
        </div>
        """, unsafe_allow_html=True)

def get_selected_data(selected_state, df, df_processed, df_encoded, df_scaled):
    """Get the appropriate dataset based on selection"""
    if selected_state == "Intial DataFrame":
        return df
    elif selected_state == "DataFrame after Missing value Imputation":
        if df_processed is None:
            st.warning("⚠️ Please impute missing values first.")
            st.stop()
        return df_processed
    elif selected_state == "DataFrame after Feature Encoding":
        if df_encoded is None:
            st.warning("⚠️ Please encode features first.")
            st.stop()
        return df_encoded
    else:
        if df_scaled is None:
            st.warning("⚠️ Please scale features first.")
            st.stop()
        return df_scaled

def find_outliers(dataframe, selected_columns=None, k=1.5):
    """
    Find outliers in selected numeric columns
    Args:
        dataframe: Input Polars dataframe
        selected_columns: List of column names to analyze. If None, analyzes all numeric columns.
        k: IQR multiplier for outlier detection (default=1.5)
    """
    # Get numeric columns
    numeric_cols = [col_name for col_name, dtype in zip(dataframe.columns, dataframe.dtypes) 
                   if isinstance(dtype, (pl.Float32, pl.Float64, pl.Int32, pl.Int64))]
    
    # Validate selected columns
    if selected_columns is None:
        columns_to_analyze = numeric_cols
    else:
        columns_to_analyze = [col for col in selected_columns if col in numeric_cols]
        if not columns_to_analyze:
            st.warning("None of the selected columns are numeric.")
            return
    
    columns_with_outliers = []
    
    for column in columns_to_analyze:
        try:
            # Get column data
            values = dataframe[column].drop_nulls().to_numpy()
            
            if len(values) < 4:  # Need at least 4 points for meaningful quartiles
                st.warning(f"Not enough data points in {column} for outlier detection")
                continue
            
            # Calculate quartiles and IQR
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            
            # Find outliers
            outliers = values[(values < lower_bound) | (values > upper_bound)]
            
            # Create container for each column's analysis
            with st.container():
                st.info(f"### Outlier Detection in Column:{column}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Summary statistics
                    st.markdown("#### Summary")
                    col3, col4, col5 = st.columns(3)
                    with col3:
                        st.metric("Total Points", len(values))
                    with col4:
                        st.metric("Outliers Found", len(outliers))
                    with col5:
                        st.metric("Outlier Percentage", f"{(len(outliers)/len(values))*100:.2f}%")
                
                with col2:
                    # Boundary information
                    st.markdown(f"""
                                #### Boundaries
                                
                                1. Lower Bound: {lower_bound:.2f}
                                
                                2. Upper Bound: {upper_bound:.2f}
                                
                                3. IQR: {iqr:.2f}
                                """)
                    
                
                # outlier annotations
                if len(outliers) > 0:
                    # outlier details in an expander
                    with st.expander("View Outlier Details"):
                        st.write("#### Outlier Values")
                        outlier_df = pd.DataFrame({
                            'Value': outliers,
                            'Type': ['Low' if x < lower_bound else 'High' for x in outliers]
                        })
                        st.dataframe(outlier_df)
                else:
                    st.info(f"No outliers detected in {column}")
                st.divider()
            
        except Exception as e:
            st.error(f"Error processing {column}: {str(e)}")
            continue    




def hist_qq_plots(dataframe, selected_columns=None):
    """
    Create histogram and Q-Q plots using seaborn for efficiency
    """
    # numeric columns
    numeric_cols = [col_name for col_name, dtype in zip(dataframe.columns, dataframe.dtypes) 
                   if isinstance(dtype, (pl.Float32, pl.Float64, pl.Int32, pl.Int64))]
    
    # validate selected columns
    if selected_columns is None:
        columns_to_analyze = numeric_cols
    else:
        columns_to_analyze = [col for col in selected_columns if col in numeric_cols]
        if not columns_to_analyze:
            st.warning("None of the selected columns are numeric.")
            return

    # one column at a time
    for column in columns_to_analyze:
        try:
            st.info(f"### Analysis for {column}")
            
            data = dataframe[column].drop_nulls().to_numpy()
            
            if len(data) < 3:
                st.warning(f"Not enough data points in {column} for analysis")
                continue
            
            # plots layout
            col1, col2 = st.columns(2)
            
            with col1:
                # distribution plot using plotly
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data,
                    name="Histogram",
                    nbinsx=30,
                    histnorm='probability density'
                ))
                
                # KDE plot
                kde_x = np.linspace(np.min(data), np.max(data), 100)
                kde = stats.gaussian_kde(data)
                fig.add_trace(go.Scatter(
                    x=kde_x,
                    y=kde(kde_x),
                    name="KDE",
                    line=dict(color='red')
                ))
                
                fig.update_layout(
                    title="Distribution Plot",
                    xaxis_title=column,
                    yaxis_title="Density",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig)
            
            with col2:
                # QQ plot using seaborn/matplotlib
                fig, ax = plt.subplots(figsize=(8, 6))
                stats.probplot(data, dist="norm", plot=ax)
                ax.set_title("Q-Q Plot")
                
                # matplotlib to streamlit
                st.pyplot(fig)
                plt.close(fig)
            
            # statistical tests
            with st.container():
                st.markdown("### Statistical Tests for Normality")
                test_col1, test_col2 = st.columns(2)
                
                with test_col1:
                    try:
                        # data to float64 to ensure compatibility
                        test_data = data.astype(np.float64)
                        
                        if len(test_data) < 3:
                            st.warning("Not enough samples for normality tests")
                        elif len(test_data) <= 5000:
                            # Use Shapiro-Wilk test for smaller samples
                            shapiro_test = stats.shapiro(test_data)
                            st.markdown("#### Shapiro-Wilk Test")
                            st.markdown(f"- Statistic: {shapiro_test[0]:.4f}")
                            st.markdown(f"- p-value: {shapiro_test[1]:.4f}")
                            
                            if shapiro_test[1] < 0.05:
                                st.markdown("❌ Data is **not** normally distributed (p < 0.05)")
                            else:
                                st.markdown("✅ Data appears to be normally distributed (p >= 0.05)")
                        else:
                            # Kolmogorov-Smirnov test for larger samples
                            # Generate normal distribution parameters from the data
                            mu, std = np.mean(test_data), np.std(test_data)
                            ks_test = stats.kstest(test_data, 'norm', args=(mu, std))
                            
                            st.markdown("#### Kolmogorov-Smirnov Test")
                            st.markdown(f"- Statistic: {ks_test[0]:.4f}")
                            st.markdown(f"- p-value: {ks_test[1]:.4f}")
                            
                            if ks_test[1] < 0.05:
                                st.markdown("❌ Data is **not** normally distributed (p < 0.05)")
                            else:
                                st.markdown("✅ Data appears to be normally distributed (p >= 0.05)")
                        
                        # Anderson-Darling test (works for any sample size)
                        anderson_test = stats.anderson(test_data)
                        st.markdown("#### Anderson-Darling Test")
                        st.markdown(f"- Statistic: {anderson_test.statistic:.4f}")
                        
                        # critical values are at significance levels [15%, 10%, 5%, 2.5%, 1%]
                        significance_levels = [15, 10, 5, 2.5, 1]
                        
                        # compare test statistic with critical values
                        for sl, cv in zip(significance_levels, anderson_test.critical_values):
                            if anderson_test.statistic < cv:
                                st.markdown(f"✅ Normal at {sl}% significance level")
                                break
                        else:
                            st.markdown("❌ Not normally distributed at any common significance level")
                            
                    except Exception as e:
                        st.warning(f"Could not perform normality tests: {str(e)}")
                
                with test_col2:
                    st.markdown("### Summary Statistics")
                    try:
                        col3, col4, col5 = st.columns(3)
                        with col3:
                            st.metric("Mean", f"{np.mean(data):.2f}")
                            st.metric("Standard Deviation", f"{np.std(data):.2f}")
                        
                        with col4:
                            st.metric("Skewness", f"{stats.skew(data):.2f}")
                            st.metric("Kurtosis", f"{stats.kurtosis(data):.2f}")
                        
                        with col5:
                            st.metric("Sample Size", len(data))
                            st.metric("Missing Values", dataframe[column].null_count())
                        
                    except Exception as e:
                        st.warning(f"Could not compute some statistics: {str(e)}")
            
            st.divider()
            
            # Clean up
            del data
            gc.collect()
            
        except Exception as e:
            st.error(f"Error processing {column}: {str(e)}")
            continue



def calculate_skewness(dataframe):
    """Calculate and visualize skewness"""
    numeric_data = SelectDataTypes(dataframe)
    skewness_values = []
    
    for column in numeric_data.columns:
        values = numeric_data[column].values
        n = len(values)
        mean = np.mean(values)
        std_dev = np.std(values)
        
        if std_dev == 0:
            skewness = 0
        else:
            skewness = (np.sum((values - mean) ** 3) * n / 
                       ((n - 1) * (n - 2) * std_dev ** 3))
        skewness_values.append(skewness)
    
    skewness_df = pd.DataFrame({
        'Column': numeric_data.columns,
        'Skewness': skewness_values
    }).sort_values('Skewness', ascending=True)
    
    # Visualization
    st.write("### Skewness Analysis")
    fig = px.bar(
        skewness_df,
        x='Column',
        y='Skewness',
        title='Skewness by Feature'
    )
    st.plotly_chart(fig)
    
    st.write("### Skewness Values")
    st.dataframe(skewness_df)
    
    return skewness_df

def calculate_kurtosis(dataframe):
    """Calculate and visualize kurtosis"""
    numeric_data = SelectDataTypes(dataframe)
    kurtosis_values = []
    
    # Calculate kurtosis using original formula
    for column in numeric_data.columns:
        values = numeric_data[column].values
        mean = np.mean(values)
        std_dev = np.std(values)
        
        if std_dev == 0:
            kurtosis = 0
        else:
            fourth_moment = np.mean((values - mean) ** 4)
            kurtosis = fourth_moment / (std_dev ** 4) - 3
            
        kurtosis_values.append(kurtosis)

    # dataFrame
    kurtosis_df = pd.DataFrame({
        'Column': numeric_data.columns,
        'Kurtosis': kurtosis_values
    }).sort_values(by='Kurtosis', ascending=True)
    
    # add interpretation column
    def interpret_kurtosis(k):
        if abs(k) < 0.5:
            return "Near Normal"
        elif k > 0:
            return f"Leptokurtic (+{k:.2f})"
        else:
            return f"Platykurtic ({k:.2f})"
    
    kurtosis_df['Interpretation'] = kurtosis_df['Kurtosis'].apply(interpret_kurtosis)
    
    # transposed version for AI analysis
    global k_df
    k_df = kurtosis_df.transpose()
    
    # display results
    with st.spinner("Please wait... DataFrame Loading"):
        st.markdown("### 📘 Interpretation Guide")
        st.markdown("""
        - **Kurtosis = 0**: Normal distribution (mesokurtic)
        - **Kurtosis > 0**: Heavy-tailed distribution (leptokurtic)
        - **Kurtosis < 0**: Light-tailed distribution (platykurtic)
        """)
        
        st.markdown("### 📋 Detailed Results")
        st.dataframe(
            kurtosis_df.style
            .background_gradient(subset=['Kurtosis'], cmap='RdYlBu')
            .format({'Kurtosis': '{:.2f}'})
        )
    
    # Create visualization
    with st.spinner("Please wait...Plotting Kurtosis values"):
        fig = px.line(
            kurtosis_df, 
            x='Column', 
            y='Kurtosis', 
            title="Kurtosis of Columns",
            markers=True
        )
        
        # reference line for normal distribution
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="grey", 
            annotation_text="Normal Distribution"
        )
        
        # update traces
        fig.update_traces(
            mode="lines+markers+text",
            text=kurtosis_df['Kurtosis'].round(2),
            textposition="top center"
        )
        
        # update layout
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Kurtosis Value",
            height=500
        )
        
        st.plotly_chart(fig)
    
    return kurtosis_df


def main():
    # initialize
    api_key, df, df_processed, df_encoded, df_scaled = initialize_page()
    if api_key:
        openai.api_key = api_key
    
    # create header
    create_header()
    
    # create container for data source selection
    with st.container():
        st.markdown("### 📂 Select Data Source")
        selected_state = st.selectbox(
            "Choose which version of your dataset to analyze",
            ["Intial DataFrame", 
             "DataFrame after Missing value Imputation",
             "DataFrame after Feature Encoding",
             "DataFrame after Feature Scaling"]
        )
        current_df = get_selected_data(selected_state, df, df_processed, df_encoded, df_scaled)
        st.divider()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribution & Q-Q Plots",
        "🏔️ Kurtosis",
        "↗️ Skewness",
        "🔍 Outliers"
    ])
    
    # tab1 content
    with tab1:
        st.markdown("## Distribution Analysis")
        st.markdown("""
        Analyze data distribution and normality using:
        - Distribution plot with density curve
        - Q-Q plot for normality assessment
        - Statistical tests
        """)
        
        # numeric columns
        numeric_cols = [col_name for col_name, dtype in zip(current_df.columns, current_df.dtypes) 
                    if isinstance(dtype, (pl.Float32, pl.Float64, pl.Int32, pl.Int64))]
        
        if len(numeric_cols) > 0:
            # Column selection with a limit
            max_columns = 3  # Limit to prevent memory issues
            selected_columns = st.multiselect(
                f"Select columns for analysis (max {max_columns})",
                numeric_cols,
                default=numeric_cols[0] if numeric_cols else None,
                max_selections=max_columns
            )
            
            if selected_columns:
                if st.button("📊 Generate Plots", key="dist_btn", use_container_width=True):
                    try:
                        with st.spinner("Generating plots..."):
                            hist_qq_plots(current_df, selected_columns)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Try selecting fewer columns or clearing cache")
            else:
                st.warning("Please select at least one column.")
        else:
            st.warning("No numeric columns found in the dataset.")

        

    with tab2:
        st.markdown("## Kurtosis Analysis")
        st.markdown("""
        Kurtosis measures the 'tailedness' of a distribution compared to a normal distribution.
        - **Positive kurtosis**: Heavy tails, more outliers
        - **Negative kurtosis**: Light tails, fewer outliers
        - **Normal distribution**: Kurtosis = 0
        """)
        
        if st.button("🔍 Calculate Kurtosis", key="kurtosis_btn", use_container_width=True):
            with st.spinner("Calculating kurtosis..."):
                kurtosis_df = calculate_kurtosis(current_df)
                
                if api_key:
                    with st.spinner("Generating AI insights..."):
                        agent = create_pandas_dataframe_agent(
                            ChatOpenAI(temperature=0, model="gpt-4-turbo", api_key=api_key),
                            kurtosis_df,
                            verbose=True,
                            allow_dangerous_code=True,
                            agent_type=AgentType.OPENAI_FUNCTIONS,
                        )
                        st.markdown("### 🤖 AI Insights")
                        st.markdown(agent.run("Analyze the kurtosis values and explain their implications."))
    
    with tab3:
        st.markdown("## Skewness Analysis")
        st.markdown("""
        Skewness measures the asymmetry of a distribution.
        - **Positive skewness**: Right tail is longer
        - **Negative skewness**: Left tail is longer
        - **Normal distribution**: Skewness = 0
        """)
        
        if st.button("🔍 Calculate Skewness", key="skewness_btn", use_container_width=True):
            with st.spinner("Calculating skewness..."):
                skewness_df = calculate_skewness(current_df)
                
                if api_key:
                    with st.spinner("Generating AI insights..."):
                        agent = create_pandas_dataframe_agent(
                            ChatOpenAI(temperature=0, model="gpt-4-turbo", api_key=api_key),
                            skewness_df,
                            verbose=True,
                            allow_dangerous_code=True,
                            agent_type=AgentType.OPENAI_FUNCTIONS,
                        )
                        st.markdown("### 🤖 AI Insights")
                        st.markdown(agent.run("Analyze the skewness values and explain their implications."))
    
    with tab4:
        st.markdown("## Outlier Analysis")
        st.markdown("""
        Detect and visualize outliers using the Interquartile Range (IQR) method.
        - **IQR**: Q3 - Q1
        - **Outlier threshold**: 1.5 × IQR below Q1 or above Q3
        """)
        
        # Get numeric columns
        numeric_cols = [col_name for col_name, dtype in zip(current_df.columns, current_df.dtypes) 
                    if isinstance(dtype, (pl.Float32, pl.Float64, pl.Int32, pl.Int64))]
        
        if len(numeric_cols) > 0:
            # Column selection with a limit
            max_columns = 3  # Limit to prevent memory issues
            selected_columns = st.multiselect(
                f"Select columns for outlier analysis (max {max_columns})",
                numeric_cols,
                default=numeric_cols[0] if numeric_cols else None,
                max_selections=max_columns,
                key="outlier_cols"
            )
            
            # IQR multiplier selection
            k_value = st.slider(
                "Select IQR multiplier (k)",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Higher values are more conservative in detecting outliers"
            )
            
            if selected_columns:
                if st.button("🔍 Find Outliers", key="outlier_btn", use_container_width=True):
                    try:
                        with st.spinner("Detecting outliers..."):
                            find_outliers(current_df, selected_columns, k=k_value)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Try selecting fewer columns or clearing cache")
            else:
                st.warning("Please select at least one column for analysis.")
        else:
            st.warning("No numeric columns found in the dataset.")


if __name__ == "__main__":
    main()




