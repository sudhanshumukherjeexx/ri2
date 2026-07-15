import streamlit as st
import polars as pl
import plotly.graph_objects as go
from utilities.select_data_types import SelectDataTypes
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from utilities.image_explanation import *
import openai
import os
import time
cache_buster = int(time.time())


# dataframe exists, else user upload
def initialize_page():
    """Initialize page requirements and API setup"""
    if 'openai_api_key' not in st.session_state:
        st.error("⚠️ Please add your OpenAI API key on the Home Page to continue.")
        st.stop()
        
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("📤 Please upload a dataset to get started.")
        st.stop()
    
    # Create plot_images directory if it doesn't exist
    os.makedirs("plot_images", exist_ok=True)
    
    return st.session_state.openai_api_key, st.session_state.df


# panel gif and info
def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/page_1.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>📊 Dataset Overview</h2>
            <p style='font-size: 1.1em;'>Explore your dataset's key characteristics including features, 
            data types, shape, missing values, and correlations. Get instant insights powered by AI analysis.</p>
        </div>
        """, unsafe_allow_html=True)

# panel features
def display_feature_cards():
    """Display feature information cards"""
    features = {
        "Data Header": "A glimpse of your dataset's initial rows, revealing key information and a snapshot of your data.",
        "Features Available": "A list of the dataset's variables, providing insights into the data's dimensions and content.",
        "Features Datatypes": "An overview of the data types associated with each feature, essential for data interpretation and manipulation.",
        "Shape of Data": "A concise summary of the dataset's dimensions, highlighting the number of rows and columns.",
        "Missing value count": "A tally of the number of missing or undefined values in the dataset, crucial for data quality assessment.",
        "Feature Correlation": "Insights into the relationships and associations among different variables in the dataset."
    }
    
    cols = st.columns(3)
    for idx, (title, description) in enumerate(features.items()):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style='background-color: white; padding: 15px; border-radius: 5px; border: 1px solid #e1e4e8; margin: 5px;'>
                <h4 style='color: #1f77b4;'>{title}</h4>
                <p style='font-size: 0.9em;'>{description}</p>
            </div>
            """, unsafe_allow_html=True)

# analyze results with llm
def analyze_with_ai(data, model, prompt):
    """Analyze data using AI agent"""
    with st.spinner('Analyzing data with AI...'):
        try:
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model=model, api_key=st.session_state.openai_api_key),
                data,
                verbose=True,
                number_of_head_rows=15,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code=True
            )
            return agent.run(prompt)
        except Exception as e:
            st.error(f"❌ AI Analysis Error: {str(e)}")
            return None

# correlation heatmap
def create_correlation_heatmap(df_numeric):
    """Create and display correlation heatmap"""
    corr = df_numeric.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.index.values,
        y=corr.columns.values,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="Correlation Heatmap",
        height=600,
        width=800
    )
    
    return fig

def main():
    # initialize
    api_key, df = initialize_page()
    openai.api_key = api_key
    
    create_header()
    
    display_feature_cards()
    
    # tabs for different analyses
    tab1, tab2 = st.tabs(["📊 Basic Statistics", "🔄 Correlations"])
    
    # dataset overview
    with tab1:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # data summary
        st.markdown("### Statistical Summary")
        stats_df = df.to_pandas().describe()
        st.dataframe(stats_df, use_container_width=True)


        # feature details
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Dataset Information")
            st.markdown(f"**Rows:** {df.shape[0]}")
            st.markdown(f"**Columns:** {df.shape[1]}")
            st.markdown("**Features:**")
            st.code("\n".join(df.columns))
        
        with col2:
            st.markdown("### Data Types")
            st.dataframe(df.to_pandas().dtypes)

            st.markdown("### Missing Values")
            missing_df = df.to_pandas().isnull().sum()
            missing_with_values = missing_df[missing_df > 0]
            if not missing_with_values.empty:
                st.dataframe(missing_with_values)
            else:
                st.success("✅ No missing values found!")
            
        if api_key:
            with st.expander("🤖 View AI Analysis"):
                stats_analysis = analyze_with_ai(
                    stats_df,
                    "gpt-4o-mini",
                    "Analyze the summary statistics and highlight key insights about the dataset's distribution and characteristics."
                )
                if stats_analysis:
                    st.info(stats_analysis)
        
    with tab2:
        st.markdown("### Correlation Analysis")
        df_numeric = SelectDataTypes(df)
        
        if df_numeric.shape[1] < 2:
            st.warning("⚠️ Not enough numeric columns for correlation analysis.")
        else:
            # correlation matrix
            st.markdown("#### Correlation Matrix")
            correlation_matrix = df_numeric.corr()
            st.dataframe(correlation_matrix, use_container_width=True)

            if api_key:
                with st.expander("🤖 View Correlation Analysis"):
                    corr_analysis = analyze_with_ai(
                        correlation_matrix,
                        "gpt-4o-mini",
                        "Analyze the correlation matrix and highlight significant relationships between variables."
                    )
                    if corr_analysis:
                        st.info(corr_analysis)
            
            # correlation heatmap
            st.markdown("#### Correlation Heatmap")
            fig = create_correlation_heatmap(df_numeric)
            st.plotly_chart(fig, use_container_width=True)
        
            if api_key:
                with st.expander("🤖 View Heatmap Analysis"):
                    fig.write_image("plot_images/corr_heatmap.png")
                    heatmap_analysis = AnalyzeImage(
                        "plot_images/corr_heatmap.png",
                        openai,
                        "Analyze the correlation heatmap and highlight key patterns and relationships."
                    )
                    st.info(ExtractContent(heatmap_analysis))
    
    
if __name__ == "__main__":
    main()