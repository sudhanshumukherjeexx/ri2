import streamlit as st
from utilities.select_data_types import SelectDataTypes
from utilities.viz import *
import time
cache_buster = int(time.time())



def initialize_page():
    """Initialize the page and check requirements"""
    if 'openai_api_key' not in st.session_state:
        st.error("⚠️ Please add your OpenAI API key on the Home Page to continue.")
        st.stop()
    
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("📤 Please upload a dataset to get started.")
        st.stop()
    
    # Initialize the visualization state if not exists
    if 'current_viz' not in st.session_state:
        st.session_state.current_viz = None
        
    return (st.session_state.openai_api_key, st.session_state.df,
            st.session_state.get('df_processed'),
            st.session_state.get('df_scaled'),
            st.session_state.get('df_encoded'))      #, st.session_state.df


def get_selected_data(selected_state, df, df_processed, df_scaled, df_encoded):
    """Get appropriate dataset based on selection"""
    data_dict = {
        "Initial DataFrame": df,
        "DataFrame after Missing value Imputation": df_processed,
        "DataFrame after Feature Scaling": df_scaled,
        "DataFrame after Feature Encoding": df_encoded
    }
    
    selected_df = data_dict.get(selected_state)
    if selected_df is None:
        st.warning(f"⚠️ Please complete {selected_state.lower()} first.")
        st.stop()
        
    return selected_df



def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/eda.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>📊 Exploratory Data Analysis</h2>
            <p style='font-size: 1.1em;'>Uncover patterns, trends, and insights in your data through interactive visualizations. 
            Choose from our comprehensive suite of plotting options below to begin your data exploration journey.</p>
        </div>
        """, unsafe_allow_html=True)

def create_visualization_grid():
    """Create a grid of visualization options"""
    VISUALIZATION_CARDS = [
        {
            "category": "📊Basic Plots",
            "plots": [
                {"image": "images/box_plot.png", "label": "Box Plot", "function": plot_boxplot},
                {"image": "images/histogram.png", "label": "Histogram", "function": plot_histogram},
                {"image": "images/scatter_plot.png", "label": "Scatter Plot", "function": plot_scatter},
                {"image": "images/statistic.png", "label": "Bar Chart", "function": plot_bar},
                {"image": "images/pie_chart.png", "label": "Pie Chart", "function": plot_pie},
                {"image": "images/line.png", "label": "Line Plot", "function": plot_line},
            ]
        },
        {
            "category": "📈Advanced Plots",
            "plots": [
                {"image": "images/2D_hist-contour_new.png", "label": "2D Hist Contour", "function": plot_hist_contour},
                {"image": "images/contour.png", "label": "Contour Plot", "function": plot_contour},
                {"image": "images/violin.png", "label": "Violin Plot", "function": plot_violin},
                {"image": "images/3d_scatter.png", "label": "3D Scatter", "function": plot_scatter_3d},
                {"image": "images/3d_line.png", "label": "3D Line", "function": plot_line_3d},
            ]
        },
        {
            "category": "🎯Specialized Plots",
            "plots": [
                {"image": "images/polar_scatter.png", "label": "Polar Scatter", "function": plot_polar_scatter},
                {"image": "images/polar_bar.png", "label": "Polar Bar", "function": plot_polar_bar},
            ]
        },
        {
            "category": "🗺️Geospatial Visualizations",
            "plots": [
                {"image": "images/tile_map.png", "label": "Scatter Map", "function": plot_st_map},
                {"image": "images/choropleth_map.png", "label": "Choropleth Map", "function": plot_choropleth_map},
                {"image": "images/bubble_plot.png", "label": "Bubble Map", "function": plot_bubble_map},
            ]
        }
    ]

    for category in VISUALIZATION_CARDS:
        st.markdown(f"#### {category['category']}")
        cols = st.columns(3)
        for idx, plot in enumerate(category['plots']):
            with cols[idx % 3]:
                with st.container():
                    st.image(plot['image'], width=100)
                    if st.button(plot['label'], 
                               key=f"btn_{plot['label']}", 
                               use_container_width=True,
                               help=f"Click to create a {plot['label']}"):
                        st.session_state.current_viz = plot['function'].__name__
                        st.rerun()

def render_plot(df, dtype_select_df):
    """Render the selected plot"""
    if not st.session_state.current_viz:
        return

    plot_name = st.session_state.current_viz
    
    # Add a back button
    col1, col2 = st.columns([1, 11])
    with col1:
        if st.button("⬅️ Back "): #←
            st.session_state.current_viz = None
            st.rerun()
    with col2:
        st.markdown(f"### 📈 {plot_name.replace('plot_', '').replace('_', ' ').title()}")
    
    with st.spinner("🎨🖌️Creating visualization..."):
        plot_function = globals()[plot_name]
        if plot_name in ['plot_bar', 'plot_pie', 'plot_violin', 'plot_scatter_3d', 
                        'plot_line_3d', 'plot_line', 'plot_polar_scatter', 'plot_polar_bar', 
                        'plot_st_map', 'plot_choropleth_map', 'plot_bubble_map']:
            plot_function(df)
        else:
            plot_function(dtype_select_df)



def main():
    st.set_page_config(page_title="Data Visualization", layout="wide")
    
    # Initialize and check requirements
    api_key, df, df_processed, df_scaled, df_encoded = initialize_page()
    
    # Create header
    create_header()
    st.markdown(f"#### 📂 Select Data Source")

    selected_state = st.selectbox(
            "Select DataFrame ↘️",
            ["Initial DataFrame", 
             "DataFrame after Missing value Imputation",
             "DataFrame after Feature Scaling",
             "DataFrame after Feature Encoding"]
        )
    
    current_df = get_selected_data(selected_state, df, df_processed, df_scaled, df_encoded)
    
    # Initialize data processing
    dtype_select_df = current_df#SelectDataTypes(current_df)
    
    # Show either visualization grid or the selected plot
    if st.session_state.current_viz is None:
        create_visualization_grid()
    else:
        #render_plot(df, dtype_select_df)
        render_plot(current_df, dtype_select_df)

if __name__ == "__main__":
    main()