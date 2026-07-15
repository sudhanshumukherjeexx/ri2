import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from utilities.image_explanation import *
from functools import partial
import numpy as np
import uuid
import hashlib
import time
import openai
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from utilities.select_data_types import SelectDataTypes

class ClientNotInitializedError(Exception):
    """Exception raised when OpenAI client is not properly initialized."""
    pass


# def get_numerical_columns(data):
#     # Check if data is a pandas DataFrame or a polars DataFrame
#     if hasattr(data, 'select_dtypes'):  # pandas DataFrame
#         return data.select_dtypes(include=['int64', 'float64']).columns
#     elif hasattr(data, 'select'):       # polars DataFrame
#         return data.select([
#             pl.col(pl.FLOAT_DTYPES + pl.INTEGER_DTYPES)
#         ]).columns
#     else:
#         raise TypeError("Input must be either a pandas DataFrame or a polars DataFrame")

def get_numerical_columns(data):
    """
    Get numerical columns by first ensuring we have a pandas DataFrame.
    
    Args:
        data: Either pandas DataFrame or polars DataFrame
        
    Returns:
        list: List of column names with numerical data types
    """
    # Check if it's a pandas DataFrame
    if hasattr(data, 'select_dtypes'):
        pandas_df = data
    # If it's a polars DataFrame, convert to pandas
    elif hasattr(data, 'to_pandas'):
        pandas_df = data.to_pandas()
    else:
        raise TypeError("Input must be either a pandas DataFrame or a polars DataFrame")
    
    # Now we have a pandas DataFrame, we can use select_dtypes
    return list(pandas_df.select_dtypes(include=['int64', 'float64']).columns)





# plotting config class

@dataclass
class PlotConfig:
    """Base configuration for plot visualization and analysis."""
    image_dir: Path = Path("plot_images")
    image_name: str = "plot.png"
    prompt_template: str = ""

    def __post_init__(self):
        self.image_dir.mkdir(exist_ok=True)
        self.image_path = self.image_dir / self.image_name

@dataclass
class BoxPlotConfig(PlotConfig):
    """Configuration for box plot visualization and analysis."""
    image_name: str = "box.png"
    prompt_template: str = """
        Review the box plot image and succinctly outline its principal features. 
        Within 80 words, assess the central tendency, spread, and symmetry of the data.
        Highlight any outliers, and interpret the quartiles and median to understand the 
        distribution's shape and variability.
    """

@dataclass
class ScatterPlotConfig(PlotConfig):
    """Configuration for scatter plot visualization and analysis."""
    image_name: str = "scatter.png"
    prompt_template: str = """
        Examine the provided scatter plot image and concisely describe its key characteristics. 
        In 80 words or less, analyze the data's relationship, pattern, trend direction, and any 
        notable outliers or clusters that offer insights into the variables' correlation.
    """

@dataclass
class HistogramConfig(PlotConfig):
    """Configuration for histogram visualization and analysis."""
    image_name: str = "hist.png"
    prompt_template: str = """
        Analyze the provided histogram image, summarizing its distribution shape, 
        central tendency, variability, and any visible anomalies such as peaks, 
        skewness, or gaps in 80 words or less. Highlight the key insights this 
        histogram reveals about the dataset's underlying structure and possible 
        data groupings.
    """

@dataclass
class BarPlotConfig(PlotConfig):
    """Configuration for bar plot visualization and analysis."""
    image_name: str = "bar.png"
    prompt_template: str = """
        Examine the given bar plot image and succinctly discuss its main features. 
        In 80 words or less, describe the comparative magnitudes, variations between 
        categories, and any patterns or trends that emerge, including notable highs 
        and lows, to convey the categorical data's overall distribution and key takeaways.
    """

@dataclass
class ContourPlotConfig(PlotConfig):
    """Configuration for contour plot visualization and analysis."""
    image_name: str = "contour.png"
    prompt_template: str = """
        Review the contour plot image provided, and in 80 words or less, 
        explain its density levels, peak regions, and the spatial relationship 
        between variables. Highlight any significant gradients or clusters that 
        offer insights into the distribution and concentration areas within the 
        data landscape.
    """

@dataclass
class HistContourPlotConfig(PlotConfig):
    """Configuration for 2D histogram contour plot visualization and analysis."""
    image_name: str = "hist-contour.png"
    prompt_template: str = """
        Analyze the 2D histogram contour plot provided, focusing on density distribution 
        across two variables in 80 words or less. Note areas of highest and lowest concentration, 
        pattern gradients, and any distinct clusters, summarizing the key insights on variable 
        interaction and data spread.
    """

@dataclass
class PieChartConfig(PlotConfig):
    """Configuration for pie chart visualization and analysis."""
    image_name: str = "pie.png"
    prompt_template: str = """
        Examine the provided pie chart and succinctly describe its composition. In 80 words or less,
        discuss the proportional distribution among categories, identifying any dominant or minor 
        segments and the overall diversity within the dataset, to summarize key insights into the 
        categorical representation and relative significance.
    """

@dataclass
class ViolinPlotConfig(PlotConfig):
    """Configuration for violin plot visualization and analysis."""
    image_name: str = "violin.png"
    prompt_template: str = """
        Review the given violin plot, and in 80 words or less, explain its distribution shape, 
        central tendencies, and variability. Highlight any noticeable peaks, widths, and symmetries 
        or asymmetries to convey insights into the dataset's density and the spread of values across 
        categories.
    """

@dataclass
class LinePlotConfig(PlotConfig):
    """Configuration for line plot visualization and analysis."""
    image_name: str = "line.png"
    prompt_template: str = """
        Examine the provided line plot and briefly discuss its key aspects. In 80 words or less, 
        describe the trend, fluctuations, and any patterns observed over time or another continuous 
        variable, noting significant peaks or troughs to summarize the dataset's temporal or sequential
        behavior and insights.
    """

@dataclass
class Scatter3DConfig(PlotConfig):
    """Configuration for 3D scatter plot visualization and analysis."""
    image_name: str = "scatter_3d.png"
    prompt_template: str = """
        Analyze the 3D scatter plot presented, focusing on the spatial distribution 
        and relationship among three variables in 80 words or less. Note clusters, 
        outliers, and density trends to summarize the multidimensional interactions 
        and insights into how these variables correlate and distribute in a 
        three-dimensional space.
    """

@dataclass
class Line3DConfig(PlotConfig):
    """Configuration for 3D line plot visualization and analysis."""
    image_name: str = "line_3d.png"
    prompt_template: str = """
        Inspect the 3D line plot provided, and in 80 words or less, elucidate on 
        its trajectory through three-dimensional space. Highlight any patterns, 
        directionality, or notable features such as loops or peaks to convey the 
        dynamic interactions and trends observed across the three axes over time 
        or conditions.
    """

@dataclass
class PolarScatterConfig(PlotConfig):
    """Configuration for polar scatter plot visualization and analysis."""
    image_name: str = "polar_scatter.png"
    prompt_template: str = """
        Review the polar scatter plot image provided, and concisely describe its 
        distribution and pattern in 80 words or less. Highlight radial distances, 
        angular clustering, and any notable outliers, summarizing insights into the 
        dataset's cyclical tendencies or variations based on angle and radius within 
        the polar coordinate system.
    """

@dataclass
class PolarBarConfig(PlotConfig):
    """Configuration for polar bar plot visualization and analysis."""
    image_name: str = "polar_bar.png"
    prompt_template: str = """
        Examine the provided polar bar plot, summarizing its radial distribution 
        and categorical representation in 80 words or less. Note the angular 
        segments and their lengths, identifying significant variations and patterns
        that emerge in a circular layout, to convey insights into category 
        performance or prevalence in a polar context.
    """

@dataclass
class TileMapConfig(PlotConfig):
    """Configuration for tile map visualization and analysis."""
    image_name: str = "tile_map.png"
    prompt_template: str = """
        Analyze the tile map provided, focusing on spatial distribution and patterns 
        in 80 words or less. Observe the arrangement of tiles, color intensities, and 
        any clustering, to summarize geographical or categorical variations, highlighting 
        key areas of interest or anomalies within the mapped dataset.
    """

@dataclass
class ChoroplethMapConfig(PlotConfig):
    """Configuration for choropleth map visualization."""
    image_name: str = "choropleth_map.png"

# plotting classes

def ensure_client():
    """Ensure OpenAI client is initialized in session state."""
    if 'openai_api_key' not in st.session_state:
        st.error("OpenAI client not initialized. Please initialize it first.")
        raise ClientNotInitializedError("OpenAI client must be initialized in session state before using visualization functions.")
    else:
        user_api_key = st.session_state.openai_api_key
        # Set up the OpenAI client
        openai.api_key = user_api_key
        client = openai
    return client

class PlotAnalyzer:
    """Base class for plot generation and analysis."""
    
    def __init__(self, config: PlotConfig):
        self.config = config

    def analyze_plot(self, plot_fig):
        """Save and analyze the plot using GPT Vision."""
        try:
            client = ensure_client()
            plot_fig.write_image(str(self.config.image_path))
            with st.spinner("Analyzing this plot using AI"):
                result = AnalyzeImage(str(self.config.image_path), client, self.config.prompt_template)
                return ExtractContent(result)
        except ClientNotInitializedError:
            return "Plot analysis unavailable - OpenAI client not initialized."
        except Exception as e:
            st.error(f"Error analyzing plot: {str(e)}")
            return "Unable to analyze plot due to an error."

# class BoxPlotAnalyzer(PlotAnalyzer):
#     """Class to handle box plot generation and analysis."""
    
#     def create_box_plot(self, data, feature: str) -> px.box:
#         """Create a box plot for a specific feature."""
#         return px.box(data, y=feature, notched=False)

#     def display_single_plot(self, data, feature: str) -> None:
#         """Display and analyze a single feature box plot."""
#         st.subheader(f"Box Plot for {feature}")
#         plot_fig = self.create_box_plot(data, feature)
#         st.plotly_chart(plot_fig, use_container_width=True)
#         analysis = self.analyze_plot(plot_fig)
#         st.markdown(analysis)

class BoxPlotAnalyzer(PlotAnalyzer):
    """Class to handle box plot generation and analysis with statistical measures."""
    def calculate_statistics(self, data, feature: str) -> dict:
        """Calculate statistical measures including outliers for both pandas and polars dataframes."""
        import pandas as pd
        
        # Check if data is pandas DataFrame or Series
        is_pandas = isinstance(data, pd.DataFrame) or isinstance(data[feature], pd.Series)
        
        if is_pandas:
            # Pandas approach
            values = data[feature].dropna()
            
            # Basic statistics
            mean_val = values.mean()
            std_val = values.std()
            
            # Calculate IQR and outlier bounds
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers
            outliers = values[(values < lower_bound) | (values > upper_bound)]
        else:
            # Polars approach
            values = data[feature].drop_nulls()
            
            # Basic statistics
            mean_val = values.mean()
            std_val = values.std()
            
            # Calculate IQR and outlier bounds
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers
            outliers = values.filter((values < lower_bound) | (values > upper_bound))
        
        return {
            'mean': mean_val,
            'std': std_val,
            'outliers_count': len(outliers),
            'outliers': outliers,
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        }
    # def calculate_statistics(self, data, feature: str) -> dict:
    #     """Calculate statistical measures including outliers."""
    #     values = data[feature].dropna()
        
    #     # Basic statistics
    #     mean_val = values.mean()
    #     std_val = values.std()
        
    #     # Calculate IQR and outlier bounds
    #     q1 = values.quantile(0.25)
    #     q3 = values.quantile(0.75)
    #     iqr = q3 - q1
    #     lower_bound = q1 - 1.5 * iqr
    #     upper_bound = q3 + 1.5 * iqr
        
    #     # Find outliers
    #     outliers = values[(values < lower_bound) | (values > upper_bound)]
        
    #     return {
    #         'mean': mean_val,
    #         'std': std_val,
    #         'outliers_count': len(outliers),
    #         'outliers': outliers,
    #         'q1': q1,
    #         'q3': q3,
    #         'iqr': iqr
    #     }
    
    def create_box_plot(self, data, feature: str) -> px.box:
        """Create an enhanced box plot with mean marker."""
        fig = px.box(data, y=feature, notched=False)
        
        # Add mean line
        mean_val = data[feature].mean()
        fig.add_hline(
            y=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text="Mean",
            annotation_position="top right"
        )
        
        return fig
    
    def format_statistics(self, stats: dict) -> str:
        """Format statistical measures for display."""
        col1, col2 = st.columns(2)
        with col1:
            
            st.info(f"""
                    ### Statistical Analysis\n
                    **Mean**: {stats['mean']:.2f}\n
                    **Standard Deviation**: {stats['std']:.2f}\n
                    **Number of Outliers**: {stats['outliers_count']}\n
                    **IQR (Interquartile Range)**: {stats['iqr']:.2f}""")
        with col2:
            st.info(f"""
                    #### Outlier Details\n
                    **Q1 (25th percentile)**: {stats['q1']:.2f}\n
                    **Q3 (75th percentile)**: {stats['q3']:.2f}\n
                    **Outlier values**: {', '.join(f'{x:.2f}' for x in stats['outliers'][:5])}{"..." if len(stats['outliers']) > 5 else "No Outliers Present"}
                """)
        return ""
    
    def display_single_plot(self, data, feature: str) -> None:
        """Display and analyze a single feature box plot with enhanced statistics."""
        st.subheader(f"Box Plot Analysis for {feature}")
        
        # Calculate statistics
        stats = self.calculate_statistics(data, feature)
        
        # Create and display plot
        plot_fig = self.create_box_plot(data, feature)
        st.plotly_chart(plot_fig, use_container_width=True)
        
        # Display statistical analysis
        st.markdown(self.format_statistics(stats))

class ScatterPlotAnalyzer(PlotAnalyzer):
    """Class to handle scatter plot generation and analysis."""
    
    def create_scatter_plot(self, data, x_axis: str, y_axis: str, hue: Optional[str] = None) -> px.scatter:
        """Create a scatter plot with optional color grouping."""
        return px.scatter(data, x=x_axis, y=y_axis, color=hue)

    def display_plot(self, data) -> None:
        """Display and analyze scatter plot with user-selected features."""
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox(
                'Choose X-Axis Feature',
                options=data.columns,
                index=None,
                placeholder="[ Select X-Axis value ]"
            )
        with col2:
            y_axis = st.selectbox(
                'Choose Y-Axis Feature',
                options=data.columns,
                index=None,
                placeholder="[ Select Y-Axis value ]"
            )

        if x_axis and y_axis:
            hue = st.selectbox(
                "Color points by feature (optional)",
                options=[None] + list(data.columns),
                index=0,
                placeholder="[ Select Feature for Color Grouping ]"
            )
            
            plot_fig = self.create_scatter_plot(data, x_axis, y_axis, hue)
            st.plotly_chart(plot_fig, use_container_width=True)
            analysis = self.analyze_plot(plot_fig)
            st.markdown(analysis)
        else:
            st.warning("Please select both X and Y axes for the scatter plot.")

class HistogramAnalyzer(PlotAnalyzer):
    """Class to handle histogram generation and analysis."""
    
    def create_histogram(self, data, feature: str, bins: Optional[int] = None) -> px.histogram:
        """Create a histogram for a specific feature."""
        hist_fig = px.histogram(
            data,
            x=feature,
            nbins=bins,
            marginal="box",
            histnorm="probability density"
        )
        hist_fig.update_layout(
            showlegend=False,
            xaxis_title=feature,
            yaxis_title="Density",
            bargap=0.1
        )
        return hist_fig

    def display_single_plot(self, data, feature: str, bins: Optional[int] = None, key_prefix: str = "") -> None:
        """
        Display and analyze a single feature histogram.
        
        Args:
            data: DataFrame containing the data
            feature: Feature to plot
            bins: Number of bins for the histogram
            key_prefix: Prefix for the streamlit widget key to ensure uniqueness
        """
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"Histogram for {feature}")
        with col2:
            # Create a unique key by combining prefix and feature name
            widget_key = f"{key_prefix}_bins_{feature}"
            bins = st.number_input(
                "Number of bins",
                min_value=5,
                max_value=100,
                value=30,
                key=widget_key
            )

        plot_fig = self.create_histogram(data, feature, bins)
        st.plotly_chart(plot_fig, use_container_width=True)
        analysis = self.analyze_plot(plot_fig)
        st.markdown(analysis)

@st.cache_resource
def get_histogram_analyzer():
    """Cache the histogram analyzer to prevent recreation."""
    return HistogramAnalyzer(HistogramConfig())

class BarPlotAnalyzer(PlotAnalyzer):
    """Class to handle bar plot generation and analysis."""
    
    def create_bar_plot(
        self, 
        data, 
        x_axis: str, 
        y_axis: str,
        color: Optional[str] = None,
        orientation: str = 'v'
    ) -> Optional[px.bar]:
        """Create a bar plot with the specified parameters."""
        try:
            plot_fig = px.bar(
                data,
                x=x_axis,
                y=y_axis,
                color=color,
                orientation=orientation,
                barmode='group' if color else 'relative',
                labels={
                    x_axis: x_axis.replace('_', ' ').title(),
                    y_axis: y_axis.replace('_', ' ').title()
                }
            )
            
            plot_fig.update_layout(
                bargap=0.2,
                bargroupgap=0.1,
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title()
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating bar plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze bar plot with user-selected features."""
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox(
                'Select X-Axis (Categories)',
                options=data.columns,
                index=None,
                placeholder="[ Select X-Axis value ]",
                key="bar_x_axis"
            )
        
        with col2:
            y_axis = st.selectbox(
                'Select Y-Axis (Values)',
                options=data.columns, #.select_dtypes(include=['int64', 'float64']).columns,
                index=None,
                placeholder="[ Select Y-Axis value ]",
                key="bar_y_axis"
            )

        if x_axis and y_axis:
            with st.expander("Plot Options", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    orientation = st.radio(
                        "Plot Orientation",
                        options=['Vertical', 'Horizontal'],
                        index=0,
                        key="bar_orientation"
                    )
                with col2:
                    color_var = st.selectbox(
                        "Group by (optional)",
                        options=[None] + list(data.columns),
                        index=0,
                        key="bar_color"
                    )

            try:
                plot_fig = self.create_bar_plot(
                    data,
                    x_axis,
                    y_axis,
                    color=color_var,
                    orientation='v' if orientation == 'Vertical' else 'h'
                )
                
                if plot_fig:
                    st.plotly_chart(plot_fig, use_container_width=True)
                    # Use the base class's analyze_plot method which handles client checking
                    analysis = self.analyze_plot(plot_fig)
                    st.markdown(analysis)
                    # if analysis:  # Only show analysis if it's available
                    #     st.markdown(analysis)
                    
            except Exception as e:
                st.error(f"""
                    Error creating plot. Please ensure:
                    1. Y-axis selection is numerical
                    2. X-axis values are suitable for categories
                    3. Data is properly formatted
                    
                    Error details: {str(e)}
                """)
        else:
            st.info("Please select both X and Y axes to create the bar plot.")

class ContourPlotAnalyzer(PlotAnalyzer):
    """Class to handle contour plot generation and analysis."""
    
    def create_contour_plot(
        self, 
        data, 
        x_axis: str, 
        y_axis: str,
        z_axis: str,
        colorscale: str = 'Electric',
        smoothing: float = 1.3
    ) -> Optional[go.Figure]:
        """
        Create a contour plot with the specified parameters.
        
        Args:
            data: DataFrame containing the data
            x_axis: Column name for x-axis
            y_axis: Column name for y-axis
            z_axis: Column name for z-axis (contour values)
            colorscale: Color scheme for the contour plot
            smoothing: Line smoothing factor
        """
        try:
            # Convert to pandas if needed
            if hasattr(data, 'to_pandas'):
                data = data.to_pandas()
            
            # Verify numerical data
            for col in [x_axis, y_axis, z_axis]:
                if not np.issubdtype(data[col].dtype, np.number):
                    raise ValueError(f"Column '{col}' must contain numerical data")

            fig = go.Figure()
            fig.add_trace(
                go.Contour(
                    x=data[x_axis],
                    y=data[y_axis],
                    z=data[z_axis],
                    line_smoothing=smoothing,
                    colorscale=colorscale,
                    colorbar=dict(
                        title=z_axis.replace('_', ' ').title()
                    )
                )
            )
            
            # Update layout with better formatting
            fig.update_layout(
                title=f"Contour Plot: {z_axis} vs {x_axis} and {y_axis}",
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                height=600
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating contour plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze contour plot with user-selected features."""
        
        # Create columns for inputs
        col1, col2, col3 = st.columns(3)
        
        numerical_cols = get_numerical_columns(data)           #data.select_dtypes(include=['int64', 'float64']).columns
        
        with col1:
            x_axis = st.selectbox(
                'Select X-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select X-Axis value ]",
                key="contour_x_axis"
            )
        
        with col2:
            y_axis = st.selectbox(
                'Select Y-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Y-Axis value ]",
                key="contour_y_axis"
            )
            
        with col3:
            z_axis = st.selectbox(
                'Select Z-Axis (Contour Values)',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Z-Axis value ]",
                key="contour_z_axis"
            )

        if all([x_axis, y_axis, z_axis]):
            # Additional plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    colorscale = st.selectbox(
                        "Color Scale",
                        options=['Electric', 'Viridis', 'Plasma', 'Hot', 'Cool'],
                        index=0,
                        key="contour_colorscale"
                    )
                with col2:
                    smoothing = st.slider(
                        "Line Smoothing",
                        min_value=0.0,
                        max_value=2.0,
                        value=1.3,
                        step=0.1,
                        key="contour_smoothing"
                    )

            try:
                plot_fig = self.create_contour_plot(
                    data,
                    x_axis,
                    y_axis,
                    z_axis,
                    colorscale=colorscale,
                    smoothing=smoothing
                )
                
                if plot_fig:
                    st.plotly_chart(plot_fig, use_container_width=True)
                    analysis = self.analyze_plot(plot_fig)
                    if analysis:
                        st.markdown(analysis)
                    
            except Exception as e:
                st.error(f"""
                    Error creating plot. Please ensure:
                    1. All selected columns contain numerical data
                    2. Data is properly formatted and contains no missing values
                    3. There are enough unique values to create contours
                    
                    Error details: {str(e)}
                """)
        else:
            st.warning("Please select X, Y and Z axes for the Contour plot.")

class HistContourPlotAnalyzer(PlotAnalyzer):
    """Class to handle 2D histogram contour plot generation and analysis."""
    
    def create_hist_contour_plot(
        self, 
        data, 
        x_axis: str, 
        y_axis: str,
        color: Optional[str] = None,
        nbinsx: int = 20,
        nbinsy: int = 20
    ) -> Optional[px.density_contour]:
        """
        Create a 2D histogram contour plot with the specified parameters.
        
        Args:
            data: DataFrame containing the data
            x_axis: Column name for x-axis
            y_axis: Column name for y-axis
            color: Optional column name for color grouping
            nbinsx: Number of bins for x-axis histogram
            nbinsy: Number of bins for y-axis histogram
        """
        try:
            # Convert to pandas if needed
            if hasattr(data, 'to_pandas'):
                data = data.to_pandas()
            
            # Verify numerical data
            for col in [x_axis, y_axis]:
                if not np.issubdtype(data[col].dtype, np.number):
                    raise ValueError(f"Column '{col}' must contain numerical data")

            plot_fig = px.density_contour(
                data,
                x=x_axis,
                y=y_axis,
                color=color,
                marginal_x="histogram",
                marginal_y="histogram",
                nbinsx=nbinsx,
                nbinsy=nbinsy
            )
            
            # Update layout with better formatting
            plot_fig.update_layout(
                title=f"2D Histogram Contour: {x_axis} vs {y_axis}",
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                height=700  # Increased height to accommodate marginal plots
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating 2D histogram contour plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze 2D histogram contour plot with user-selected features."""
        
        # Get numerical columns for axis selection
        numerical_cols =  get_numerical_columns(data)          #data.select_dtypes(include=['int64', 'float64']).columns
        
        # Create columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox(
                'Select X-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select X-Axis value ]",
                key="hist_contour_x_axis"
            )
        
        with col2:
            y_axis = st.selectbox(
                'Select Y-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Y-Axis value ]",
                key="hist_contour_y_axis"
            )

        # Color grouping selection
        color = st.selectbox(
            "Color by Feature (optional)",
            options=[None] + list(data.columns),
            index=0,
            key="hist_contour_color"
        )

        if x_axis and y_axis:
            # Additional plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    nbinsx = st.slider(
                        "Number of X-axis Bins",
                        min_value=5,
                        max_value=50,
                        value=20,
                        key="hist_contour_nbinsx"
                    )
                with col2:
                    nbinsy = st.slider(
                        "Number of Y-axis Bins",
                        min_value=5,
                        max_value=50,
                        value=20,
                        key="hist_contour_nbinsy"
                    )

            try:
                plot_fig = self.create_hist_contour_plot(
                    data,
                    x_axis,
                    y_axis,
                    color=color,
                    nbinsx=nbinsx,
                    nbinsy=nbinsy
                )
                
                if plot_fig:
                    st.plotly_chart(plot_fig, use_container_width=True)
                    analysis = self.analyze_plot(plot_fig)
                    if analysis:
                        st.markdown(analysis)
                    
            except Exception as e:
                st.error(f"""
                    Error creating plot. Please ensure:
                    1. Selected columns contain numerical data
                    2. Data is properly formatted and contains no missing values
                    3. There are enough unique values to create meaningful contours
                    
                    Error details: {str(e)}
                """)
        else:
            st.warning("Please select both X and Y axes for the 2D Histogram Contour plot.")

class PieChartAnalyzer(PlotAnalyzer):
    """Class to handle pie chart generation and analysis."""
    
    def create_pie_chart(
        self, 
        data, 
        values: str, 
        names: str,
        color_scheme: str = 'Teal',
        hole: float = 0.0
    ) -> Optional[px.pie]:
        """
        Create a pie chart with the specified parameters.
        """
        try:
            plot_fig = px.pie(
                data,
                values=values,
                names=names,
                title=f"Distribution of {values} by {names}",
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                hole=hole
            )
            
            plot_fig.update_layout(
                showlegend=True,
                legend_title=names.replace('_', ' ').title(),
                height=600
            )
            
            plot_fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze pie chart with user-selected features."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            values = st.selectbox(
                'Select Values (Numerical Feature)',
                options=data.columns,
                index=None,
                placeholder="[ Numerical Feature ]",
                key="pie_values"
            )
        
        with col2:
            names = st.selectbox(
                'Select Categories (Categorical Feature)',
                options=data.columns,
                index=None,
                placeholder="[ Categorical Feature ]",
                key="pie_names"
            )

        if values and names:
            with st.expander("Plot Options", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=['Teal', 'Viridis', 'Plasma', 'Blues', 'Greens', 'Purples'],
                        index=0,
                        key="pie_colorscheme"
                    )
                with col2:
                    chart_type = st.radio(
                        "Chart Type",
                        options=["Pie Chart", "Donut Chart"],
                        index=0,
                        key="pie_type"
                    )
                    hole = 0.4 if chart_type == "Donut Chart" else 0.0

            plot_fig = self.create_pie_chart(
                data,
                values,
                names,
                color_scheme=color_scheme,
                hole=hole
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select both values and categories for the Pie Chart.")

class ViolinPlotAnalyzer(PlotAnalyzer):
    """Class to handle violin plot generation and analysis."""
    
    def create_violin_plot(
        self, 
        data, 
        y_axis: str,
        color: Optional[str] = None,
        box: bool = True,
        points: str = 'outliers',
        color_scheme: str = 'RdBu'
    ) -> Optional[px.violin]:
        """
        Create a violin plot with the specified parameters.
        
        Args:
            data: DataFrame containing the data
            y_axis: Column name for the violin plot
            color: Optional column name for grouping
            box: Whether to show box plot inside violin
            points: How to show points ('all', 'outliers')
            color_scheme: Color scheme for the plot
        """
        try:
            plot_fig = px.violin(
                data,
                y=y_axis,
                color=color,
                box=box,
                points=points,
                title=f"Distribution of {y_axis}",
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme)
            )
            
            # Update layout
            plot_fig.update_layout(
                height=600,
                showlegend=True if color else False,
                yaxis_title=y_axis.replace('_', ' ').title(),
                title_x=0.5
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating violin plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze violin plot with user-selected features."""
        
        # Get numerical columns
        numerical_cols = data.columns
        
        col1, col2 = st.columns(2)
        
        with col1:
            y_axis = st.selectbox(
                'Select Feature to Visualize',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Feature ]",
                key="violin_y_axis"
            )
            
        with col2:
            color = st.selectbox(
                'Group by (optional)',
                options=[None] + list(data.columns),
                index=0,
                key="violin_color"
            )

        if y_axis:
            # Plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=[
                            'RdBu',      # Red-Blue
                            'Viridis',   # Green-Yellow-Blue
                            'Plasma',    # Purple-Orange
                            'Blues',     # Blue shades
                            'Greens',    # Green shades
                            'Reds',      # Red shades
                            'Purples',   # Purple shades
                            'Teal',      # Teal shades
                            'BuPu',      # Blue-Purple
                            'OrRd'       # Orange-Red
                        ],
                        index=0,
                        key="violin_colorscheme"
                    )
                
                with col2:
                    show_box = st.checkbox(
                        "Show Box Plot",
                        value=True,
                        key="violin_show_box"
                    )
                
                with col3:
                    points_display = st.radio(
                        "Show Points",
                        options=['outliers', 'all'],
                        index=0,
                        key="violin_points"
                    )

            plot_fig = self.create_violin_plot(
                data,
                y_axis,
                color=color,
                box=show_box,
                points=points_display,
                color_scheme=color_scheme
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select a feature to display the Violin Plot")

class LinePlotAnalyzer(PlotAnalyzer):
    """Class to handle line plot generation and analysis."""
    
    def create_line_plot(
        self, 
        data, 
        x_axis: str, 
        y_axis: str,
        color: Optional[str] = None,
        line_shape: str = 'linear',
        markers: bool = True,
        color_scheme: str = 'viridis'
    ) -> Optional[px.line]:
        """
        Create a line plot with the specified parameters.
        """
        try:
            plot_fig = px.line(
                data,
                x=x_axis,
                y=y_axis,
                color=color,
                markers=markers,
                line_shape=line_shape,
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                title=f"{y_axis} vs {x_axis}"
            )
            
            # Update layout
            plot_fig.update_layout(
                height=600,
                showlegend=True if color else False,
                xaxis_title=x_axis.replace('_', ' ').title(),
                yaxis_title=y_axis.replace('_', ' ').title(),
                title_x=0.5,
                hovermode='x unified'
            )
            
            # Update line and marker properties
            plot_fig.update_traces(
                line=dict(width=2),
                marker=dict(size=6)
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating line plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze line plot with user-selected features."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox(
                'Select X-Axis',
                options=data.columns,
                index=None,
                placeholder="[ Select X-Axis value ]",
                key="line_x_axis"
            )
        
        with col2:
            y_axis = st.selectbox(
                'Select Y-Axis',
                options=data.columns,
                index=None,
                placeholder="[ Select Y-Axis value ]",
                key="line_y_axis"
            )

        if x_axis and y_axis:
            # Color grouping selection
            color = st.selectbox(
                "Color lines by feature (optional)",
                options=[None] + list(data.columns),
                index=0,
                key="line_color"
            )

            # Plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=[
                            'Jet',
                            'Blugrn',
                            'Teal',
                            'Aggrnyl',    
                            'Bluered',    
                            'Blackbody',  
                            'Rainbow',    
                            'Sunset'     
                        ],
                        index=0,
                        key="line_colorscheme"
                    )
                
                with col2:
                    line_shape = st.selectbox(
                        "Line Shape",
                        options=[
                            'linear',
                            'spline',
                            'hv',
                            'vh',
                            'hvh',
                            'vhv'
                        ],
                        index=0,
                        key="line_shape"
                    )
                
                with col3:
                    show_markers = st.checkbox(
                        "Show Markers",
                        value=True,
                        key="line_markers"
                    )

            plot_fig = self.create_line_plot(
                data,
                x_axis,
                y_axis,
                color=color,
                line_shape=line_shape,
                markers=show_markers,
                color_scheme=color_scheme
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select both X and Y axes for the line plot.")

class Scatter3DAnalyzer(PlotAnalyzer):
    """Class to handle 3D scatter plot generation and analysis."""
    
    # color sequences
    COLOR_SEQUENCES = [
        'Jet',
        'Blugrn',
        'Teal',
        'Aggrnyl',    
        'Bluered',    
        'Blackbody',  
        'Rainbow',    
        'Sunset'      
    ]
    
    def create_3d_scatter(
        self, 
        data, 
        x_axis: str, 
        y_axis: str,
        z_axis: str,
        color: Optional[str] = None,
        color_scheme: str = 'Teal',
        marker_size: int = 5,
        opacity: float = 0.7
    ) -> Optional[px.scatter_3d]:
        """
        Create a 3D scatter plot with the specified parameters.
        """
        try:
            plot_fig = px.scatter_3d(
                data,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color,
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                title=f"3D Scatter Plot: {x_axis} vs {y_axis} vs {z_axis}"
            )
            
            # Update layout
            plot_fig.update_layout(
                height=700,  # Larger height for better 3D visualization
                showlegend=True if color else False,
                title_x=0.5,
                scene=dict(
                    xaxis_title=x_axis.replace('_', ' ').title(),
                    yaxis_title=y_axis.replace('_', ' ').title(),
                    zaxis_title=z_axis.replace('_', ' ').title(),
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            )
            
            # Update marker properties
            plot_fig.update_traces(
                marker=dict(
                    size=marker_size,
                    opacity=opacity
                )
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating 3D scatter plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze 3D scatter plot with user-selected features."""
        
        numerical_cols = data.columns
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                'Select X-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select X-Axis value ]",
                key="scatter3d_x_axis"
            )
        
        with col2:
            y_axis = st.selectbox(
                'Select Y-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Y-Axis value ]",
                key="scatter3d_y_axis"
            )
            
        with col3:
            z_axis = st.selectbox(
                'Select Z-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Z-Axis value ]",
                key="scatter3d_z_axis"
            )

        if all([x_axis, y_axis, z_axis]):
            # Color grouping selection
            color = st.selectbox(
                "Color points by feature (optional)",
                options=[None] + list(data.columns),
                index=0,
                key="scatter3d_color"
            )

            # Plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=self.COLOR_SEQUENCES,
                        index=0,
                        key="scatter3d_colorscheme"
                    )
                
                with col2:
                    marker_size = st.slider(
                        "Marker Size",
                        min_value=3,
                        max_value=15,
                        value=5,
                        key="scatter3d_markersize"
                    )
                
                with col3:
                    opacity = st.slider(
                        "Point Opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        key="scatter3d_opacity"
                    )

            plot_fig = self.create_3d_scatter(
                data,
                x_axis,
                y_axis,
                z_axis,
                color=color,
                color_scheme=color_scheme,
                marker_size=marker_size,
                opacity=opacity
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select X, Y and Z axes for the 3D Scatter plot.")

class Line3DAnalyzer(PlotAnalyzer):
    """Class to handle 3D line plot generation and analysis."""
    
    # List of verified working color sequences
    COLOR_SEQUENCES = [
        'Jet',
        'Blugrn',
        'Teal',
        'Aggrnyl',    
        'Bluered',    
        'Blackbody',  
        'Rainbow',    
        'Sunset'
    ]
    
    def create_3d_line(
        self, 
        data, 
        x_axis: str, 
        y_axis: str,
        z_axis: str,
        color: Optional[str] = None,
        color_scheme: str = 'Teal',
        line_width: int = 3,
        markers: bool = True,
        marker_size: int = 4
    ) -> Optional[px.line_3d]:
        """
        Create a 3D line plot with the specified parameters.
        """
        try:
            plot_fig = px.line_3d(
                data,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color=color,
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                title=f"3D Line Plot: {x_axis} vs {y_axis} vs {z_axis}",
                markers=markers
            )
            
            # Update layout
            plot_fig.update_layout(
                height=700,  # Larger height for better 3D visualization
                showlegend=True if color else False,
                title_x=0.5,
                scene=dict(
                    xaxis_title=x_axis.replace('_', ' ').title(),
                    yaxis_title=y_axis.replace('_', ' ').title(),
                    zaxis_title=z_axis.replace('_', ' ').title(),
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    )
                )
            )
            
            # Update line and marker properties
            plot_fig.update_traces(
                line=dict(width=line_width),
                marker=dict(size=marker_size)
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating 3D line plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze 3D line plot with user-selected features."""
        
        numerical_cols = data.columns
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox(
                'Select X-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select X-Axis value ]",
                key="line3d_x_axis"
            )
        
        with col2:
            y_axis = st.selectbox(
                'Select Y-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Y-Axis value ]",
                key="line3d_y_axis"
            )
            
        with col3:
            z_axis = st.selectbox(
                'Select Z-Axis',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Z-Axis value ]",
                key="line3d_z_axis"
            )

        if all([x_axis, y_axis, z_axis]):
            # Color grouping selection
            color = st.selectbox(
                "Color lines by feature (optional)",
                options=[None] + list(data.columns),
                index=0,
                key="line3d_color"
            )

            # Plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=self.COLOR_SEQUENCES,
                        index=0,
                        key="line3d_colorscheme"
                    )
                    
                    show_markers = st.checkbox(
                        "Show Markers",
                        value=True,
                        key="line3d_markers"
                    )
                
                with col2:
                    line_width = st.slider(
                        "Line Width",
                        min_value=1,
                        max_value=10,
                        value=3,
                        key="line3d_linewidth"
                    )
                    
                    if show_markers:
                        marker_size = st.slider(
                            "Marker Size",
                            min_value=2,
                            max_value=10,
                            value=4,
                            key="line3d_markersize"
                        )
                    else:
                        marker_size = 0

            plot_fig = self.create_3d_line(
                data,
                x_axis,
                y_axis,
                z_axis,
                color=color,
                color_scheme=color_scheme,
                line_width=line_width,
                markers=show_markers,
                marker_size=marker_size
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select X, Y and Z axes for the 3D Line plot.")

class PolarScatterAnalyzer(PlotAnalyzer):
    """Class to handle polar scatter plot generation and analysis."""
    
    # List of verified working color sequences
    COLOR_SEQUENCES = [
        'Jet',
        'Blugrn',
        'Teal',
        'Aggrnyl',    
        'Bluered',    
        'Blackbody',  
        'Rainbow',    
        'Sunset'
    ]

    # Available marker symbols
    MARKER_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 
                     'triangle-down', 'pentagon', 'hexagon', 'star']
    
    def create_polar_scatter(
        self, 
        data, 
        r_axis: str,  # Radial axis
        theta_axis: str,  # Angular axis
        color: Optional[str] = None,
        symbol: Optional[str] = None,
        color_scheme: str = 'Teal',
        marker_size: int = 8,
        opacity: float = 0.7
    ) -> Optional[px.scatter_polar]:
        """
        Create a polar scatter plot with the specified parameters.
        """
        try:
            plot_fig = px.scatter_polar(
                data,
                r=r_axis,
                theta=theta_axis,
                color=color,
                symbol=symbol,
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                title=f"Polar Scatter Plot: {r_axis} vs {theta_axis}",
                symbol_sequence=self.MARKER_SYMBOLS
            )
            
            # Update layout
            plot_fig.update_layout(
                height=600,
                showlegend=True if (color or symbol) else False,
                title_x=0.5,
                polar=dict(
                    radialaxis=dict(
                        tickprefix=r_axis.replace('_', ' ').title() + ": ",
                        gridcolor='lightgrey'
                    ),
                    angularaxis=dict(
                        tickprefix=theta_axis.replace('_', ' ').title() + ": ",
                        gridcolor='lightgrey'
                    )
                )
            )
            
            # Update marker properties
            plot_fig.update_traces(
                marker=dict(
                    size=marker_size,
                    opacity=opacity
                )
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating polar scatter plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze polar scatter plot with user-selected features."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            r_axis = st.selectbox(
                'Select Radial Axis (Distance from center)',
                options=data.columns,
                index=None,
                placeholder="[ Select Radial value ]",
                key="polar_r_axis"
            )
        
        with col2:
            theta_axis = st.selectbox(
                'Select Angular Axis (Angle)',
                options=data.columns,
                index=None,
                placeholder="[ Select Angular value ]",
                key="polar_theta_axis"
            )

        if r_axis and theta_axis:
            col1, col2 = st.columns(2)
            
            with col1:
                color = st.selectbox(
                    "Color by feature (optional)",
                    options=[None] + list(data.columns),
                    index=0,
                    key="polar_color"
                )
            
            with col2:
                symbol = st.selectbox(
                    "Shape by feature (optional)",
                    options=[None] + list(data.columns),
                    index=0,
                    key="polar_symbol"
                )

            # Plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=self.COLOR_SEQUENCES,
                        index=0,
                        key="polar_colorscheme"
                    )
                
                with col2:
                    marker_size = st.slider(
                        "Marker Size",
                        min_value=3,
                        max_value=20,
                        value=8,
                        key="polar_markersize"
                    )
                
                with col3:
                    opacity = st.slider(
                        "Point Opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        key="polar_opacity"
                    )

            plot_fig = self.create_polar_scatter(
                data,
                r_axis,
                theta_axis,
                color=color,
                symbol=symbol,
                color_scheme=color_scheme,
                marker_size=marker_size,
                opacity=opacity
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select both Radial and Angular axes for the Polar Scatter plot.")

class PolarBarAnalyzer(PlotAnalyzer):
    """Class to handle polar bar plot generation and analysis."""
    
    # List of verified working color sequences
    COLOR_SEQUENCES = [
        'Jet',
        'Blugrn',
        'Teal',
        'Aggrnyl',    
        'Bluered',    
        'Blackbody',  
        'Rainbow',    
        'Sunset'
    ]
    
    def create_polar_bar(
        self, 
        data, 
        r_axis: str,  # Radial axis (bar length)
        theta_axis: str,  # Angular axis
        color: Optional[str] = None,
        color_scheme: str = 'Teal',
        opacity: float = 0.8,
        bar_mode: str = 'relative'
    ) -> Optional[px.bar_polar]:
        """
        Create a polar bar plot with the specified parameters.
        """
        try:
            plot_fig = px.bar_polar(
                data,
                r=r_axis,
                theta=theta_axis,
                color=color,
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                title=f"Polar Bar Plot: {r_axis} vs {theta_axis}",
                barmode=bar_mode
            )
            
            # Update layout
            plot_fig.update_layout(
                height=600,
                showlegend=True if color else False,
                title_x=0.5,
                polar=dict(
                    radialaxis=dict(
                        tickprefix=r_axis.replace('_', ' ').title() + ": ",
                        gridcolor='lightgrey'
                    ),
                    angularaxis=dict(
                        tickprefix=theta_axis.replace('_', ' ').title() + ": ",
                        gridcolor='lightgrey'
                    )
                )
            )
            
            # Update trace properties
            plot_fig.update_traces(opacity=opacity)
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating polar bar plot: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze polar bar plot with user-selected features."""
        
        col1, col2 = st.columns(2)
        
        with col1:
            r_axis = st.selectbox(
                'Select Radial Axis (Bar Length)',
                options=data.columns,
                index=None,
                placeholder="[ Select Radial value ]",
                key="polar_bar_r_axis"
            )
        
        with col2:
            theta_axis = st.selectbox(
                'Select Angular Axis (Categories)',
                options=data.columns,
                index=None,
                placeholder="[ Select Angular value ]",
                key="polar_bar_theta_axis"
            )

        if r_axis and theta_axis:
            # Color selection
            color = st.selectbox(
                "Color bars by feature (optional)",
                options=[None] + list(data.columns),
                index=0,
                key="polar_bar_color"
            )

            # Plot options
            with st.expander("Plot Options", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=self.COLOR_SEQUENCES,
                        index=0,
                        key="polar_bar_colorscheme"
                    )
                
                with col2:
                    bar_mode = st.selectbox(
                        "Bar Mode",
                        options=['relative', 'group', 'overlay'],
                        index=0,
                        key="polar_bar_mode"
                    )
                
                with col3:
                    opacity = st.slider(
                        "Bar Opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.8,
                        step=0.1,
                        key="polar_bar_opacity"
                    )

            plot_fig = self.create_polar_bar(
                data,
                r_axis,
                theta_axis,
                color=color,
                color_scheme=color_scheme,
                opacity=opacity,
                bar_mode=bar_mode
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select both Radial and Angular axes for the Polar Bar plot.")

class TileMapAnalyzer(PlotAnalyzer):
    """Class to handle tile map generation and analysis."""
    
    # List of verified working color sequences
    COLOR_SEQUENCES = [     
        'RdBu',     
        'Viridis',  
        'Plasma',   
        'Hot',      
        'Rainbow',  
        'Sunset'   
    ]
    
    # Map styles available in Mapbox
    MAP_STYLES = [
        'open-street-map',
        'carto-positron',
        'carto-darkmatter',
        'stamen-terrain',
        'stamen-toner'
    ]
    
    def create_tile_map(
        self, 
        data, 
        lat: str,
        lon: str,
        hover_data: List[str],
        color: Optional[str] = None,
        size: Optional[str] = None,
        color_scheme: str = 'Viridis',
        map_style: str = 'open-street-map',
        zoom: float = 1.0
    ) -> Optional[px.scatter_mapbox]:
        """
        Create a tile map with the specified parameters.
        """
        try:
            # Create the map
            plot_fig = px.scatter_mapbox(
                data,
                lat=lat,
                lon=lon,
                hover_data=hover_data,
                color=color,
                size=size,
                color_discrete_sequence=getattr(px.colors.sequential, color_scheme),
                title=f"Geographic Distribution"
            )
            
            # Update layout
            plot_fig.update_layout(
                mapbox=dict(
                    style=map_style,
                    zoom=zoom,
                    center=dict(
                        lat=data[lat].mean(),
                        lon=data[lon].mean()
                    )
                ),
                height=600,
                showlegend=True if color else False,
                title_x=0.5,
                margin=dict(r=0, t=30, l=0, b=0)
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating tile map: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display and analyze tile map with user-selected features."""
        
        col1, col2 = st.columns(2)
        
        numerical_cols = data.columns
        
        with col1:
            lat = st.selectbox(
                'Select Latitude',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Latitude ]",
                key="tilemap_lat"
            )
        
        with col2:
            lon = st.selectbox(
                'Select Longitude',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Longitude ]",
                key="tilemap_lon"
            )

        if lat and lon:
            # Select hover data
            hover_data = st.multiselect(
                "Select data to show on hover",
                options=data.columns,
                default=[],
                key="tilemap_hover"
            )

            # Plot options
            with st.expander("Map Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    color = st.selectbox(
                        "Color points by",
                        options=[None] + list(data.columns),
                        index=0,
                        key="tilemap_color"
                    )
                    
                    size = st.selectbox(
                        "Size points by",
                        options=[None] + list(numerical_cols),
                        index=0,
                        key="tilemap_size"
                    )
                
                with col2:
                    map_style = st.selectbox(
                        "Map Style",
                        options=self.MAP_STYLES,
                        index=0,
                        key="tilemap_style"
                    )
                    
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=self.COLOR_SEQUENCES,
                        index=0,
                        key="tilemap_colorscheme"
                    )
                
                zoom = st.slider(
                    "Zoom Level",
                    min_value=0.0,
                    max_value=20.0,
                    value=1.0,
                    step=0.5,
                    key="tilemap_zoom"
                )

            plot_fig = self.create_tile_map(
                data,
                lat,
                lon,
                hover_data,
                color=color,
                size=size,
                color_scheme=color_scheme,
                map_style=map_style,
                zoom=zoom
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
                analysis = self.analyze_plot(plot_fig)
                if analysis:
                    st.markdown(analysis)
        else:
            st.warning("Please select both Latitude and Longitude for the map.")

class ChoroplethMapAnalyzer:
    """Class to handle choropleth map generation."""
    
    # List of verified working color sequences
    COLOR_SEQUENCES = [
        'Viridis',     # Default
        'RdBu',        # Red-Blue
        'RdYlBu',      # Red-Yellow-Blue
        'YlOrRd',      # Yellow-Orange-Red
        'YlGnBu',      # Yellow-Green-Blue
        'Plasma',      # Plasma
        'Inferno',     # Inferno
        'Magma',       # Magma
        'Cividis'      # Cividis
    ]
    
    # Available map scopes
    MAP_SCOPES = {
        'USA': 'usa',
        'North America': 'north america',
        'South America': 'south america',
        'Europe': 'europe',
        'Asia': 'asia',
        'Africa': 'africa',
        'World': 'world'
    }
    
    def __init__(self):
        # Load US counties GeoJSON data
        with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
            self.counties_geojson = json.load(response)
    
    def create_choropleth_map(
        self, 
        data, 
        locations: str,
        color: str,
        scope: str = 'usa',
        color_scheme: str = 'Viridis',
        hover_data: Optional[List[str]] = None,
        range_color: Optional[tuple[float, float]] = None
    ) -> Optional[px.choropleth]:
        """
        Create a choropleth map with the specified parameters.
        """
        try:
            # Calculate default range if not provided
            if range_color is None:
                min_val = data[color].min()
                max_val = data[color].max()
                range_color = (min_val, max_val)
            
            # Create the map
            plot_fig = px.choropleth(
                data,
                geojson=self.counties_geojson,
                locations=locations,
                color=color,
                color_continuous_scale=color_scheme,
                scope=scope,
                hover_data=hover_data,
                range_color=range_color,
                labels={color: color.replace('_', ' ').title()}
            )
            
            # Update layout
            plot_fig.update_layout(
                height=600,
                title=f"Geographic Distribution of {color.replace('_', ' ').title()}",
                title_x=0.5,
                margin=dict(r=0, t=30, l=0, b=0)
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating choropleth map: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display choropleth map with user-selected features."""
        
        st.markdown("ℹ️ Best suited for US data with FIPS codes.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            locations = st.selectbox(
                'Select FIPS Code Column',
                options=data.columns,
                index=None,
                placeholder="[ FIPS Code ]",
                key="choropleth_locations"
            )
        
        with col2:
            color = st.selectbox(
                'Select Value to Display',
                options=data.columns,
                index=None,
                placeholder="[ Select value to visualize ]",
                key="choropleth_color"
            )

        if locations and color:
            # Plot options
            with st.expander("Map Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=self.COLOR_SEQUENCES,
                        index=0,
                        key="choropleth_colorscheme"
                    )
                    
                    scope = st.selectbox(
                        "Map Scope",
                        options=list(self.MAP_SCOPES.keys()),
                        index=0,
                        key="choropleth_scope"
                    )
                
                with col2:
                    hover_data = st.multiselect(
                        "Additional Hover Data",
                        options=[col for col in data.columns if col not in [locations, color]],
                        key="choropleth_hover"
                    )
                    
                    use_custom_range = st.checkbox(
                        "Custom Color Range",
                        value=False,
                        key="choropleth_custom_range"
                    )
                
                if use_custom_range:
                    min_val = float(data[color].min())
                    max_val = float(data[color].max())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        range_min = st.number_input(
                            "Minimum Value",
                            value=min_val,
                            key="choropleth_range_min"
                        )
                    with col2:
                        range_max = st.number_input(
                            "Maximum Value",
                            value=max_val,
                            key="choropleth_range_max"
                        )
                    range_color = (range_min, range_max)
                else:
                    range_color = None

            plot_fig = self.create_choropleth_map(
                data,
                locations,
                color,
                scope=self.MAP_SCOPES[scope],
                color_scheme=color_scheme,
                hover_data=hover_data if hover_data else None,
                range_color=range_color
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
        else:
            st.warning("Please select both FIPS code column and value to display on the map.")

class BubbleMapAnalyzer:
    """Class to handle bubble map generation."""
    
    # Available map projections
    MAP_PROJECTIONS = [
        'natural earth',    
        'equirectangular',  
        'mercator',         
        'orthographic',     
        'kavrayskiy7'       
    ]
    
    # color sequences
    COLOR_SEQUENCES = [
        'Viridis',     
        'RdBu',        
        'RdYlBu',      
        'YlOrRd',      
        'YlGnBu',      
        'Plasma',      
        'Inferno',     
        'Magma',       
        'Cividis'      
    ]
    
    def create_bubble_map(
        self,
        data,
        lat: str,
        lon: str,
        size: str,
        color: Optional[str] = None,
        hover_data: Optional[List[str]] = None,
        projection: str = 'natural earth',
        color_scheme: str = 'Viridis',
        size_max: int = 50,
        opacity: float = 0.7
    ) -> Optional[go.Figure]:
        """
        Create a bubble map with the specified parameters.
        """
        try:
            # Create the bubble map
            plot_fig = px.scatter_geo(
                data,
                lat=lat,
                lon=lon,
                size=size,
                color=color,
                color_continuous_scale=color_scheme if color else None,
                hover_name=color,
                hover_data=hover_data,
                projection=projection,
                size_max=size_max,
                template="plotly",
                title=f"Geographic Distribution of {size}"
            )
            
            # Update layout
            plot_fig.update_layout(
                height=600,
                title_x=0.5,
                margin=dict(r=0, t=30, l=0, b=0),
                showlegend=True if color else False
            )
            
            # Update marker properties
            plot_fig.update_traces(
                marker=dict(opacity=opacity)
            )
            
            return plot_fig
            
        except Exception as e:
            st.error(f"Error creating bubble map: {str(e)}")
            return None

    def display_plot(self, data) -> None:
        """Display bubble map with user-selected features."""
        
        numerical_cols = data.columns
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lat = st.selectbox(
                'Select Latitude Column',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Latitude ]",
                key="bubble_lat"
            )
        
        with col2:
            lon = st.selectbox(
                'Select Longitude Column',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Longitude ]",
                key="bubble_lon"
            )
            
        with col3:
            size = st.selectbox(
                'Select Size Column',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Size ]",
                key="bubble_size"
            )

        if all([lat, lon, size]):
            # Color selection
            color = st.selectbox(
                'Color bubbles by (optional)',
                options=[None] + list(data.columns),
                index=0,
                key="bubble_color"
            )

            # Plot options
            with st.expander("Map Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    projection = st.selectbox(
                        "Map Projection",
                        options=self.MAP_PROJECTIONS,
                        index=0,
                        key="bubble_projection"
                    )
                    
                    size_max = st.slider(
                        "Maximum Bubble Size",
                        min_value=10,
                        max_value=100,
                        value=50,
                        key="bubble_size_max"
                    )
                
                with col2:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=self.COLOR_SEQUENCES,
                        index=0,
                        key="bubble_colorscheme"
                    )
                    
                    opacity = st.slider(
                        "Bubble Opacity",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.7,
                        step=0.1,
                        key="bubble_opacity"
                    )
                
                hover_data = st.multiselect(
                    "Additional Hover Data",
                    options=[col for col in data.columns if col not in [lat, lon, size, color]],
                    key="bubble_hover"
                )

            plot_fig = self.create_bubble_map(
                data,
                lat,
                lon,
                size,
                color=color,
                hover_data=hover_data if hover_data else None,
                projection=projection,
                color_scheme=color_scheme,
                size_max=size_max,
                opacity=opacity
            )
            
            if plot_fig:
                st.plotly_chart(plot_fig, use_container_width=True)
        else:
            st.warning("Please select Latitude, Longitude, and Size columns to create the bubble map.")

class StMapAnalyzer:
    """Class to handle Streamlit map generation."""
    
    def create_st_map(
        self,
        data,
        latitude: str,
        longitude: str,
        size: int = 40,
        zoom: int = 2
    ) -> None:
        """
        Create a Streamlit map with scatter plot overlay.
        
        Args:
            data: DataFrame containing the data
            latitude: Column name for latitude
            longitude: Column name for longitude
            size: Size of the scatter points
            zoom: Initial zoom level of the map
        """
        try:
            # Create a copy of the data with float64 type for coordinates
            map_data = data.to_pandas().copy()
            map_data[latitude] = map_data[latitude].astype('float64')
            map_data[longitude] = map_data[longitude].astype('float64')
            
            # Check if coordinates are within valid ranges
            if (map_data[latitude].min() < -90 or map_data[latitude].max() > 90 or
                map_data[longitude].min() < -180 or map_data[longitude].max() > 180):
                st.warning("Some coordinates appear to be outside valid ranges. Map may not display correctly.")
            
            # Create the map
            st.map(
                data=map_data,
                latitude=latitude,
                longitude=longitude,
                size=size,
                zoom=zoom
            )
            
        except Exception as e:
            st.error(f"Error creating map: {str(e)}")

    def display_plot(self, data) -> None:
        """Display Streamlit map with user-selected features."""
        
        # Get numerical columns for coordinates
        numerical_cols = data.columns
        
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.selectbox(
                'Select Latitude',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Latitude ]",
                key="stmap_lat"
            )
        
        with col2:
            longitude = st.selectbox(
                'Select Longitude',
                options=numerical_cols,
                index=None,
                placeholder="[ Select Longitude ]",
                key="stmap_lon"
            )

        if latitude and longitude:
            # Map options
            with st.expander("Map Options", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    point_size = st.slider(
                        "Point Size",
                        min_value=1,
                        max_value=100,
                        value=40,
                        key="stmap_size"
                    )
                
                with col2:
                    zoom_level = st.slider(
                        "Zoom Level",
                        min_value=0,
                        max_value=20,
                        value=2,
                        key="stmap_zoom"
                    )

            self.create_st_map(
                data,
                latitude,
                longitude,
                size=point_size,
                zoom=zoom_level
            )
        else:
            st.warning("Please select both Latitude and Longitude columns to display the map.")

# Plotting Functions

def plot_histogram(dataframe) -> None:
    """Function to create and display histograms."""
    st.markdown('### Histogram')
    st.markdown("""
        A histogram is a graphical representation of data distribution, using bars to display 
        the frequency or count of values within specified intervals or bins. It offers a visual 
        overview of data patterns, highlighting peaks and variations, enabling quick insights 
        into the dataset's shape and characteristics.
    """)

    analyzer = HistogramAnalyzer(HistogramConfig())
    
    tab1, tab2 = st.tabs(["Single Feature", "All Features"])
    
    with tab1:
        feature = st.selectbox(
            'Choose Feature',
            options=dataframe.columns,
            index=None,
            placeholder="[ Select feature to analyze ]",
            key="single_feature_select"  # Added unique key
        )
        if feature:
            analyzer.display_single_plot(dataframe, feature, key_prefix="single")
    
    with tab2:
        if st.toggle('Show All Features', key="show_all_toggle"):  # Added unique key
            col1, col2 = st.columns(2)
            with col1:
                n_cols = st.number_input(
                    "Plots per row",
                    min_value=1,
                    max_value=4,
                    value=2,
                    key="n_cols_input"  # Added unique key
                )
            with col2:
                global_bins = st.number_input(
                    "Bins for all plots",
                    min_value=5,
                    max_value=100,
                    value=30,
                    key="global_bins_input"  # Added unique key
                )
            
            # Create grid layout for multiple plots
            for i in range(0, len(dataframe.columns), n_cols):
                cols = st.columns(n_cols)
                for j, col in enumerate(cols):
                    if i + j < len(dataframe.columns):
                        feature = dataframe.columns[i + j]
                        with col:
                            # Add row and column indices to make keys unique
                            key_prefix = f"grid_row{i}_col{j}"
                            analyzer.display_single_plot(
                                dataframe,
                                feature,
                                bins=global_bins,
                                key_prefix=key_prefix
                            )

def plot_boxplot(dataframe) -> None:
    """Function to create and display box plots."""
    st.markdown('### Box Plot')
    st.markdown('''
        A Box Plot, or box-and-whisker plot, is a graphical representation that displays 
        the distribution of a dataset, providing insights into its central tendency, 
        spread, and presence of outliers.
    ''')

    analyzer = BoxPlotAnalyzer(BoxPlotConfig())
    
    if st.toggle('Plot for all Features'):
        n_cols = 2
        for i in range(0, len(dataframe.columns), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                if i + j < len(dataframe.columns):
                    feature = dataframe.columns[i + j]
                    with col:
                        analyzer.display_single_plot(dataframe, feature)
    else:
        feature = st.selectbox(
            'Choose Feature',
            options=dataframe.columns,
            index=None,
            placeholder="[ Select feature to analyze ]"
        )
        if feature:
            analyzer.display_single_plot(dataframe, feature)

def plot_scatter(dataframe) -> None:
    """Function to create and display scatter plots."""
    st.markdown('### Scatter Plot')
    st.markdown('''
        A Scatter Plot is a visual representation of data points on a graph, showcasing 
        the relationship between two variables and revealing patterns or trends in a 
        concise and easily interpretable manner.
    ''')

    analyzer = ScatterPlotAnalyzer(ScatterPlotConfig())
    analyzer.display_plot(dataframe)

def plot_bar(dataframe) -> None:
    """Function to create and display bar plots."""
    st.markdown('### Bar Plot')
    st.markdown("""
        A Bar Plot is a visual representation of data using rectangular bars of varying 
        heights to illustrate the values of different categories or groups. It is a 
        widely-used tool for comparing data across categories and displaying patterns 
        or variations in a clear and straightforward manner.
    """)

    analyzer = BarPlotAnalyzer(BarPlotConfig())
    analyzer.display_plot(dataframe)

def plot_contour(dataframe) -> None:
    """Function to create and display contour plots."""
    st.markdown('### Contour Plot')
    st.markdown("""
        A Contour Plot is a visual representation that displays three-dimensional data 
        on a two-dimensional surface using contour lines to depict changes in a third 
        variable. It's a valuable tool for visualizing complex relationships and surfaces, 
        commonly used in scientific and engineering fields to illustrate topography, 
        temperature, and more.
    """)

    analyzer = ContourPlotAnalyzer(ContourPlotConfig())
    analyzer.display_plot(dataframe)

def plot_hist_contour(dataframe) -> None:
    """Function to create and display 2D histogram contour plots."""
    st.markdown('### 2D Histogram Contour')
    st.markdown("""
        A 2D Histogram Contour is a graphical representation that combines a two-dimensional 
        histogram with contour lines to visualize the distribution and density of data points 
        in a scatter plot. It provides a clear depiction of data concentration, revealing 
        areas of high and low density in the plot, making it useful for data analysis and 
        pattern recognition.
    """)

    analyzer = HistContourPlotAnalyzer(HistContourPlotConfig())
    analyzer.display_plot(dataframe)

def plot_pie(dataframe) -> None:
    """Function to create and display pie charts."""
    st.markdown('### Pie Chart')
    st.markdown("""
        A Pie Chart is a circular graph that divides data into slices or wedges, where 
        each slice represents a proportion or percentage of the whole. It is a visually 
        intuitive way to display categorical data and illustrate the composition or 
        distribution of different categories within a dataset.
    """)

    analyzer = PieChartAnalyzer(PieChartConfig())
    analyzer.display_plot(dataframe)

def plot_violin(dataframe) -> None:
    """Function to create and display violin plots."""
    st.markdown('### Violin Plot')
    st.markdown("""
        A Violin Plot is a data visualization that combines elements of a box plot and 
        a kernel density plot to display the distribution and summary statistics of a 
        dataset. It provides a detailed view of data distribution, revealing both central 
        tendency and the probability density of different values, making it useful for 
        comparing multiple categories or groups.
    """)

    analyzer = ViolinPlotAnalyzer(ViolinPlotConfig())
    analyzer.display_plot(dataframe)

def plot_line(dataframe) -> None:
    """Function to create and display line plots."""
    st.markdown('### Line Plot')
    st.markdown("""
        A Line Plot, also known as a Line Chart, is a graph that uses lines to connect 
        data points, typically over time. It's a powerful tool for visualizing trends 
        and patterns in data, making it easy to identify fluctuations and changes in 
        values. Line plots are widely used in various fields, including finance, science, 
        and data analysis, to track and illustrate changes in a variable.
    """)

    analyzer = LinePlotAnalyzer(LinePlotConfig())
    analyzer.display_plot(dataframe)

def plot_scatter_3d(dataframe) -> None:
    """Function to create and display 3D scatter plots."""
    st.markdown('### 3D Scatter Plot')
    st.markdown("""
        A 3D Scatter Plot is a three-dimensional representation of data points, where 
        each point is defined by three variables. It allows for the visualization of 
        complex relationships between multiple variables in a 3D space. This type of 
        plot is particularly useful for exploring and understanding data that involves 
        three key factors or dimensions.
    """)

    analyzer = Scatter3DAnalyzer(Scatter3DConfig())
    analyzer.display_plot(dataframe)

def plot_line_3d(dataframe) -> None:
    """Function to create and display 3D line plots."""
    st.markdown('### 3D Line Plot')
    st.markdown("""
        A 3D Line Plot is a three-dimensional representation of data that uses lines to 
        connect data points in a 3D space. It's a valuable tool for visualizing and 
        understanding data with three dimensions, such as time series data or spatial 
        data. 3D line plots enable the exploration of trends and patterns across three 
        variables, providing a comprehensive view of data relationships.
    """)

    analyzer = Line3DAnalyzer(Line3DConfig())
    analyzer.display_plot(dataframe)

def plot_polar_scatter(dataframe) -> None:
    """Function to create and display polar scatter plots."""
    st.markdown('### Polar Scatter Plot')
    st.markdown("""
        A Polar Scatter Plot is a data visualization that represents data points in a 
        polar coordinate system. It's particularly useful for displaying data with 
        angular and radial components, such as geographic data or circular patterns. 
        In a polar scatter plot, each point is positioned according to an angle and 
        distance from the center, making it easy to identify patterns and relationships 
        within the data, especially when working with circular or directional data.
    """)

    analyzer = PolarScatterAnalyzer(PolarScatterConfig())
    analyzer.display_plot(dataframe)

def plot_polar_bar(dataframe) -> None:
    """Function to create and display polar bar plots."""
    st.markdown('### Polar Bar Plot')
    st.markdown("""
        A Polar Bar Plot, also known as a Radial Bar Chart, is a unique data visualization 
        that displays data using bars arranged in a circular or radial pattern. It is 
        particularly useful for showcasing data with a directional or cyclical nature, 
        such as time series data with periodic patterns. Each bar extends from the center 
        outward, representing data values at specific angles. This type of plot is ideal 
        for visualizing and comparing data across categories or groups with an inherent 
        circular relationship.
    """)

    analyzer = PolarBarAnalyzer(PolarBarConfig())
    analyzer.display_plot(dataframe)

def plot_tile_map(dataframe) -> None:
    """Function to create and display tile maps."""
    st.markdown('### Tile Map')
    st.markdown("""
        A Tile Map is a data visualization technique that uses small, uniformly-sized 
        squares (tiles) to represent data values within a grid or map. Each tile's color 
        or shading corresponds to a specific data category or variable, allowing for the 
        visualization of spatial data patterns. Tile maps are particularly useful for 
        showcasing geographic and spatial data, making it easy to identify clusters, 
        trends, and variations across regions or areas.
    """)

    analyzer = TileMapAnalyzer(TileMapConfig())
    analyzer.display_plot(dataframe)

def plot_choropleth_map(dataframe) -> None:
    """Function to create and display choropleth maps."""
    st.markdown('### Choropleth Map')
    st.markdown("""
        A Choropleth Map is a thematic map that represents data by shading or coloring 
        geographic regions, such as countries, states, or districts, based on the 
        intensity or magnitude of a particular variable. The varying colors or patterns 
        across the map's regions provide a visual representation of data distribution, 
        making it a powerful tool for displaying spatial data and highlighting 
        geographical patterns or disparities in a clear and easily interpretable manner.
    """)

    analyzer = ChoroplethMapAnalyzer()
    analyzer.display_plot(dataframe)

def plot_bubble_map(dataframe) -> None:
    """Function to create and display bubble maps."""
    st.markdown('### Bubble Map')
    st.markdown("""
        A Bubble Map visualizes data points on a geographic map where each point is 
        represented by a bubble. The size and color of bubbles can represent different 
        metrics, making it an effective way to display multiple dimensions of data 
        across geographical locations.
    """)

    analyzer = BubbleMapAnalyzer()
    analyzer.display_plot(dataframe)

def plot_st_map(dataframe) -> None:
    """Function to create and display Streamlit maps with scatter plots."""
    st.markdown('### Map with Scatter Plot Overlay')
    st.markdown("""
        Create scatter plot charts on top of a map, with auto-centering and auto-zoom. 
        This visualization is perfect for displaying geographic point data with 
        automatic handling of map boundaries and zoom levels based on your data points.
    """)

    analyzer = StMapAnalyzer()
    analyzer.display_plot(dataframe)



