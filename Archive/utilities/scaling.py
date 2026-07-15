from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utilities.select_data_types import SelectDataTypes
import numpy as np
import streamlit as st

#---

# Modified scaling functions
def MinMaxScaling(dataframe):
    for column in dataframe.columns:
        min_max_scaler = MinMaxScaler()
        dataframe[[column]] = min_max_scaler.fit_transform(dataframe[[column]])
    return dataframe

def StandardScaling(dataframe):
    for column in dataframe.columns:
        standard_scaler = StandardScaler()
        dataframe[[column]] = standard_scaler.fit_transform(dataframe[[column]])
    return dataframe

def RobustScaling(dataframe):
    for column in dataframe.columns:
        robust_scaler = RobustScaler()
        dataframe[[column]] = robust_scaler.fit_transform(dataframe[[column]])
    return dataframe

def MaxAbsScaling(dataframe):
    for column in dataframe.columns:
        max_abs_scaler = MaxAbsScaler()
        dataframe[[column]] = max_abs_scaler.fit_transform(dataframe[[column]])
    return dataframe

def QuantileTransformerScaling(dataframe):
    for column in dataframe.columns:
        quantile_transformer_scaler = QuantileTransformer()
        dataframe[[column]] = quantile_transformer_scaler.fit_transform(dataframe[[column]])
    return dataframe

def LogTransformer(dataframe):
    for column in dataframe.columns:
        dataframe[column] = np.log1p(dataframe[column])
    return dataframe

def PowerTransformerBoxCox(dataframe):
    for column in dataframe.columns:
        power_transformer = PowerTransformer(method='box-cox')
        dataframe[[column]] = power_transformer.fit_transform(dataframe[[column]])
    return dataframe

def PowerTransformerYeoJohnson(dataframe):
    for column in dataframe.columns:
        power_transformer = PowerTransformer(method='yeo-johnson')
        dataframe[[column]] = power_transformer.fit_transform(dataframe[[column]])
    return dataframe

#---


# # Min-Max Scaling
# def MinMaxScaling(dataframe):
#    dataframe = SelectDataTypes(dataframe)
#    for column in dataframe.columns:
#        min_max_scaler = MinMaxScaler()
#        dataframe[[column]] = min_max_scaler.fit_transform(dataframe[[column]])
#    return dataframe

# # Standard Scaling (Z-Normalization)
# def StandardScaling(dataframe):
#    dataframe = SelectDataTypes(dataframe) 
#    for column in dataframe.columns:
#        standard_scaler = StandardScaler()
#        dataframe[[column]] = standard_scaler.fit_transform(dataframe[[column]])
#    return dataframe

# # Robust Scaling
# def RobustScaling(dataframe):
#     dataframe = SelectDataTypes(dataframe) 
#     for column in dataframe.columns:
#         robust_scaler = RobustScaler()
#         dataframe[[column]] = robust_scaler.fit_transform(dataframe[[column]])
#     return dataframe

# # Max Abs Scaling
# def MaxAbsScaling(dataframe):
#     dataframe = SelectDataTypes(dataframe) 
#     for column in dataframe.columns:
#         max_abs_scaler = MaxAbsScaler()
#         dataframe[[column]] = max_abs_scaler.fit_transform(dataframe[[column]])
#     return dataframe

# # Quantile Transformer Scaling
# def QuantileTransformerScaling(dataframe):
#     dataframe = SelectDataTypes(dataframe) 
#     for column in dataframe.columns:
#         quantile_transformer_scaler = QuantileTransformer()
#         dataframe[[column]] = quantile_transformer_scaler.fit_transform(dataframe[[column]])
#     return dataframe

# # Quantile Transformer Scaling
# def LogTransformer(dataframe):
#     dataframe = SelectDataTypes(dataframe) 
#     for column in dataframe.columns:
#         dataframe[column] = np.log1p(dataframe[column])
#     return dataframe

# # Power Transformer: Box-Cox transform
# def PowerTransformerBoxCox(dataframe):
#     dataframe = SelectDataTypes(dataframe) 
#     for column in dataframe.columns:
#         power_transformer = PowerTransformer(method='box-cox')
#         dataframe[[column]] = power_transformer.fit_transform(dataframe[[column]])
#     return dataframe

# # Power Transformer: Yeo-Johnson transform
# def PowerTransformerYeoJohnson(dataframe):
#     dataframe = SelectDataTypes(dataframe) 
#     for column in dataframe.columns:
#         power_transformer = PowerTransformer(method='yeo-johnson')
#         dataframe[[column]] = power_transformer.fit_transform(dataframe[[column]])
#     return dataframe

# Comparison Plot
def CompareHistograms(df_original, df_scaled):
    for column in df_original.columns:
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Original: {column}", f"Scaled: {column}"))
        # Original data histogram
        fig.add_trace(
            go.Histogram(x=df_original[column], name='Original'),
            row=1, col=1
        )
        # Scaled data histogram
        fig.add_trace(
            go.Histogram(x=df_scaled[column], name='Scaled'),
            row=1, col=2
        )
        # Update layout for better visual comparison
        fig.update_layout(height=400, width=800, title_text=f"Comparison for {column}", bargap=0.1)
        fig.update_traces(opacity=0.75)  # Adjust opacity to see overlapping bars more clearly
        # Display the plot
        st.plotly_chart(fig)