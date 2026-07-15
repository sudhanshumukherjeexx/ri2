# timeseries.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

def analyze_time_series_pattern(data, date_column):
    """
    Analyze time series data pattern for intelligent preprocessing.
    """
    # Convert Series to DataFrame if necessary
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()
    
    # Ensure date column exists
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in data")
    
    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    
    time_diffs = df[date_column].diff()[1:]
    
    min_diff = time_diffs.min()
    median_diff = time_diffs.median()
    
    if median_diff.total_seconds() < 60:
        frequency = 'seconds'
        freq_code = 'S'
    elif median_diff.total_seconds() < 3600:
        frequency = 'minutes'
        freq_code = 'T'
    elif median_diff.total_seconds() < 86400:
        frequency = 'hours'
        freq_code = 'H'
    elif median_diff.days < 7:
        frequency = 'days'
        freq_code = 'D'
    elif median_diff.days < 31:
        frequency = 'weeks'
        freq_code = 'W'
    elif median_diff.days < 365:
        frequency = 'months'
        freq_code = 'M'
    else:
        frequency = 'years'
        freq_code = 'Y'
    
    time_diffs_std = time_diffs.std()
    regularity_threshold = 0.1
    regular_intervals = (time_diffs_std / median_diff) < regularity_threshold if not median_diff.total_seconds() == 0 else False
    
    return {
        'frequency': frequency,
        'freq_code': freq_code,
        'min_interval': min_diff,
        'median_interval': median_diff,
        'has_duplicates': df[date_column].duplicated().any(),
        'unique_dates': df[date_column].nunique(),
        'total_records': len(df),
        'regular_intervals': regular_intervals
    }

def validate_time_series_data(df, date_column, target_column):
    """
    Validate time series data requirements.
    """
    errors = []
    warnings = []
    
    df = df.copy()
    
    if date_column not in df.columns:
        errors.append(f"❌ Date column '{date_column}' not found")
        return errors
    if target_column not in df.columns:
        errors.append(f"❌ Target column '{target_column}' not found")
        return errors
    
    try:
        df[date_column] = pd.to_datetime(df[date_column])
        if df[date_column].dt.tz is not None:
            df[date_column] = df[date_column].dt.tz_localize(None)
    except Exception as e:
        errors.append(f"❌ Date column conversion error: {str(e)}")
        return errors
    
    if not np.issubdtype(df[target_column].dtype, np.number):
        errors.append(f"❌ Target column must be numeric")
    
    if df[date_column].isnull().any():
        errors.append("❌ Date column has missing values")
    if df[target_column].isnull().any():
        errors.append("❌ Target column has missing values")
    
    if len(df) < 30:
        errors.append("❌ Need at least 30 data points")
        return errors
    
    df = df.sort_values(date_column)
    
    if df[date_column].duplicated().any():
        duplicates = df[df[date_column].duplicated(keep=False)]
        errors.append(f"❌ Found {len(duplicates)} duplicate dates")
    
    date_diffs = df[date_column].diff()[1:]
    common_interval = date_diffs.mode()[0]
    
    if not df[date_column].is_monotonic_increasing:
        errors.append("❌ Dates not in ascending order")
    
    irregular_intervals = date_diffs != common_interval
    if irregular_intervals.any():
        irregular_count = irregular_intervals.sum()
        irregularity_percentage = (irregular_count / len(date_diffs)) * 100
        
        if irregularity_percentage > 20:
            errors.append(f"❌ {irregularity_percentage:.1f}% irregular intervals")
        elif irregularity_percentage > 5:
            warnings.append(f"⚠️ {irregularity_percentage:.1f}% irregular intervals")
    
    errors.extend(warnings)
    return errors

def prepare_time_series_data(df, date_column, target_column, resample_freq=None, agg_method='mean'):
    """
    Prepare data for time series analysis.
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    if df[date_column].dt.tz is not None:
        df[date_column] = df[date_column].dt.tz_localize(None)
    
    df = df.sort_values(date_column)
    df = df.drop_duplicates(subset=[date_column], keep='last')
    df.set_index(date_column, inplace=True)
    
    if resample_freq:
        df_resampled = df[target_column].resample(resample_freq).agg(agg_method)
    else:
        df_resampled = df[target_column]
    
    df_resampled = df_resampled.fillna(method='ffill', limit=2)
    df_resampled = df_resampled.dropna()
    
    train_size = int(len(df_resampled) * 0.8)
    train_data = df_resampled[:train_size]
    test_data = df_resampled[train_size:]
    
    return train_data, test_data, df_resampled.to_frame(target_column)

def train_models(train_data, test_data, target_column, forecast_periods):
    """
    Train XGBoost model for time series prediction.
    """
    predictions = {}
    
    # Debug information
    st.write("Debug Info:")
    st.write(f"Train data type: {type(train_data)}")
    st.write(f"Train data shape: {train_data.shape if hasattr(train_data, 'shape') else len(train_data)}")
    
    # Since we're working with a Series from prepare_time_series_data
    train_series = train_data if isinstance(train_data, pd.Series) else train_data[target_column]
    test_series = test_data if isinstance(test_data, pd.Series) else test_data[target_column]

    # XGBoost
    try:
        n_steps = 5
        X_train, y_train = [], []
        series_values = train_series.values
        
        for i in range(len(series_values) - n_steps):
            X_train.append(series_values[i:(i + n_steps)])
            y_train.append(series_values[i + n_steps])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Debug info for XGBoost
        st.write("XGBoost preparation:")
        st.write(f"X_train shape: {X_train.shape if len(X_train) > 0 else 'Empty'}")
        st.write(f"y_train shape: {y_train.shape if len(y_train) > 0 else 'Empty'}")
        
        if len(X_train) > 0:
            model_xgb = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model_xgb.fit(X_train, y_train)
            
            # Generate predictions for test data
            X_test = []
            test_values = test_series.values
            for i in range(len(test_values) - n_steps):
                X_test.append(test_values[i:(i + n_steps)])
            X_test = np.array(X_test)
            test_pred = model_xgb.predict(X_test)
            
            # Generate future predictions
            future_X = []
            last_values = np.concatenate([test_values[-n_steps:]])
            
            for _ in range(forecast_periods):
                future_X.append(last_values[-n_steps:])
                next_pred = model_xgb.predict([last_values[-n_steps:]])[0]
                last_values = np.append(last_values[1:], next_pred)
            
            future_pred = model_xgb.predict(future_X)
            
            # Combine test predictions and future predictions
            all_dates = pd.date_range(
                start=test_series.index[n_steps],
                periods=len(test_pred) + forecast_periods,
                freq=test_series.index.freq
            )
            
            all_predictions = np.concatenate([test_pred, future_pred])
            
            predictions['XGBoost'] = pd.DataFrame({
                'ds': all_dates,
                'yhat': all_predictions,
                'yhat_lower': all_predictions * 0.9,
                'yhat_upper': all_predictions * 1.1,
                'is_future': [False] * len(test_pred) + [True] * len(future_pred)
            })
            st.success("XGBoost model trained successfully!")
        else:
            st.warning("Not enough data points for XGBoost training after sequence preparation")
    except Exception as e:
        st.error(f"XGBoost model error: {str(e)}")
    
    return predictions
    
    # Prophet
    try:
        prophet_data = pd.DataFrame({
            'ds': train_data.index,
            'y': train_series.values  # Use values directly
        })
        model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model_prophet.fit(prophet_data)
        future = model_prophet.make_future_dataframe(periods=forecast_periods, freq=train_data.index.freq)
        forecast_prophet = model_prophet.predict(future)
        predictions['Prophet'] = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        st.success("Prophet model trained successfully!")
    except Exception as e:
        st.error(f"Prophet model error: {str(e)}")
        st.write("Prophet data preview:", prophet_data.head())

    # SARIMA
    try:
        model_sarima = SARIMAX(train_series.values,  # Use values directly
                              order=(1,1,1),
                              seasonal_order=(1,1,1,12))
        results_sarima = model_sarima.fit(disp=False)
        forecast_sarima = results_sarima.get_forecast(steps=forecast_periods)
        predictions['SARIMA'] = pd.DataFrame({
            'ds': pd.date_range(start=train_data.index[-1], 
                              periods=forecast_periods+1,
                              freq=train_data.index.freq)[1:],
            'yhat': forecast_sarima.predicted_mean,
            'yhat_lower': forecast_sarima.conf_int().iloc[:,0],
            'yhat_upper': forecast_sarima.conf_int().iloc[:,1]
        })
        st.success("SARIMA model trained successfully!")
    except Exception as e:
        st.error(f"SARIMA model error: {str(e)}")
        st.write("SARIMA input shape:", train_series.shape)

    # XGBoost
    try:
        n_steps = 5
        X_train, y_train = [], []
        series_values = train_series.values
        
        for i in range(len(series_values) - n_steps):
            X_train.append(series_values[i:(i + n_steps)])
            y_train.append(series_values[i + n_steps])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Debug info for XGBoost
        st.write(f"XGBoost training data shape: X={X_train.shape}, y={y_train.shape}")
        
        model_xgb = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model_xgb.fit(X_train, y_train)
        
        # Prepare test data
        X_test = []
        test_values = test_series.values
        for i in range(len(test_values) - n_steps):
            X_test.append(test_values[i:(i + n_steps)])
        X_test = np.array(X_test)
        
        xgb_pred = model_xgb.predict(X_test)
        predictions['XGBoost'] = pd.DataFrame({
            'ds': test_data.index[n_steps:],
            'yhat': xgb_pred,
            'yhat_lower': xgb_pred * 0.9,
            'yhat_upper': xgb_pred * 1.1
        })
        st.success("XGBoost model trained successfully!")
    except Exception as e:
        st.error(f"XGBoost model error: {str(e)}")
        if len(X_train) > 0:
            st.write("XGBoost input shapes:", 
                    f"X_train: {X_train.shape}",
                    f"y_train: {y_train.shape}")
    
    return predictions

def plot_forecasts(original_df, forecasts_dict):
    """
    Create interactive plot with forecasts, highlighting future predictions.
    """
    fig = go.Figure()

    # Plot original data
    fig.add_trace(go.Scatter(
        x=original_df.index,
        y=original_df.values,
        name='Historical Data',
        mode='lines',
        line=dict(color='black', width=2)
    ))

    # Plot predictions with different colors for historical and future predictions
    forecast = forecasts_dict['XGBoost']
    
    # Historical predictions
    historical_mask = ~forecast['is_future']
    fig.add_trace(go.Scatter(
        x=forecast[historical_mask]['ds'],
        y=forecast[historical_mask]['yhat'],
        name='XGBoost Historical Predictions',
        mode='lines',
        line=dict(color='blue', width=2)
    ))
    
    # Confidence interval for historical predictions
    fig.add_trace(go.Scatter(
        x=forecast[historical_mask]['ds'].tolist() + forecast[historical_mask]['ds'].tolist()[::-1],
        y=forecast[historical_mask]['yhat_upper'].tolist() + forecast[historical_mask]['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,0,255,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Historical Prediction Interval'
    ))
    
    # Future predictions
    future_mask = forecast['is_future']
    fig.add_trace(go.Scatter(
        x=forecast[future_mask]['ds'],
        y=forecast[future_mask]['yhat'],
        name='XGBoost Future Predictions',
        mode='lines',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval for future predictions
    fig.add_trace(go.Scatter(
        x=forecast[future_mask]['ds'].tolist() + forecast[future_mask]['ds'].tolist()[::-1],
        y=forecast[future_mask]['yhat_upper'].tolist() + forecast[future_mask]['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Future Prediction Interval'
    ))

    fig.update_layout(
        title='Time Series Forecast with XGBoost',
        xaxis_title='Date',
        yaxis_title='Value',
        height=600,
        showlegend=True,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def perform_time_series_analysis(current_df):
    """
    Main function for time series analysis interface
    """
    st.header("📈 Time Series Analysis")
    
    with st.expander("ℹ️ Important Information", expanded=True):
        st.markdown("""
        ### Requirements:
        1. **Date/Time Column**: Valid dates/timestamps, no missing values
        2. **Target Column**: Numeric values, no missing values
        3. **Data**: Minimum 30 points, chronological order
        
        ### Models Used:
        - Prophet (Facebook/Meta)
        - SARIMA (Seasonal ARIMA)
        - XGBoost
        """)

    current_df = current_df.to_pandas()
    # Column selection
    col1, col2 = st.columns(2)
    with col1:
        date_column = st.selectbox("Select Date/Time Column", options=current_df.columns)
    with col2:
        target_column = st.selectbox("Select Target Column", options=current_df.columns)

    try:
        # Debug information
        st.write("Debug Info:")
        st.write(f"Data type: {type(current_df)}")
        st.write(f"Columns: {current_df.columns.tolist() if hasattr(current_df, 'columns') else 'No columns (Series)'}")
        
        # Analyze pattern
        pattern_info = analyze_time_series_pattern(current_df, date_column)
        
        # Display pattern analysis
        st.subheader("📊 Data Pattern Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frequency", pattern_info['frequency'].title())
            st.metric("Total Records", pattern_info['total_records'])
        with col2:
            st.metric("Unique Dates", pattern_info['unique_dates'])
            st.metric("Regular Intervals", "Yes" if pattern_info['regular_intervals'] else "No")
        with col3:
            st.metric("Has Duplicates", "Yes" if pattern_info['has_duplicates'] else "No")
    except Exception as e:
        st.error(f"❌ Error in pattern analysis: {str(e)}")
        st.warning("Unable to analyze time series patterns. Please check your data format.")

    # Processing options
    with st.expander("⚙️ Processing Options", expanded=True):
        resample_freq = st.selectbox(
            "Time Frequency",
            options=[None, 'D', 'W', 'M', 'Q', 'Y'],
            format_func=lambda x: 'No resampling' if x is None else f'{x} (Daily/Weekly/Monthly/Quarterly/Yearly)'
        )
        
        agg_method = st.selectbox(
            "Aggregation Method",
            options=['mean', 'sum', 'max', 'min', 'first', 'last']
        )
        
        forecast_periods = st.slider(
            "Forecast Periods",
            min_value=1,
            max_value=365,
            value=30
        )

    if st.button("🎯 Run Analysis"):
        with st.spinner("Analyzing data..."):
            # Validate
            errors = validate_time_series_data(current_df, date_column, target_column)
            if errors:
                st.error("Please fix the following:")
                for error in errors:
                    st.warning(error)
                return

            # Prepare data
            train_data, test_data, processed_df = prepare_time_series_data(
                current_df,
                date_column,
                target_column,
                resample_freq,
                agg_method
            )

            # Train models and get predictions
            forecasts = train_models(train_data, test_data, target_column, forecast_periods)

            if forecasts:
                # Create tabs for results
                tab1, tab2 = st.tabs(["📈 Visualization", "📊 Model Performance"])
                
                with tab1:
                    fig = plot_forecasts(processed_df[target_column], forecasts)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("### 📝 Forecast Interpretation")
                    st.write("""
                    - The graph shows actual historical data and predictions from multiple models
                    - Each model's forecast is shown with a confidence interval
                    - Different colors represent different models' predictions
                    - The shaded areas show the uncertainty range for each model
                    """)
                
                with tab2:
                    st.write("### Model Performance Metrics")
                    
                    # Calculate and display metrics for each model
                    metrics_dict = {}
                    for model_name, forecast in forecasts.items():
                        # Get overlapping dates between actual and predictions
                        common_dates = set(processed_df.index).intersection(set(forecast['ds']))
                        if common_dates:
                            actual = processed_df[processed_df.index.isin(common_dates)][target_column]
                            pred = forecast[forecast['ds'].isin(common_dates)]['yhat']
                            
                            metrics = {
                                'Mean Squared Error': mean_squared_error(actual, pred),
                                'Root Mean Squared Error': np.sqrt(mean_squared_error(actual, pred)),
                                'Mean Absolute Error': mean_absolute_error(actual, pred),
                                'R-squared Score': r2_score(actual, pred)
                            }
                            metrics_dict[model_name] = metrics
                    
                    # Create and display metrics DataFrame
                    metrics_df = pd.DataFrame(metrics_dict).round(4)
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Add metrics interpretation
                    st.info("""
                    💡 Metrics Interpretation:
                    - MSE (Mean Squared Error): Lower is better
                    - RMSE (Root Mean Squared Error): Lower is better, same units as data
                    - MAE (Mean Absolute Error): Lower is better, same units as data
                    - R-squared: Higher is better (0-1 range), indicates fit quality
                    """)
                    
                    # Download results
                    for model_name, forecast in forecasts.items():
                        forecast_csv = forecast.to_csv(index=False)
                        st.download_button(
                            f"Download {model_name} Forecast",
                            forecast_csv,
                            f"{model_name.lower()}_forecast.csv",
                            "text/csv",
                            key=f'download_{model_name}'
                        )





#---------------------------------------------------

# # timeseries.py

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from prophet import Prophet
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import streamlit as st

# def analyze_time_series_pattern(data, date_column):
#     """
#     Analyze time series data pattern for intelligent preprocessing.
#     """
#     # Convert Series to DataFrame if necessary
#     if isinstance(data, pd.Series):
#         df = data.to_frame()
#     else:
#         df = data.copy()
    
#     # Ensure date column exists
#     if date_column not in df.columns:
#         raise ValueError(f"Date column '{date_column}' not found in data")
    
#     # Convert date column to datetime
#     df[date_column] = pd.to_datetime(df[date_column])
#     df = df.sort_values(date_column)
    
#     time_diffs = df[date_column].diff()[1:]
    
#     min_diff = time_diffs.min()
#     median_diff = time_diffs.median()
    
#     if median_diff.total_seconds() < 60:
#         frequency = 'seconds'
#         freq_code = 'S'
#     elif median_diff.total_seconds() < 3600:
#         frequency = 'minutes'
#         freq_code = 'T'
#     elif median_diff.total_seconds() < 86400:
#         frequency = 'hours'
#         freq_code = 'H'
#     elif median_diff.days < 7:
#         frequency = 'days'
#         freq_code = 'D'
#     elif median_diff.days < 31:
#         frequency = 'weeks'
#         freq_code = 'W'
#     elif median_diff.days < 365:
#         frequency = 'months'
#         freq_code = 'M'
#     else:
#         frequency = 'years'
#         freq_code = 'Y'
    
#     time_diffs_std = time_diffs.std()
#     regularity_threshold = 0.1
#     regular_intervals = (time_diffs_std / median_diff) < regularity_threshold if not median_diff.total_seconds() == 0 else False
    
#     return {
#         'frequency': frequency,
#         'freq_code': freq_code,
#         'min_interval': min_diff,
#         'median_interval': median_diff,
#         'has_duplicates': df[date_column].duplicated().any(),
#         'unique_dates': df[date_column].nunique(),
#         'total_records': len(df),
#         'regular_intervals': regular_intervals
#     }

# def validate_time_series_data(df, date_column, target_column):
#     """
#     Validate time series data requirements.
#     """
#     errors = []
#     warnings = []
    
#     df = df.copy()
    
#     if date_column not in df.columns:
#         errors.append(f"❌ Date column '{date_column}' not found")
#         return errors
#     if target_column not in df.columns:
#         errors.append(f"❌ Target column '{target_column}' not found")
#         return errors
    
#     try:
#         df[date_column] = pd.to_datetime(df[date_column])
#         if df[date_column].dt.tz is not None:
#             df[date_column] = df[date_column].dt.tz_localize(None)
#     except Exception as e:
#         errors.append(f"❌ Date column conversion error: {str(e)}")
#         return errors
    
#     if not np.issubdtype(df[target_column].dtype, np.number):
#         errors.append(f"❌ Target column must be numeric")
    
#     if df[date_column].isnull().any():
#         errors.append("❌ Date column has missing values")
#     if df[target_column].isnull().any():
#         errors.append("❌ Target column has missing values")
    
#     if len(df) < 30:
#         errors.append("❌ Need at least 30 data points")
#         return errors
    
#     df = df.sort_values(date_column)
    
#     if df[date_column].duplicated().any():
#         duplicates = df[df[date_column].duplicated(keep=False)]
#         errors.append(f"❌ Found {len(duplicates)} duplicate dates")
    
#     date_diffs = df[date_column].diff()[1:]
#     common_interval = date_diffs.mode()[0]
    
#     if not df[date_column].is_monotonic_increasing:
#         errors.append("❌ Dates not in ascending order")
    
#     irregular_intervals = date_diffs != common_interval
#     if irregular_intervals.any():
#         irregular_count = irregular_intervals.sum()
#         irregularity_percentage = (irregular_count / len(date_diffs)) * 100
        
#         if irregularity_percentage > 20:
#             errors.append(f"❌ {irregularity_percentage:.1f}% irregular intervals")
#         elif irregularity_percentage > 5:
#             warnings.append(f"⚠️ {irregularity_percentage:.1f}% irregular intervals")
    
#     errors.extend(warnings)
#     return errors

# def prepare_time_series_data(df, date_column, target_column, resample_freq=None, agg_method='mean'):
#     """
#     Prepare data for time series analysis.
#     """
#     df = df.copy()
    
#     if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
#         df[date_column] = pd.to_datetime(df[date_column])
    
#     if df[date_column].dt.tz is not None:
#         df[date_column] = df[date_column].dt.tz_localize(None)
    
#     df = df.sort_values(date_column)
#     df = df.drop_duplicates(subset=[date_column], keep='last')
#     df.set_index(date_column, inplace=True)
    
#     if resample_freq:
#         df_resampled = df[target_column].resample(resample_freq).agg(agg_method)
#     else:
#         df_resampled = df[target_column]
    
#     df_resampled = df_resampled.fillna(method='ffill', limit=2)
#     df_resampled = df_resampled.dropna()
    
#     train_size = int(len(df_resampled) * 0.8)
#     train_data = df_resampled[:train_size]
#     test_data = df_resampled[train_size:]
    
#     return train_data, test_data, df_resampled.to_frame(target_column)

# def train_models(train_data, test_data, target_column, forecast_periods):
#     """
#     Train multiple time series models.
#     """
#     predictions = {}
    
#     # Debug information
#     st.write("Debug Info:")
#     st.write(f"Train data type: {type(train_data)}")
#     st.write(f"Train data shape: {train_data.shape if hasattr(train_data, 'shape') else len(train_data)}")
    
#     # Since we're working with a Series from prepare_time_series_data
#     train_series = train_data if isinstance(train_data, pd.Series) else train_data[target_column]
#     test_series = test_data if isinstance(test_data, pd.Series) else test_data[target_column]
    
#     # Prophet
#     try:
#         prophet_data = pd.DataFrame({
#             'ds': train_series.index,
#             'y': train_series.values
#         })
#         model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
#         model_prophet.fit(prophet_data)
#         future = model_prophet.make_future_dataframe(periods=forecast_periods)
#         forecast_prophet = model_prophet.predict(future)
#         predictions['Prophet'] = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#         st.success("Prophet model trained successfully!")
#     except Exception as e:
#         st.error(f"Prophet model error: {str(e)}")
#         st.write("Prophet data preview:", prophet_data.head() if 'prophet_data' in locals() else "No data preview available")

#     # XGBoost
#     try:
#         n_steps = 5
#         X_train, y_train = [], []
#         series_values = train_series.values
        
#         for i in range(len(series_values) - n_steps):
#             X_train.append(series_values[i:(i + n_steps)])
#             y_train.append(series_values[i + n_steps])
        
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
        
#         # Debug info for XGBoost
#         st.write("XGBoost preparation:")
#         st.write(f"X_train shape: {X_train.shape if len(X_train) > 0 else 'Empty'}")
#         st.write(f"y_train shape: {y_train.shape if len(y_train) > 0 else 'Empty'}")
        
#         if len(X_train) > 0:
#             model_xgb = xgb.XGBRegressor(
#                 objective='reg:squarederror',
#                 n_estimators=200,
#                 max_depth=5,
#                 learning_rate=0.05,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 random_state=42
#             )
#             model_xgb.fit(X_train, y_train)
            
#             # Prepare test data
#             X_test = []
#             test_values = test_series.values
#             for i in range(len(test_values) - n_steps):
#                 X_test.append(test_values[i:(i + n_steps)])
#             X_test = np.array(X_test)
            
#             xgb_pred = model_xgb.predict(X_test)
#             predictions['XGBoost'] = pd.DataFrame({
#                 'ds': test_series.index[n_steps:],
#                 'yhat': xgb_pred,
#                 'yhat_lower': xgb_pred * 0.9,
#                 'yhat_upper': xgb_pred * 1.1
#             })
#             st.success("XGBoost model trained successfully!")
#         else:
#             st.warning("Not enough data points for XGBoost training after sequence preparation")
#     except Exception as e:
#         st.error(f"XGBoost model error: {str(e)}")
    
#     return predictions
    
#     # Prophet
#     try:
#         prophet_data = pd.DataFrame({
#             'ds': train_data.index,
#             'y': train_series.values  # Use values directly
#         })
#         model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
#         model_prophet.fit(prophet_data)
#         future = model_prophet.make_future_dataframe(periods=forecast_periods, freq=train_data.index.freq)
#         forecast_prophet = model_prophet.predict(future)
#         predictions['Prophet'] = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#         st.success("Prophet model trained successfully!")
#     except Exception as e:
#         st.error(f"Prophet model error: {str(e)}")
#         st.write("Prophet data preview:", prophet_data.head())

#     # SARIMA
#     try:
#         model_sarima = SARIMAX(train_series.values,  # Use values directly
#                               order=(1,1,1),
#                               seasonal_order=(1,1,1,12))
#         results_sarima = model_sarima.fit(disp=False)
#         forecast_sarima = results_sarima.get_forecast(steps=forecast_periods)
#         predictions['SARIMA'] = pd.DataFrame({
#             'ds': pd.date_range(start=train_data.index[-1], 
#                               periods=forecast_periods+1,
#                               freq=train_data.index.freq)[1:],
#             'yhat': forecast_sarima.predicted_mean,
#             'yhat_lower': forecast_sarima.conf_int().iloc[:,0],
#             'yhat_upper': forecast_sarima.conf_int().iloc[:,1]
#         })
#         st.success("SARIMA model trained successfully!")
#     except Exception as e:
#         st.error(f"SARIMA model error: {str(e)}")
#         st.write("SARIMA input shape:", train_series.shape)

#     # XGBoost
#     try:
#         n_steps = 5
#         X_train, y_train = [], []
#         series_values = train_series.values
        
#         for i in range(len(series_values) - n_steps):
#             X_train.append(series_values[i:(i + n_steps)])
#             y_train.append(series_values[i + n_steps])
        
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
        
#         # Debug info for XGBoost
#         st.write(f"XGBoost training data shape: X={X_train.shape}, y={y_train.shape}")
        
#         model_xgb = xgb.XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=200,
#             max_depth=5,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )
#         model_xgb.fit(X_train, y_train)
        
#         # Prepare test data
#         X_test = []
#         test_values = test_series.values
#         for i in range(len(test_values) - n_steps):
#             X_test.append(test_values[i:(i + n_steps)])
#         X_test = np.array(X_test)
        
#         xgb_pred = model_xgb.predict(X_test)
#         predictions['XGBoost'] = pd.DataFrame({
#             'ds': test_data.index[n_steps:],
#             'yhat': xgb_pred,
#             'yhat_lower': xgb_pred * 0.9,
#             'yhat_upper': xgb_pred * 1.1
#         })
#         st.success("XGBoost model trained successfully!")
#     except Exception as e:
#         st.error(f"XGBoost model error: {str(e)}")
#         if len(X_train) > 0:
#             st.write("XGBoost input shapes:", 
#                     f"X_train: {X_train.shape}",
#                     f"y_train: {y_train.shape}")
    
#     return predictions

# def plot_forecasts(original_df, forecasts_dict):
#     """
#     Create interactive plot with forecasts from all models.
#     """
#     fig = go.Figure()

#     # Plot original data
#     fig.add_trace(go.Scatter(
#         x=original_df.index,
#         y=original_df.values,
#         name='Actual Data',
#         mode='lines'
#     ))

#     colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)']
#     for (model_name, forecast), color in zip(forecasts_dict.items(), colors):
#         fig.add_trace(go.Scatter(
#             x=forecast['ds'],
#             y=forecast['yhat'],
#             name=f'{model_name} Forecast',
#             mode='lines'
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
#             y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
#             fill='toself',
#             fillcolor=color,
#             line=dict(color='rgba(255,255,255,0)'),
#             name=f'{model_name} Confidence Interval'
#         ))

#     fig.update_layout(
#         title='Time Series Forecasts',
#         xaxis_title='Date',
#         yaxis_title='Value',
#         height=600,
#         showlegend=True,
#         template='plotly_white'
#     )
    
#     return fig

# def perform_time_series_analysis(current_df):
#     """
#     Main function for time series analysis interface
#     """
#     st.header("📈 Time Series Analysis")
    
#     with st.expander("ℹ️ Important Information", expanded=True):
#         st.markdown("""
#         ### Requirements:
#         1. **Date/Time Column**: Valid dates/timestamps, no missing values
#         2. **Target Column**: Numeric values, no missing values
#         3. **Data**: Minimum 30 points, chronological order
        
#         ### Models Used:
#         - Prophet (Facebook/Meta)
#         - SARIMA (Seasonal ARIMA)
#         - XGBoost
#         """)

#     current_df = current_df.to_pandas()
#     # Column selection
#     col1, col2 = st.columns(2)
#     with col1:
#         date_column = st.selectbox("Select Date/Time Column", options=current_df.columns)
#     with col2:
#         target_column = st.selectbox("Select Target Column", options=current_df.columns)

#     try:
#         # Debug information
#         st.write("Debug Info:")
#         st.write(f"Data type: {type(current_df)}")
#         st.write(f"Columns: {current_df.columns.tolist() if hasattr(current_df, 'columns') else 'No columns (Series)'}")
        
#         # Analyze pattern
#         pattern_info = analyze_time_series_pattern(current_df, date_column)
        
#         # Display pattern analysis
#         st.subheader("📊 Data Pattern Analysis")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Frequency", pattern_info['frequency'].title())
#             st.metric("Total Records", pattern_info['total_records'])
#         with col2:
#             st.metric("Unique Dates", pattern_info['unique_dates'])
#             st.metric("Regular Intervals", "Yes" if pattern_info['regular_intervals'] else "No")
#         with col3:
#             st.metric("Has Duplicates", "Yes" if pattern_info['has_duplicates'] else "No")
#     except Exception as e:
#         st.error(f"❌ Error in pattern analysis: {str(e)}")
#         st.warning("Unable to analyze time series patterns. Please check your data format.")

#     # Processing options
#     with st.expander("⚙️ Processing Options", expanded=True):
#         resample_freq = st.selectbox(
#             "Time Frequency",
#             options=[None, 'D', 'W', 'M', 'Q', 'Y'],
#             format_func=lambda x: 'No resampling' if x is None else f'{x} (Daily/Weekly/Monthly/Quarterly/Yearly)'
#         )
        
#         agg_method = st.selectbox(
#             "Aggregation Method",
#             options=['mean', 'sum', 'max', 'min', 'first', 'last']
#         )
        
#         forecast_periods = st.slider(
#             "Forecast Periods",
#             min_value=1,
#             max_value=365,
#             value=30
#         )

#     if st.button("🎯 Run Analysis"):
#         with st.spinner("Analyzing data..."):
#             # Validate
#             errors = validate_time_series_data(current_df, date_column, target_column)
#             if errors:
#                 st.error("Please fix the following:")
#                 for error in errors:
#                     st.warning(error)
#                 return

#             # Prepare data
#             train_data, test_data, processed_df = prepare_time_series_data(
#                 current_df,
#                 date_column,
#                 target_column,
#                 resample_freq,
#                 agg_method
#             )

#             # Train models and get predictions
#             forecasts = train_models(train_data, test_data, target_column, forecast_periods)

#             if forecasts:
#                 # Create tabs for results
#                 tab1, tab2 = st.tabs(["📈 Visualization", "📊 Model Performance"])
                
#                 with tab1:
#                     fig = plot_forecasts(processed_df[target_column], forecasts)
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     st.write("### 📝 Forecast Interpretation")
#                     st.write("""
#                     - The graph shows actual historical data and predictions from multiple models
#                     - Each model's forecast is shown with a confidence interval
#                     - Different colors represent different models' predictions
#                     - The shaded areas show the uncertainty range for each model
#                     """)
                
#                 with tab2:
#                     st.write("### Model Performance Metrics")
                    
#                     # Calculate and display metrics for each model
#                     metrics_dict = {}
#                     for model_name, forecast in forecasts.items():
#                         # Get overlapping dates between actual and predictions
#                         common_dates = set(processed_df.index).intersection(set(forecast['ds']))
#                         if common_dates:
#                             actual = processed_df[processed_df.index.isin(common_dates)][target_column]
#                             pred = forecast[forecast['ds'].isin(common_dates)]['yhat']
                            
#                             metrics = {
#                                 'Mean Squared Error': mean_squared_error(actual, pred),
#                                 'Root Mean Squared Error': np.sqrt(mean_squared_error(actual, pred)),
#                                 'Mean Absolute Error': mean_absolute_error(actual, pred),
#                                 'R-squared Score': r2_score(actual, pred)
#                             }
#                             metrics_dict[model_name] = metrics
                    
#                     # Create and display metrics DataFrame
#                     metrics_df = pd.DataFrame(metrics_dict).round(4)
#                     st.dataframe(metrics_df, use_container_width=True)
                    
#                     # Add metrics interpretation
#                     st.info("""
#                     💡 Metrics Interpretation:
#                     - MSE (Mean Squared Error): Lower is better
#                     - RMSE (Root Mean Squared Error): Lower is better, same units as data
#                     - MAE (Mean Absolute Error): Lower is better, same units as data
#                     - R-squared: Higher is better (0-1 range), indicates fit quality
#                     """)
                    
#                     # Download results
#                     for model_name, forecast in forecasts.items():
#                         forecast_csv = forecast.to_csv(index=False)
#                         st.download_button(
#                             f"Download {model_name} Forecast",
#                             forecast_csv,
#                             f"{model_name.lower()}_forecast.csv",
#                             "text/csv",
#                             key=f'download_{model_name}'
#                         )






#------------
# # timeseries.py

# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from prophet import Prophet
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import plotly.graph_objects as go
# from datetime import datetime, timedelta
# import streamlit as st

# def analyze_time_series_pattern(data, date_column):
#     """
#     Analyze time series data pattern for intelligent preprocessing.
#     """
#     # Convert Series to DataFrame if necessary
#     if isinstance(data, pd.Series):
#         df = data.to_frame()
#     else:
#         df = data.copy()
    
#     # Ensure date column exists
#     if date_column not in df.columns:
#         raise ValueError(f"Date column '{date_column}' not found in data")
    
#     # Convert date column to datetime
#     df[date_column] = pd.to_datetime(df[date_column])
#     df = df.sort_values(date_column)
    
#     time_diffs = df[date_column].diff()[1:]
    
#     min_diff = time_diffs.min()
#     median_diff = time_diffs.median()
    
#     if median_diff.total_seconds() < 60:
#         frequency = 'seconds'
#         freq_code = 'S'
#     elif median_diff.total_seconds() < 3600:
#         frequency = 'minutes'
#         freq_code = 'T'
#     elif median_diff.total_seconds() < 86400:
#         frequency = 'hours'
#         freq_code = 'H'
#     elif median_diff.days < 7:
#         frequency = 'days'
#         freq_code = 'D'
#     elif median_diff.days < 31:
#         frequency = 'weeks'
#         freq_code = 'W'
#     elif median_diff.days < 365:
#         frequency = 'months'
#         freq_code = 'M'
#     else:
#         frequency = 'years'
#         freq_code = 'Y'
    
#     time_diffs_std = time_diffs.std()
#     regularity_threshold = 0.1
#     regular_intervals = (time_diffs_std / median_diff) < regularity_threshold if not median_diff.total_seconds() == 0 else False
    
#     return {
#         'frequency': frequency,
#         'freq_code': freq_code,
#         'min_interval': min_diff,
#         'median_interval': median_diff,
#         'has_duplicates': df[date_column].duplicated().any(),
#         'unique_dates': df[date_column].nunique(),
#         'total_records': len(df),
#         'regular_intervals': regular_intervals
#     }

# def validate_time_series_data(df, date_column, target_column):
#     """
#     Validate time series data requirements.
#     """
#     errors = []
#     warnings = []
    
#     df = df.copy()
    
#     if date_column not in df.columns:
#         errors.append(f"❌ Date column '{date_column}' not found")
#         return errors
#     if target_column not in df.columns:
#         errors.append(f"❌ Target column '{target_column}' not found")
#         return errors
    
#     try:
#         df[date_column] = pd.to_datetime(df[date_column])
#         if df[date_column].dt.tz is not None:
#             df[date_column] = df[date_column].dt.tz_localize(None)
#     except Exception as e:
#         errors.append(f"❌ Date column conversion error: {str(e)}")
#         return errors
    
#     if not np.issubdtype(df[target_column].dtype, np.number):
#         errors.append(f"❌ Target column must be numeric")
    
#     if df[date_column].isnull().any():
#         errors.append("❌ Date column has missing values")
#     if df[target_column].isnull().any():
#         errors.append("❌ Target column has missing values")
    
#     if len(df) < 30:
#         errors.append("❌ Need at least 30 data points")
#         return errors
    
#     df = df.sort_values(date_column)
    
#     if df[date_column].duplicated().any():
#         duplicates = df[df[date_column].duplicated(keep=False)]
#         errors.append(f"❌ Found {len(duplicates)} duplicate dates")
    
#     date_diffs = df[date_column].diff()[1:]
#     common_interval = date_diffs.mode()[0]
    
#     if not df[date_column].is_monotonic_increasing:
#         errors.append("❌ Dates not in ascending order")
    
#     irregular_intervals = date_diffs != common_interval
#     if irregular_intervals.any():
#         irregular_count = irregular_intervals.sum()
#         irregularity_percentage = (irregular_count / len(date_diffs)) * 100
        
#         if irregularity_percentage > 20:
#             errors.append(f"❌ {irregularity_percentage:.1f}% irregular intervals")
#         elif irregularity_percentage > 5:
#             warnings.append(f"⚠️ {irregularity_percentage:.1f}% irregular intervals")
    
#     errors.extend(warnings)
#     return errors

# def prepare_time_series_data(df, date_column, target_column, resample_freq=None, agg_method='mean'):
#     """
#     Prepare data for time series analysis.
#     """
#     df = df.copy()
    
#     if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
#         df[date_column] = pd.to_datetime(df[date_column])
    
#     if df[date_column].dt.tz is not None:
#         df[date_column] = df[date_column].dt.tz_localize(None)
    
#     df = df.sort_values(date_column)
#     df = df.drop_duplicates(subset=[date_column], keep='last')
#     df.set_index(date_column, inplace=True)
    
#     if resample_freq:
#         df_resampled = df[target_column].resample(resample_freq).agg(agg_method)
#     else:
#         df_resampled = df[target_column]
    
#     df_resampled = df_resampled.fillna(method='ffill', limit=2)
#     df_resampled = df_resampled.dropna()
    
#     train_size = int(len(df_resampled) * 0.8)
#     train_data = df_resampled[:train_size]
#     test_data = df_resampled[train_size:]
    
#     return train_data, test_data, df_resampled.to_frame(target_column)

# def train_models(train_data, test_data, target_column, forecast_periods):
#     """
#     Train multiple time series models.
#     """
#     predictions = {}
    
#     # Debug information
#     st.write("Debug Info:")
#     st.write(f"Train data type: {type(train_data)}")
#     st.write(f"Train data shape: {train_data.shape if hasattr(train_data, 'shape') else len(train_data)}")
    
#     # Since we're working with a Series from prepare_time_series_data
#     train_series = train_data if isinstance(train_data, pd.Series) else train_data[target_column]
#     test_series = test_data if isinstance(test_data, pd.Series) else test_data[target_column]
    
#     # Prophet
#     try:
#         prophet_data = pd.DataFrame({
#             'ds': train_series.index,
#             'y': train_series.values
#         })
#         model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
#         model_prophet.fit(prophet_data)
#         future = model_prophet.make_future_dataframe(periods=forecast_periods)
#         forecast_prophet = model_prophet.predict(future)
#         predictions['Prophet'] = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#         st.success("Prophet model trained successfully!")
#     except Exception as e:
#         st.error(f"Prophet model error: {str(e)}")
#         st.write("Prophet data preview:", prophet_data.head() if 'prophet_data' in locals() else "No data preview available")

#     # SARIMA
#     try:
#         model_sarima = SARIMAX(train_series.values,
#                               order=(1,1,1),
#                               seasonal_order=(1,1,1,12))
#         results_sarima = model_sarima.fit(disp=False)
#         forecast_sarima = results_sarima.get_forecast(steps=forecast_periods)
#         predictions['SARIMA'] = pd.DataFrame({
#             'ds': pd.date_range(start=train_series.index[-1], 
#                               periods=forecast_periods+1,
#                               freq=train_series.index.freq)[1:],
#             'yhat': forecast_sarima.predicted_mean,
#             'yhat_lower': forecast_sarima.conf_int().iloc[:,0],
#             'yhat_upper': forecast_sarima.conf_int().iloc[:,1]
#         })
#         st.success("SARIMA model trained successfully!")
#     except Exception as e:
#         st.error(f"SARIMA model error: {str(e)}")
#         st.write("SARIMA input shape:", train_series.shape)

#     # XGBoost
#     try:
#         n_steps = 5
#         X_train, y_train = [], []
#         series_values = train_series.values
        
#         for i in range(len(series_values) - n_steps):
#             X_train.append(series_values[i:(i + n_steps)])
#             y_train.append(series_values[i + n_steps])
        
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
        
#         # Debug info for XGBoost
#         st.write("XGBoost preparation:")
#         st.write(f"X_train shape: {X_train.shape if len(X_train) > 0 else 'Empty'}")
#         st.write(f"y_train shape: {y_train.shape if len(y_train) > 0 else 'Empty'}")
        
#         if len(X_train) > 0:
#             model_xgb = xgb.XGBRegressor(
#                 objective='reg:squarederror',
#                 n_estimators=200,
#                 max_depth=5,
#                 learning_rate=0.05,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 random_state=42
#             )
#             model_xgb.fit(X_train, y_train)
            
#             # Prepare test data
#             X_test = []
#             test_values = test_series.values
#             for i in range(len(test_values) - n_steps):
#                 X_test.append(test_values[i:(i + n_steps)])
#             X_test = np.array(X_test)
            
#             xgb_pred = model_xgb.predict(X_test)
#             predictions['XGBoost'] = pd.DataFrame({
#                 'ds': test_series.index[n_steps:],
#                 'yhat': xgb_pred,
#                 'yhat_lower': xgb_pred * 0.9,
#                 'yhat_upper': xgb_pred * 1.1
#             })
#             st.success("XGBoost model trained successfully!")
#         else:
#             st.warning("Not enough data points for XGBoost training after sequence preparation")
#     except Exception as e:
#         st.error(f"XGBoost model error: {str(e)}")
    
#     return predictions
    
#     # Prophet
#     try:
#         prophet_data = pd.DataFrame({
#             'ds': train_data.index,
#             'y': train_series.values  # Use values directly
#         })
#         model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True)
#         model_prophet.fit(prophet_data)
#         future = model_prophet.make_future_dataframe(periods=forecast_periods, freq=train_data.index.freq)
#         forecast_prophet = model_prophet.predict(future)
#         predictions['Prophet'] = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
#         st.success("Prophet model trained successfully!")
#     except Exception as e:
#         st.error(f"Prophet model error: {str(e)}")
#         st.write("Prophet data preview:", prophet_data.head())

#     # SARIMA
#     try:
#         model_sarima = SARIMAX(train_series.values,  # Use values directly
#                               order=(1,1,1),
#                               seasonal_order=(1,1,1,12))
#         results_sarima = model_sarima.fit(disp=False)
#         forecast_sarima = results_sarima.get_forecast(steps=forecast_periods)
#         predictions['SARIMA'] = pd.DataFrame({
#             'ds': pd.date_range(start=train_data.index[-1], 
#                               periods=forecast_periods+1,
#                               freq=train_data.index.freq)[1:],
#             'yhat': forecast_sarima.predicted_mean,
#             'yhat_lower': forecast_sarima.conf_int().iloc[:,0],
#             'yhat_upper': forecast_sarima.conf_int().iloc[:,1]
#         })
#         st.success("SARIMA model trained successfully!")
#     except Exception as e:
#         st.error(f"SARIMA model error: {str(e)}")
#         st.write("SARIMA input shape:", train_series.shape)

#     # XGBoost
#     try:
#         n_steps = 5
#         X_train, y_train = [], []
#         series_values = train_series.values
        
#         for i in range(len(series_values) - n_steps):
#             X_train.append(series_values[i:(i + n_steps)])
#             y_train.append(series_values[i + n_steps])
        
#         X_train = np.array(X_train)
#         y_train = np.array(y_train)
        
#         # Debug info for XGBoost
#         st.write(f"XGBoost training data shape: X={X_train.shape}, y={y_train.shape}")
        
#         model_xgb = xgb.XGBRegressor(
#             objective='reg:squarederror',
#             n_estimators=200,
#             max_depth=5,
#             learning_rate=0.05,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             random_state=42
#         )
#         model_xgb.fit(X_train, y_train)
        
#         # Prepare test data
#         X_test = []
#         test_values = test_series.values
#         for i in range(len(test_values) - n_steps):
#             X_test.append(test_values[i:(i + n_steps)])
#         X_test = np.array(X_test)
        
#         xgb_pred = model_xgb.predict(X_test)
#         predictions['XGBoost'] = pd.DataFrame({
#             'ds': test_data.index[n_steps:],
#             'yhat': xgb_pred,
#             'yhat_lower': xgb_pred * 0.9,
#             'yhat_upper': xgb_pred * 1.1
#         })
#         st.success("XGBoost model trained successfully!")
#     except Exception as e:
#         st.error(f"XGBoost model error: {str(e)}")
#         if len(X_train) > 0:
#             st.write("XGBoost input shapes:", 
#                     f"X_train: {X_train.shape}",
#                     f"y_train: {y_train.shape}")
    
#     return predictions

# def plot_forecasts(original_df, forecasts_dict):
#     """
#     Create interactive plot with forecasts from all models.
#     """
#     fig = go.Figure()

#     # Plot original data
#     fig.add_trace(go.Scatter(
#         x=original_df.index,
#         y=original_df.values,
#         name='Actual Data',
#         mode='lines'
#     ))

#     colors = ['rgba(255,0,0,0.3)', 'rgba(0,255,0,0.3)', 'rgba(0,0,255,0.3)']
#     for (model_name, forecast), color in zip(forecasts_dict.items(), colors):
#         fig.add_trace(go.Scatter(
#             x=forecast['ds'],
#             y=forecast['yhat'],
#             name=f'{model_name} Forecast',
#             mode='lines'
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
#             y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
#             fill='toself',
#             fillcolor=color,
#             line=dict(color='rgba(255,255,255,0)'),
#             name=f'{model_name} Confidence Interval'
#         ))

#     fig.update_layout(
#         title='Time Series Forecasts',
#         xaxis_title='Date',
#         yaxis_title='Value',
#         height=600,
#         showlegend=True,
#         template='plotly_white'
#     )
    
#     return fig

# def perform_time_series_analysis(current_df):
#     """
#     Main function for time series analysis interface
#     """
#     st.header("📈 Time Series Analysis")
    
#     with st.expander("ℹ️ Important Information", expanded=True):
#         st.markdown("""
#         ### Requirements:
#         1. **Date/Time Column**: Valid dates/timestamps, no missing values
#         2. **Target Column**: Numeric values, no missing values
#         3. **Data**: Minimum 30 points, chronological order
        
#         ### Models Used:
#         - Prophet (Facebook/Meta)
#         - SARIMA (Seasonal ARIMA)
#         - XGBoost
#         """)
    
#     current_df = current_df.to_pandas()

#     # Column selection
#     col1, col2 = st.columns(2)
#     with col1:
#         date_column = st.selectbox("Select Date/Time Column", options=current_df.columns)
#     with col2:
#         target_column = st.selectbox("Select Target Column", options=current_df.columns)

#     try:
#         # Debug information
#         st.write("Debug Info:")
#         st.write(f"Data type: {type(current_df)}")
#         st.write(f"Columns: {current_df.columns.tolist() if hasattr(current_df, 'columns') else 'No columns (Series)'}")
        
#         # Analyze pattern
#         pattern_info = analyze_time_series_pattern(current_df, date_column)
        
#         # Display pattern analysis
#         st.subheader("📊 Data Pattern Analysis")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Frequency", pattern_info['frequency'].title())
#             st.metric("Total Records", pattern_info['total_records'])
#         with col2:
#             st.metric("Unique Dates", pattern_info['unique_dates'])
#             st.metric("Regular Intervals", "Yes" if pattern_info['regular_intervals'] else "No")
#         with col3:
#             st.metric("Has Duplicates", "Yes" if pattern_info['has_duplicates'] else "No")
#     except Exception as e:
#         st.error(f"❌ Error in pattern analysis: {str(e)}")
#         st.warning("Unable to analyze time series patterns. Please check your data format.")

#     # Processing options
#     with st.expander("⚙️ Processing Options", expanded=True):
#         resample_freq = st.selectbox(
#             "Time Frequency",
#             options=[None, 'D', 'W', 'M', 'Q', 'Y'],
#             format_func=lambda x: 'No resampling' if x is None else f'{x} (Daily/Weekly/Monthly/Quarterly/Yearly)'
#         )
        
#         agg_method = st.selectbox(
#             "Aggregation Method",
#             options=['mean', 'sum', 'max', 'min', 'first', 'last']
#         )
        
#         forecast_periods = st.slider(
#             "Forecast Periods",
#             min_value=1,
#             max_value=365,
#             value=30
#         )

#     if st.button("🎯 Run Analysis"):
#         with st.spinner("Analyzing data..."):
#             # Validate
#             errors = validate_time_series_data(current_df, date_column, target_column)
#             if errors:
#                 st.error("Please fix the following:")
#                 for error in errors:
#                     st.warning(error)
#                 return

#             # Prepare data
#             train_data, test_data, processed_df = prepare_time_series_data(
#                 current_df,
#                 date_column,
#                 target_column,
#                 resample_freq,
#                 agg_method
#             )

#             # Train models and get predictions
#             forecasts = train_models(train_data, test_data, target_column, forecast_periods)

#             if forecasts:
#                 # Create tabs for results
#                 tab1, tab2 = st.tabs(["📈 Visualization", "📊 Model Performance"])
                
#                 with tab1:
#                     fig = plot_forecasts(processed_df[target_column], forecasts)
#                     st.plotly_chart(fig, use_container_width=True)
                    
#                     st.write("### 📝 Forecast Interpretation")
#                     st.write("""
#                     - The graph shows actual historical data and predictions from multiple models
#                     - Each model's forecast is shown with a confidence interval
#                     - Different colors represent different models' predictions
#                     - The shaded areas show the uncertainty range for each model
#                     """)
                
#                 with tab2:
#                     st.write("### Model Performance Metrics")
                    
#                     # Calculate and display metrics for each model
#                     metrics_dict = {}
#                     for model_name, forecast in forecasts.items():
#                         # Get overlapping dates between actual and predictions
#                         common_dates = set(processed_df.index).intersection(set(forecast['ds']))
#                         if common_dates:
#                             actual = processed_df[processed_df.index.isin(common_dates)][target_column]
#                             pred = forecast[forecast['ds'].isin(common_dates)]['yhat']
                            
#                             metrics = {
#                                 'Mean Squared Error': mean_squared_error(actual, pred),
#                                 'Root Mean Squared Error': np.sqrt(mean_squared_error(actual, pred)),
#                                 'Mean Absolute Error': mean_absolute_error(actual, pred),
#                                 'R-squared Score': r2_score(actual, pred)
#                             }
#                             metrics_dict[model_name] = metrics
                    
#                     # Create and display metrics DataFrame
#                     metrics_df = pd.DataFrame(metrics_dict).round(4)
#                     st.dataframe(metrics_df, use_container_width=True)
                    
#                     # Add metrics interpretation
#                     st.info("""
#                     💡 Metrics Interpretation:
#                     - MSE (Mean Squared Error): Lower is better
#                     - RMSE (Root Mean Squared Error): Lower is better, same units as data
#                     - MAE (Mean Absolute Error): Lower is better, same units as data
#                     - R-squared: Higher is better (0-1 range), indicates fit quality
#                     """)
                    
#                     # Download results
#                     for model_name, forecast in forecasts.items():
#                         forecast_csv = forecast.to_csv(index=False)
#                         st.download_button(
#                             f"Download {model_name} Forecast",
#                             forecast_csv,
#                             f"{model_name.lower()}_forecast.csv",
#                             "text/csv",
#                             key=f'download_{model_name}'
#                         )