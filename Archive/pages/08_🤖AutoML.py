import streamlit as st
import polars as pl
from sklearn.model_selection import train_test_split
from utilities.eval import JarvisPredict, JarvisClassify
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from utilities.select_data_types import SelectDataTypes
from utilities.usage_profiler import monitor
import plotly.graph_objects as go
import pandas as pd
from utilities.timeseries import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utilities.utils import sample_dataframe_with_outliers
from utilities.target_validation import check_and_handle_target


import time
cache_buster = int(time.time())



def initialize_page():
    """Initialize page and check requirements"""
    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("📤 Please upload a dataset to get started.")
        st.stop()
        
    return (st.session_state.df,
            st.session_state.get('df_processed'),
            st.session_state.get('df_scaled'),
            st.session_state.get('df_encoded'))

def create_header():
    """Create an attractive header section"""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/brain.gif", use_container_width=True)
    with col2:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>🤖 Machine Learning Explorer</h2>
            <p style='font-size: 1.1em;'>Explore regression, classification, and clustering analysis. 
            Evaluate model performance and get insights into your data.</p>
        </div>
        """, unsafe_allow_html=True)

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



def check_data_quality(df):
    """Check data for issues"""
    # Convert to pandas if it's a Polars DataFrame
    if isinstance(df, pl.DataFrame):
        pandas_df = df.to_pandas()
        has_nulls = pandas_df.isnull().any().any()
        numeric_cols = len(pandas_df.select_dtypes(include=['float64', 'int64']).columns)
    else:
        # Already a pandas DataFrame
        has_nulls = df.isnull().any().any()
        numeric_cols = len(df.select_dtypes(include=['float64', 'int64']).columns)
    
    return has_nulls, numeric_cols


def should_use_direct_preprocessing(selected_state):
    """
    Determines whether to use direct preprocessing from eval.py
    or continue with the app's step-by-step preprocessing.
    """
    if selected_state == "Initial DataFrame":
        # Add checkbox to let users choose preprocessing approach
        use_direct = st.checkbox(
            "🔄 Use automatic preprocessing from eval.py", 
            value=True,
            help="When checked, the data will be processed automatically by the model's pipeline. "
                 "This handles missing values, encoding, and scaling in one step."
        )
        
        if use_direct:
            st.info("""
            ℹ️ **Using automatic preprocessing from eval.py**
            
            The following steps will be performed automatically:
            - Missing values in numeric columns will be imputed with mean values
            - Missing values in categorical columns will be imputed with 'missing'
            - Numeric features will be standardized (zero mean, unit variance)
            - Low-cardinality categorical features (<= 11 unique values) will be one-hot encoded
            - High-cardinality categorical features (> 11 unique values) will be ordinal encoded
            
            This is equivalent to the manual preprocessing steps but happens inside the model pipeline.
            """)
            
        return use_direct
    
    # For other data sources, use the app's preprocessing approach
    return False




def run_model_with_progress(model_type, X_train, X_test, y_train, y_test, use_direct_preprocessing=False):
    """Run model with improved loading indicators"""
    start_time = time.time()
    
    # Create a placeholder for the progress section
    progress_section = st.empty()
    
    try:
        with st.spinner('🤖 Initializing model...'):
            if model_type == "regression":
                model = JarvisPredict(verbose=1, ignore_warnings=use_direct_preprocessing, predictions=True)
                analysis_type = "Regression"
            else:  # classification
                model = JarvisClassify(verbose=1, ignore_warnings=True, predictions=True)
                analysis_type = "Classification"
        
        # Rest of the function remains the same
        with progress_section.container():
            st.info(f"🔄 Running {analysis_type} Analysis...")
            
            # Progress bar
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            # Model training phases
            phases = [
                "Loading data...",
                "Preprocessing features...",
                "Training models...",
                "Evaluating performance...",
                "Generating results..."
            ]
            
            # Simulate progress through phases
            for idx, phase in enumerate(phases):
                progress_text.text(f"Phase {idx + 1}/{len(phases)}: {phase}")
                progress_bar.progress((idx * 20) + 10)
                time.sleep(0.5)  # Brief delay for visual feedback
            
            with st.spinner('🎯 Finalizing model evaluation...'):
                scores, predictions = model.fit(X_train, X_test, y_train, y_test)
            
            progress_bar.progress(100)
            progress_text.text("Analysis Complete!")
        
        # Clear the progress section
        progress_section.empty()
        
        if scores.empty:
            st.error("❌ No results returned!")
            st.warning("""
            This might be due to:
            - Problematic columns (e.g., text data)
            - Columns needing scaling/encoding
            - Non-numeric data requiring preprocessing
            
            Please check your selected columns and ensure appropriate preprocessing.
            """)
        else:
            execution_time = time.time() - start_time
            st.success("✅ Analysis completed successfully!")
            
            # Results section
            with st.expander("📊 Detailed Results", expanded=True):
                st.markdown(f"### {analysis_type} Results")
                st.dataframe(scores, use_container_width=True)
                st.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")
        
        return scores, predictions
    except Exception as e:
        progress_section.empty()
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        return None, None


def prepare_data_for_modeling(df, target_column, is_initial=False):
    """Prepare data for modeling"""
    if is_initial and isinstance(df, pl.DataFrame):
        df_numeric = SelectDataTypes(df)
        if len(df_numeric.columns) == 0:
            st.warning("⚠️ No numeric features available. Please encode categorical features first.")
            return None, None
            
        if target_column in df_numeric.columns:
            X = df_numeric.drop(columns=[target_column])
            y = df_numeric[target_column]
        else:
            X = df_numeric
            y = df[target_column].to_pandas()
    else:
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()
        X = df.drop(columns=[target_column], axis=1)
        y = df[target_column]
    
    return X, y

def perform_clustering(df_numeric, n_clusters):
    """Perform KMeans clustering"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_numeric)
    return kmeans.labels_, kmeans.inertia_

def plot_clustering_metrics(df_numeric, K_range=range(2, 11)):
    """Plot clustering metrics"""
    inertia = []
    silhouette = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(df_numeric)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(df_numeric, kmeans.labels_))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(K_range), 
        y=inertia, 
        name='Inertia',
        mode='lines+markers',
        line=dict(color='firebrick'),
        yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=list(K_range),
        y=silhouette,
        name='Silhouette Score',
        mode='lines+markers',
        line=dict(color='royalblue'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title_text='Elbow Method and Silhouette Analysis',
        xaxis=dict(title='Number of Clusters'),
        yaxis=dict(title='Inertia', color='firebrick'),
        yaxis2=dict(
            title='Silhouette Score',
            color='royalblue',
            overlaying='y',
            side='right'
        )
    )
    
    return fig


def perform_pca_clustering(df_numeric, n_clusters):
    """
    Perform PCA followed by KMeans clustering
    
    Parameters:
    df_numeric (pd.DataFrame): Numeric dataframe to cluster
    n_clusters (int): Number of clusters to create
    
    Returns:
    tuple: (cluster_labels, pca_df, explained_variance_ratio, fig)
        - cluster_labels: Array of cluster assignments
        - pca_df: DataFrame with PCA components
        - explained_variance_ratio: Explained variance by PCA components
        - fig: Plotly figure showing clusters in PCA space
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=['PCA1', 'PCA2']
    )
    
    # Perform clustering on PCA results
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(pca_result)
    
    # Create visualization
    fig = px.scatter(
        pca_df,
        x='PCA1',
        y='PCA2',
        color=cluster_labels,
        color_continuous_scale="viridis",
        title=f'Clusters in PCA Space (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})'
    )
    
    fig.update_layout(
        height=600,
        title_x=0.5,
        title_font_size=20,
        coloraxis_showscale=True,
        coloraxis_colorbar_title='Cluster'
    )
    
    # Add cluster centers to the plot
    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=['PCA1', 'PCA2']
    )
    
    fig.add_trace(
        go.Scatter(
            x=centers['PCA1'],
            y=centers['PCA2'],
            mode='markers',
            marker=dict(
                color='red',
                size=15,
                symbol='x'
            ),
            name='Cluster Centers'
        )
    )
    
    return cluster_labels, pca_df, pca.explained_variance_ratio_, fig

def plot_pca_explained_variance(df_numeric, max_components=10):
    """
    Plot explained variance ratio for different numbers of PCA components
    
    Parameters:
    df_numeric (pd.DataFrame): Numeric dataframe to analyze
    max_components (int): Maximum number of components to consider
    
    Returns:
    go.Figure: Plotly figure showing explained variance ratio
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Calculate explained variance for different numbers of components
    pca = PCA(n_components=min(max_components, df_numeric.shape[1]))
    pca.fit(scaled_data)
    
    # Create cumulative sum of explained variance
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    
    # Create figure
    fig = go.Figure()
    
    # Add individual explained variance bars
    fig.add_trace(go.Bar(
        x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
        y=pca.explained_variance_ratio_,
        name='Individual',
        marker_color='royalblue'
    ))
    
    # Add cumulative explained variance line
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_variance_ratio) + 1)),
        y=cumulative_variance_ratio,
        name='Cumulative',
        mode='lines+markers',
        line=dict(color='firebrick')
    ))
    
    fig.update_layout(
        title='Explained Variance Ratio by PCA Components',
        xaxis_title='Number of Components',
        yaxis_title='Explained Variance Ratio',
        yaxis_tickformat='.0%',
        showlegend=True,
        height=500
    )
    
    return fig


def main():
    # Initialize page
    df, df_processed, df_scaled, df_encoded = initialize_page()
    
    # Create header
    create_header()
    
    # Create analysis selection
    col1, col2 = st.columns(2)
    with col1:
        selected_state = st.selectbox(
            "📂 Select Data Source",
            ["Initial DataFrame", 
             "DataFrame after Missing value Imputation",
             "DataFrame after Feature Scaling",
             "DataFrame after Feature Encoding"]
        )
    
    with col2:
        analysis_type = st.selectbox(
            "🔬 Select Analysis Type",
            ["Regression Analysis", "Classification Analysis", "Clustering Analysis", "Time Series Analysis"],
            help="Choose the type of machine learning analysis to perform"
        )
    
    # Get selected data
    current_df = get_selected_data(selected_state, df, df_processed, df_scaled, df_encoded)
    
    # Display current data
    st.markdown("### 📊 Current Dataset")
    st.dataframe(current_df.head(), use_container_width=True)
    
    # Column selection
    selected_columns = st.multiselect(
        "🎯 Select Features for the analysis including the `Target variable`",
        current_df.columns,
        help="Choose the columns to include in your analysis"
    )
    
    if not selected_columns:
        st.info("Please select columns to proceed with analysis.")
        return
    
    current_df = current_df[selected_columns]
    
    if analysis_type in ["Regression Analysis", "Classification Analysis"]:
        target_column = st.selectbox(
            "🎯 Select Target Variable", 
            selected_columns,
            help="Choose the variable you want to predict"
        )
        
        if not target_column:
            st.info("Please select a target variable.")
            return
        #----------------------------------------------------
        
        # Check if target has missing values
        sample_df, had_missing_target, message = check_and_handle_target(
            current_df, 
            target_column, 
            analysis_type
        )
        
        # If target had missing values, show a warning and the imputation message
        if had_missing_target:
            st.warning(f"⚠️ Missing values detected in target variable '{target_column}'")
            st.info(message)
            
            # Offer to continue with the imputed values or stop
            continue_with_imputed = st.radio(
                "How would you like to proceed?",
                ["Continue with imputed values", "Stop and fix missing values manually"],
                index=0,
                help="Choose whether to continue with automatically imputed values or stop to fix the data manually"
            )
            
            if continue_with_imputed == "Stop and fix missing values manually":
                st.info("Please fix the missing values in your target variable and try again.")
                return
            else:
                st.success("Continuing with imputed target values.")
                # Update the current_df with the fixed target
                current_df = sample_df

        #-----------------------------------------------------
        sample_df = sample_dataframe_with_outliers(current_df)
        X, y = prepare_data_for_modeling(
            sample_df, 
            target_column, 
            is_initial=(selected_state == "Initial DataFrame")
        )
        
        if X is None or y is None:
            return
        
        # Data split info
        st.markdown("### 📊 Data Split Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Training Data: 80%")
        with col2:
            st.info("Testing Data: 20%")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=70)
        
        
        # Then replace your regression analysis button click handler with this code:
        if analysis_type == "Regression Analysis":
            if st.button("🎯 Run Regression Analysis", use_container_width=True):
                # Check if we should use direct preprocessing
                use_direct = should_use_direct_preprocessing(selected_state)
                
                # If using direct preprocessing, skip the missing values check
                if not use_direct:
                    has_nulls, _ = check_data_quality(current_df)
                    if has_nulls:
                        st.warning("⚠️ Dataset contains missing values. Please handle them first!")
                        return
                # Pass the use_direct parameter to the function
                run_model_with_progress("regression", X_train, X_test, y_train, y_test, use_direct_preprocessing=use_direct)
            

        # And similarly for classification analysis:
        elif analysis_type == "Classification Analysis":
            if st.button("🎯 Run Classification Analysis", use_container_width=True):
                # Check if we should use direct preprocessing
                use_direct = should_use_direct_preprocessing(selected_state)
                
                # If using direct preprocessing, skip the missing values check
                if not use_direct:
                    has_nulls, _ = check_data_quality(current_df)
                    if has_nulls:
                        st.warning("⚠️ Dataset contains missing values. Please handle them first!")
                        return
                
                # Pass the use_direct parameter to the function
                run_model_with_progress("classification", X_train, X_test, y_train, y_test, use_direct_preprocessing=use_direct)

    elif analysis_type == "Time Series Analysis":
        #sample_df = sample_dataframe_with_outliers(current_df)
        #perform_time_series_analysis(sample_df)
        st.markdown("### Build in Progress! Coming soon...")
        st.image("images/comingsoon.gif", use_container_width=True)

    else: 
        # Clustering Analysis
        st.header("🔍 Clustering Analysis")
        st.write("""
        Clustering helps you discover natural groups in your data. 
        Follow these steps to perform clustering analysis:
        """)
        
        # Get numeric columns for clustering
        sample_df = sample_dataframe_with_outliers(current_df)
        df_numeric = SelectDataTypes(sample_df)
        if df_numeric.shape[1] < 2:
            st.error("⚠️ Clustering requires at least 2 numeric columns. Please select more numeric features.")
            return
            
        # Step 1: Choose Clustering Method
        st.subheader("Step 1: Choose Clustering Method")
        clustering_method = st.radio(
            "Select how you want to perform clustering:",
            ["Standard Clustering", "PCA Clustering"],
            help="Standard clustering uses original features, PCA clustering reduces dimensions before clustering"
        )
        try:
            if clustering_method == "PCA Clustering":
                st.info("""
                💡 About PCA Clustering:
                - PCA (Principal Component Analysis) reduces the complexity of your data
                - It combines your features into new components that capture the most important patterns
                - This can help reveal clusters that might be hidden in high-dimensional data
                """)
                
                # Show PCA explained variance plot
                with st.spinner("Analyzing PCA components..."):
                    pca_fig = plot_pca_explained_variance(df_numeric)
                    st.plotly_chart(pca_fig, use_container_width=True)
                    st.info("""
                    💡 How to interpret the PCA plot:
                    - Bars show how much information each component captures
                    - Line shows the cumulative information captured
                    - Higher percentage means better representation of your data
                    """)
        except Exception as e:
            st.error(f"❌ Please check if you have selected the right dataset. If the problem still persists please resolve this issue.\n\n {str(e)}")
        
        # Step 2: Cluster Number Selection
        st.subheader("Step 2: Choose Number of Clusters")
        show_recommendations = st.checkbox(
            '📊 Show recommended number of clusters',
            help="View a graph that helps you choose the optimal number of clusters"
        )
        
        if show_recommendations:
            with st.spinner("Calculating cluster recommendations..."):
                fig = plot_clustering_metrics(df_numeric)
                st.plotly_chart(fig, use_container_width=True)
                st.info("""
                💡 How to interpret the graph:
                - Look for the 'elbow' point in the blue line - this suggests a good number of clusters
                - Higher silhouette score (orange line) indicates better-defined clusters
                """)
        
        n_clusters = st.slider(
            'How many groups do you want to divide your data into?',
            min_value=2,
            max_value=min(20, len(current_df) - 1),
            value=3,
            help="Choose the number of clusters you want to create"
        )
        
        # Step 3: Feature Selection (for visualization)
        if clustering_method == "Standard Clustering":
            st.subheader("Step 3: Select Features to Visualize")
            st.info("""
            💡 Tips for selecting visualization features:
            - Choose columns that you think might help distinguish between different groups
            - Good choices include: measurements, quantities, or calculated metrics
            - Features with high variation between groups but low variation within groups work best
            - Examples: customer spending patterns, product dimensions, performance metrics
            """)

            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox(
                    "Choose X-axis feature",
                    options=df_numeric.columns,
                    help="Select a numeric feature for the horizontal axis. Choose features that might show distinct patterns between groups."
                )
            with col2:
                y_axis = st.selectbox(
                    "Choose Y-axis feature",
                    options=df_numeric.columns,
                    help="Select a numeric feature for the vertical axis. Choose a different feature that might help separate the groups."
                )
        
        # Step 4: Run Clustering
        if st.button('✨ Generate Clusters', use_container_width=True):
            with st.spinner("Creating clusters..."):
                try:
                    if clustering_method == "Standard Clustering":
                        # Perform standard clustering
                        labels, inertia = perform_clustering(df_numeric, n_clusters)
                        
                        # Prepare results DataFrame
                        results_df = current_df.clone() if isinstance(current_df, pl.DataFrame) else current_df.copy()
                        if isinstance(results_df, pl.DataFrame):
                            results_df = results_df.to_pandas()
                        
                        # Add cluster labels
                        results_df['Cluster Group'] = labels + 1
                        
                        # Display results
                        st.success("✅ Clustering complete! Your data has been divided into groups.")
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["📊 Cluster Visualization", "📋 Detailed Data"])
                        
                        with tab1:
                            # Scatter plot
                            fig = px.scatter(
                                results_df,
                                x=x_axis,
                                y=y_axis,
                                color='Cluster Group',
                                color_continuous_scale="viridis",
                                hover_data=results_df.columns,
                                title=f"Cluster Visualization: {x_axis} vs {y_axis}"
                            )
                            fig.update_layout(
                                height=600,
                                title_x=0.5,
                                title_font_size=20
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Cluster statistics
                            st.subheader("Cluster Statistics")
                            cluster_stats = pd.DataFrame()
                            for feature in [x_axis, y_axis]:
                                feature_stats = results_df.groupby('Cluster Group')[feature].agg(['mean', 'min', 'max']).round(2)
                                feature_stats.columns = [f'{feature} ({col})' for col in feature_stats.columns]
                                if cluster_stats.empty:
                                    cluster_stats = feature_stats
                                else:
                                    cluster_stats = pd.concat([cluster_stats, feature_stats], axis=1)
                            
                            cluster_counts = results_df.groupby('Cluster Group').size().to_frame('Points in Cluster')
                            cluster_stats = pd.concat([cluster_counts, cluster_stats], axis=1)
                            st.dataframe(cluster_stats, use_container_width=True)
                    
                    else:  # PCA Clustering
                        # Perform PCA clustering
                        labels, pca_df, explained_variance, pca_fig = perform_pca_clustering(df_numeric, n_clusters)
                        
                        # Prepare results DataFrame
                        results_df = current_df.clone() if isinstance(current_df, pl.DataFrame) else current_df.copy()
                        if isinstance(results_df, pl.DataFrame):
                            results_df = results_df.to_pandas()
                        results_df['Cluster Group'] = labels + 1
                        
                        # Display results
                        st.success("✅ PCA Clustering complete! Your data has been divided into groups.")
                        
                        # Create tabs for different views
                        tab1, tab2 = st.tabs(["📊 PCA Visualization", "📋 Detailed Data"])
                        
                        with tab1:
                            # PCA scatter plot
                            st.plotly_chart(pca_fig, use_container_width=True)
                            st.info(f"""
                            💡 PCA Results:
                            - Total variance explained by 2 components: {explained_variance.sum():.2%}
                            - First component (PCA1) explains: {explained_variance[0]:.2%}
                            - Second component (PCA2) explains: {explained_variance[1]:.2%}
                            """)
                            
                            # Cluster statistics in PCA space
                            st.subheader("Cluster Statistics")
                            pca_df['Cluster Group'] = labels + 1
                            cluster_stats = pd.DataFrame()
                            for feature in ['PCA1', 'PCA2']:
                                feature_stats = pca_df.groupby('Cluster Group')[feature].agg(['mean', 'min', 'max']).round(2)
                                feature_stats.columns = [f'{feature} ({col})' for col in feature_stats.columns]
                                if cluster_stats.empty:
                                    cluster_stats = feature_stats
                                else:
                                    cluster_stats = pd.concat([cluster_stats, feature_stats], axis=1)
                            
                            cluster_counts = pca_df.groupby('Cluster Group').size().to_frame('Points in Cluster')
                            cluster_stats = pd.concat([cluster_counts, cluster_stats], axis=1)
                            st.dataframe(cluster_stats, use_container_width=True)
                    
                    # Common tab for both methods
                    with tab2:
                        st.subheader("Data with Cluster Assignments")
                        st.dataframe(
                            results_df.style.background_gradient(
                                subset=['Cluster Group'],
                                cmap='viridis'
                            ),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "💾 Download Results",
                            csv,
                            "clustered_data.csv",
                            "text/csv",
                            help="Download the data with cluster assignments"
                        )
                
                except Exception as e:
                    st.error(f"❌ An error occurred during clustering: {str(e)}")
                    st.info("Please check your data and try again with different parameters.")

if __name__ == "__main__":
    main()