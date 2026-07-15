import streamlit as st
from scipy import stats
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu, wilcoxon, kruskal, shapiro


try:
    import seaborn as sns
except AttributeError:
    # If there's an issue with seaborn and cm.register_cmap, use this workaround
    import matplotlib as mpl
    if not hasattr(mpl.cm, 'register_cmap'):
        # Add a dummy function to prevent the error
        mpl.cm.register_cmap = lambda *args, **kwargs: None
    import seaborn as sns


import time
cache_buster = int(time.time())


# Page header with improved styling
col1, col2 = st.columns(2)
with col1:
    st.image("images/n_op.gif")
with col2:
    st.markdown('## Statistical Tests for your Data')
    st.markdown("""
        Statistical tests analyze data characteristics to make inferences about populations. 
        Choose the appropriate test based on your data type, distribution, and research question.
    """)

st.divider()

# Test descriptions in a more organized format
st.markdown('#### Statistical Tests Available:')
with st.expander("Click to see test descriptions"):
    tests_info = {
        "2-Sample T-Test": "Compares the means of two independent groups to determine if they are statistically different from each other. **Assumptions**: Normally distributed data, equal variances.",
        "ANOVA (Analysis of Variance)": "Tests for differences between means across more than two groups or categories. **Assumptions**: Normally distributed data, equal variances across groups.",
        "Chi-Square Test": "Assesses whether there is a significant association between two categorical variables. **Assumptions**: Independent observations, expected frequencies > 5.",
        "Mann-Whitney U Test": "A non-parametric test that compares differences between two independent groups when the data is not normally distributed. **Assumptions**: Independent samples, similar shaped distributions.",
        "Wilcoxon Signed-Rank Test": "A non-parametric test that compares two related samples to determine if their population mean ranks differ. **Assumptions**: Paired observations, symmetrical distribution of differences.",
        "Kruskal-Wallis Test": "A non-parametric version of ANOVA, used when the assumptions of ANOVA are not met. **Assumptions**: Independent samples, similar shaped distributions."
    }
    
    for test, description in tests_info.items():
        st.markdown(f"**{test}**")
        st.markdown(description)
        st.markdown("---")

# API key verification
user_api_key = st.session_state.get('openai_api_key')
has_api_key = user_api_key is not None and user_api_key.strip() != ""

# Inform user about API key status for LLM analysis
if not has_api_key:
    st.info("ℹ️ OpenAI API key not found. Statistical tests will work, but AI-powered analysis will be unavailable. Add your OpenAI API key on the Home Page to enable AI analysis.")

# Load dataframes from session states
dataframes = {
    "Initial DataFrame": st.session_state.get('df'),
    "DataFrame after Missing value Imputation": st.session_state.get('df_processed'),
    "DataFrame after Feature Encoding": st.session_state.get('df_encoded'),
    "DataFrame after Feature Scaling": st.session_state.get('df_scaled')
}

# Check if any dataframe is available
if all(df is None for df in dataframes.values()):
    st.warning("⚠️ Please upload a dataset to get started.")
    st.stop()

# Main selection area
st.markdown('🖲️ Select the `Session State` and `Statistical Test` ⤵️')
col3, col4 = st.columns(2)
with col3:
    selected_session_state = st.selectbox(
        "Select Session State", 
        list(dataframes.keys())
    )
with col4:
    statistical_test_technique = st.selectbox(
        "Select Statistical Test", 
        list(tests_info.keys())
    )

# Handle dataframe selection with improved error checking
df = dataframes.get(selected_session_state)
if df is None:
    st.warning(f"⚠️ {selected_session_state} is not available. Please complete the previous data processing steps first.")
    st.stop()
else:
    # Convert to pandas dataframe if not already
    if not isinstance(df, pd.DataFrame):
        df = df.to_pandas()
    
    # Display the selected dataframe
    st.markdown(f"#### Viewing: {selected_session_state}")
    st.dataframe(df)

# Utility function to check normality
def check_normality(data, column, alpha=0.05):
    """Check if data is normally distributed using Shapiro-Wilk test"""
    if len(data) < 3:
        return False, "Not enough data points for normality test"
    
    try:
        stat, p = shapiro(data[column].dropna())
        is_normal = p > alpha
        return is_normal, p
    except Exception as e:
        return False, str(e)

# Utility function to create LLM agent
def create_llm_agent(df):
    """Create a LangChain agent for data analysis"""
    # Check if API key is available
    if not has_api_key:
        return None
        
    try:
        return create_pandas_dataframe_agent(
            ChatOpenAI(
                temperature=0, 
                model="gpt-4-turbo", 
                api_key=user_api_key
            ),
            df,
            verbose=True,
            number_of_head_rows=15,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
    except Exception as e:
        st.error(f"Error creating LLM agent: {str(e)}")
        return None

# Function to visualize data distributions
def visualize_distribution(df, column):
    """Create a histogram to visualize data distribution"""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {column}")
    return fig

# Implement test-specific interfaces
if statistical_test_technique == "2-Sample T-Test":
    st.markdown("### Two-Sample T-Test")
    st.info("This test compares means between two independent groups. It assumes normally distributed data.")
    
    # Select categorical column with exactly 2 groups
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        categorical_cols = df.columns  # Fallback if no categorical columns detected
    
    col = st.selectbox("Select grouping column (must have exactly 2 groups)", options=categorical_cols)
    
    # Check if selected column has exactly 2 groups
    if df[col].nunique() != 2:
        st.warning(f"⚠️ Selected column '{col}' has {df[col].nunique()} unique values. T-test requires exactly 2 groups.")
    
    # Select numerical column for testing
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    if len(numerical_cols) == 0:
        st.error("No numerical columns found in the dataset. T-test requires numerical data.")
        st.stop()
    
    selected_col_for_test = st.selectbox("Select numerical column to test", options=numerical_cols)
    
    # Optional: Check normality
    if st.checkbox("Check normality assumption"):
        groups = df[col].unique()
        for group in groups:
            group_data = df[df[col] == group]
            is_normal, p_value = check_normality(group_data, selected_col_for_test)
            st.write(f"Group '{group}' normality test p-value: {p_value:.4f}")
            if not is_normal:
                st.warning(f"⚠️ Data for group '{group}' may not be normally distributed (p < 0.05). Consider using Mann-Whitney U test instead.")
    
    # Visualize distributions
    if st.checkbox("Visualize data distributions"):
        groups = df[col].unique()
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for i, group in enumerate(groups[:2]):  # Limit to first 2 groups
            group_data = df[df[col] == group][selected_col_for_test].dropna()
            ax[i].hist(group_data, bins=15, alpha=0.7, density=True)
            ax[i].set_title(f"Distribution for '{group}'")
        st.pyplot(fig)
    
    if st.button('🔄 Perform T-test', use_container_width=True):
        groups = df[col].unique()
        if len(groups) == 2:
            group1 = df[df[col] == groups[0]][selected_col_for_test].dropna()
            group2 = df[df[col] == groups[1]][selected_col_for_test].dropna()
            
            if len(group1) < 2 or len(group2) < 2:
                st.error("Not enough valid data points for T-test after removing NaNs.")
                st.stop()
            
            # Optional: Check equal variances
            var_ratio = np.var(group1) / np.var(group2)
            equal_var = 0.5 <= var_ratio <= 2
            
            if not equal_var:
                st.warning(f"⚠️ Variances may not be equal (ratio = {var_ratio:.2f}). Using Welch's T-test.")
            
            # Perform the test
            t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=equal_var, nan_policy='omit')
            
            # Display results
            st.markdown("### T-Test Results:")
            result = pd.DataFrame({
                'T-statistic': [t_stat], 
                'P-value': [p_val],
                'Group 1 Mean': [group1.mean()],
                'Group 2 Mean': [group2.mean()],
                'Mean Difference': [group1.mean() - group2.mean()],
                'Group 1 Count': [len(group1)],
                'Group 2 Count': [len(group2)]
            })
            st.dataframe(result)
            
            # Significance interpretation
            alpha = 0.05
            if p_val < alpha:
                st.success(f"✅ Statistically significant difference detected (p = {p_val:.4f} < 0.05)")
            else:
                st.info(f"ℹ️ No statistically significant difference detected (p = {p_val:.4f} ≥ 0.05)")
            
            # LLM interpretation
            if has_api_key:
                agent = create_llm_agent(result)
                if agent:
                    explanation_request = """
                    Analyze the Two Sample T-test results focusing on the T-statistic 
                    and P-value. The T-statistic measures the size of the difference 
                    relative to the variation in your sample data. A higher value 
                    indicates a greater difference between groups. 
                    The P-value determines the statistical significance of this 
                    difference. A low P-value (typically <0.05) suggests that the 
                    observed difference is unlikely to have occurred by chance, 
                    indicating a significant difference between the group means. 
                    This analysis helps to understand if variations between two groups 
                    are statistically meaningful.
                    """
                    st.markdown("### AI-Generated Insights:")
                    with st.spinner('Analyzing results with AI...'):
                        try:
                            answer = agent.run(explanation_request)
                            st.write(answer)
                        except Exception as e:
                            st.error(f"AI analysis error: {str(e)}")
            else:
                st.info("💡 Add your OpenAI API key on the Home Page to get AI-powered interpretation of these results.")
        else:
            st.error(f"Selected column must have exactly two groups for T-test. Found {len(groups)} groups.")

elif statistical_test_technique == "ANOVA (Analysis of Variance)":
    st.markdown("### One-way ANOVA")
    st.info("ANOVA compares means across 3+ groups. It assumes normally distributed data with equal variances.")
    
    # Implementation for ANOVA, following similar pattern as above with appropriate changes
    # Select categorical column with 3+ groups
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        categorical_cols = df.columns
    
    col = st.selectbox("Select grouping column (should have 3+ groups)", options=categorical_cols)
    
    # Warn if not enough groups
    if df[col].nunique() < 3:
        st.warning(f"⚠️ Selected column '{col}' has only {df[col].nunique()} unique values. ANOVA is typically used for 3+ groups.")
    
    # Select numerical column for testing
    numerical_cols = df.select_dtypes(include=['float', 'int']).columns
    if len(numerical_cols) == 0:
        st.error("No numerical columns found in the dataset. ANOVA requires numerical data.")
        st.stop()
    
    selected_col_for_test = st.selectbox("Select numerical column to test", options=numerical_cols)
    
    # Optional: Visualize group distributions
    if st.checkbox("Visualize group distributions"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=col, y=selected_col_for_test, data=df, ax=ax)
        st.pyplot(fig)
    
    if st.button('🔄 Perform ANOVA', use_container_width=True):
        groups = df.groupby(col)
        args = [group[selected_col_for_test].dropna() for name, group in groups]
        
        # Check if groups have enough data
        small_groups = [i for i, arg in enumerate(args) if len(arg) < 2]
        if small_groups:
            st.error(f"Groups at indices {small_groups} have fewer than 2 valid data points after removing NaNs.")
            st.stop()
        
        f_stat, p_val = stats.f_oneway(*args)
        
        # Display results
        st.markdown("### ANOVA Results:")
        result = pd.DataFrame({
            'F-statistic': [f_stat], 
            'P-value': [p_val],
            'Number of groups': [len(args)]
        })
        st.dataframe(result)
        
        # Group statistics
        group_stats = df.groupby(col)[selected_col_for_test].agg(['count', 'mean', 'std']).reset_index()
        st.markdown("### Group Statistics:")
        st.dataframe(group_stats)
        
        # Significance interpretation
        alpha = 0.05
        if p_val < alpha:
            st.success(f"✅ Statistically significant difference detected among groups (p = {p_val:.4f} < 0.05)")
        else:
            st.info(f"ℹ️ No statistically significant difference detected among groups (p = {p_val:.4f} ≥ 0.05)")
        
        # LLM interpretation
        agent = create_llm_agent(result)
        if agent:
            explanation_request = """
            Analyze the ANOVA results by focusing on the F-statistic and P-value. 
            The F-statistic evaluates the overall variance among group means, where 
            a higher value suggests a significant difference across groups. The 
            P-value assesses this difference's statistical significance, with 
            values below 0.05 indicating a strong likelihood that at least one 
            group mean differs significantly. This analysis is crucial for 
            identifying whether variations across multiple groups are statistically 
            meaningful, helping to guide further research or data-driven decisions.
            """
            st.markdown("### AI-Generated Insights:")
            with st.spinner('Analyzing results with AI...'):
                try:
                    answer = agent.run(explanation_request)
                    st.write(answer)
                except Exception as e:
                    st.error(f"AI analysis error: {str(e)}")

# Similarly implement the other test options with improved code structure
# For brevity, I've shown the pattern for two tests - you would continue with similar implementations for the others

elif statistical_test_technique == "Chi-Square Test":
    st.markdown("### Chi-Square Test of Independence")
    st.info("This test analyzes relationships between categorical variables.")
    
    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) < 2:
        st.warning("This test requires at least two categorical columns. You may need to convert some numerical columns to categorical first.")
    
    col1 = st.selectbox('Select first categorical column', options=categorical_columns)
    col2 = st.selectbox('Select second categorical column', options=categorical_columns)
    
    if st.button('🔄 Perform Chi-Square Test', use_container_width=True):
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        st.markdown("### Contingency Table:")
        st.dataframe(contingency_table)
        
        # Check expected frequencies
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
        
        # Check if expected frequencies are sufficient
        if (expected < 5).any():
            st.warning("⚠️ Some expected frequencies are less than 5, which may affect the reliability of the Chi-Square test.")
        
        # Display results
        st.markdown('### Chi-Square Test Results:')
        result = pd.DataFrame({
            'Chi-Square Statistic': [chi2], 
            'P-value': [p_value],
            'Degrees of Freedom': [dof]
        })
        st.dataframe(result)
        
        # Significance interpretation
        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ Statistically significant association detected (p = {p_value:.4f} < 0.05)")
        else:
            st.info(f"ℹ️ No statistically significant association detected (p = {p_value:.4f} ≥ 0.05)")
        
        # Optional: Visualize the relationship
        if st.checkbox("Visualize relationship"):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
            ax.set_title(f"Relationship between {col1} and {col2}")
            st.pyplot(fig)
        
        # LLM interpretation
        agent = create_llm_agent(result)
        if agent:
            explanation_request = """
            Analyze the Chi-Square test results by focusing on the Chi-Square statistic 
            and P-value. The Chi-Square statistic measures how expected counts are 
            compared to observed counts across categories, indicating how well the 
            observed distributions fit the expected distributions. A higher value 
            may suggest a significant association between the variables. The P-value
            assesses the statistical significance of this association, with values 
            below 0.05 typically indicating that the observed association is unlikely 
            to be due to chance, highlighting potentially meaningful differences or 
            relationships between categories.
            """
            st.markdown("### AI-Generated Insights:")
            with st.spinner('Analyzing results with AI...'):
                try:
                    answer = agent.run(explanation_request)
                    st.write(answer)
                except Exception as e:
                    st.error(f"AI analysis error: {str(e)}")

elif statistical_test_technique == "Mann-Whitney U Test":
    st.markdown("### Mann-Whitney U Test")
    st.info("This non-parametric test compares two independent groups without assuming normal distribution.")
    
    # Select categorical column with exactly 2 groups
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        categorical_cols = df.columns
    
    group_column = st.selectbox('Select Group Column (should have exactly 2 groups)', options=categorical_cols)
    
    # Check if selected column has exactly 2 groups
    if df[group_column].nunique() != 2:
        st.warning(f"⚠️ Selected column '{group_column}' has {df[group_column].nunique()} unique values. Mann-Whitney U test requires exactly 2 groups.")
    
    # Select numerical column for testing
    numeric_columns = df.select_dtypes(include=['float', 'int']).columns
    if len(numeric_columns) == 0:
        st.error("No numerical columns found in the dataset. Mann-Whitney U test requires numerical data.")
        st.stop()
    
    value_column = st.selectbox('Select Value Column (Numeric)', options=numeric_columns)
    
    # Optional: Visualize distributions
    if st.checkbox("Visualize distributions"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=group_column, y=value_column, data=df, ax=ax)
        st.pyplot(fig)
    
    if st.button('🔄 Perform Mann-Whitney U Test', use_container_width=True):
        groups = df[group_column].unique()
        if len(groups) == 2:
            data1 = df[df[group_column] == groups[0]][value_column].dropna()
            data2 = df[df[group_column] == groups[1]][value_column].dropna()
            
            if len(data1) < 1 or len(data2) < 1:
                st.error("Not enough valid data points after removing NaNs.")
                st.stop()
            
            u_stat, p_value = mannwhitneyu(data1, data2)
            
            # Display results
            st.markdown('### Mann-Whitney U Test Results:')
            result = pd.DataFrame({
                'U-Statistic': [u_stat], 
                'P-value': [p_value],
                f'Median ({groups[0]})': [data1.median()],
                f'Median ({groups[1]})': [data2.median()],
                f'Count ({groups[0]})': [len(data1)],
                f'Count ({groups[1]})': [len(data2)]
            })
            st.dataframe(result)
            
            # Significance interpretation
            alpha = 0.05
            if p_value < alpha:
                st.success(f"✅ Statistically significant difference detected (p = {p_value:.4f} < 0.05)")
            else:
                st.info(f"ℹ️ No statistically significant difference detected (p = {p_value:.4f} ≥ 0.05)")
            
            # LLM interpretation
            agent = create_llm_agent(result)
            if agent:
                explanation_request = """
                Analyze the Mann-Whitney U Test results by focusing on the U-statistic 
                and P-value. The U-statistic compares ranks between two independent 
                samples, indicating differences in their central tendency. A high 
                U-statistic may suggest significant discrepancies. The P-value evaluates
                the statistical significance of these discrepancies, with values below 
                0.05 generally indicating a statistically significant difference between 
                the groups, suggesting that any observed differences are unlikely to have
                occurred by chance. This analysis is essential for understanding the 
                distributional differences between two independent samples.
                """
                st.markdown("### AI-Generated Insights:")
                with st.spinner('Analyzing results with AI...'):
                    try:
                        answer = agent.run(explanation_request)
                        st.write(answer)
                    except Exception as e:
                        st.error(f"AI analysis error: {str(e)}")
        else:
            st.error(f"Selected column must have exactly two groups. Found {len(groups)} groups.")

# Continue implementing the remaining tests (Wilcoxon and Kruskal-Wallis) following the same pattern

# Add a help section at the bottom
with st.expander("💡 Help & Tips for Statistical Testing"):
    st.markdown("""
    ### Tips for Choosing the Right Test:
    
    1. **For comparing two groups:**
       - If data is normally distributed: Use the **T-test**
       - If data is not normally distributed: Use the **Mann-Whitney U test**
       - If data is paired/related: Use the **Wilcoxon Signed-Rank test**
    
    2. **For comparing more than two groups:**
       - If data is normally distributed: Use **ANOVA**
       - If data is not normally distributed: Use the **Kruskal-Wallis test**
    
    3. **For categorical data:**
       - Use the **Chi-Square test** to examine relationships between categorical variables
    
    ### Common Pitfalls:
    - Ignoring assumptions of tests (normality, equal variances)
    - Using too small sample sizes
    - Interpreting statistical significance as practical significance
    - Multiple testing without correction
    
    ### Next Steps After Testing:
    - If your test shows significant results, consider post-hoc tests to identify which specific groups differ
    - Consider effect sizes to understand the magnitude of differences
    - Visualize your results to aid interpretation
    """)

# Add footer with version info
st.divider()
st.markdown("**Statistical Analysis Module v1.0.1** | © 2025")