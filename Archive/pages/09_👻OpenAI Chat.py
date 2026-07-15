import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import re
from datetime import datetime
import textwrap
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import json
import io
import logging
from contextlib import redirect_stdout

# Import OpenAI components
from openai import OpenAI

try:
    import seaborn as sns
except AttributeError:
    # If there's an issue with seaborn and cm.register_cmap, use this workaround
    import matplotlib as mpl
    if not hasattr(mpl.cm, 'register_cmap'):
        # Add a dummy function to prevent the error
        mpl.cm.register_cmap = lambda *args, **kwargs: None
    import seaborn as sns




# Setup logging for security tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure page settings
st.set_page_config(page_title="Data Chat", layout="wide")

# Custom CSS for improved styling
st.markdown("""
    <style>
        /* General styling improvements */
        .main .block-container {
            padding-top: 2rem;
        }
        
        /* Code block styling */
        .highlight pre { 
            padding: 16px; 
            border-radius: 6px; 
            font-size: 14px; 
            line-height: 1.4;
        }
        .stCodeBlock {
            background-color: #f8f9fa;
            padding: 1.2rem;
            border-radius: 0.5rem;
            margin: 1.2rem 0;
            border: 1px solid #e9ecef;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Plot container */
        .plot-container {
            background-color: white;
            padding: 1.2rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 1.5rem 0;
            border: 1px solid #e9ecef;
        }
        
        /* Example queries box */
        .example-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
        }
        
        /* Chat message styling */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            animation: fadein 0.5s;
        }
        
        /* Status indicators */
        .status-indicator {
            padding: 0.3rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
            font-weight: 500;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-warning {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }
        .status-info {
            background-color: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .status-danger {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        /* Animations */
        @keyframes fadein {
            from { opacity: 0; }
            to   { opacity: 1; }
        }
        
        /* Table styling */
        .dataframe-container {
            overflow-x: auto;
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin: 1rem 0;
        }
        
        /* Metrics */
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin: 1rem 0;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            flex: 1;
            min-width: 120px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #1e88e5;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_df_name" not in st.session_state:
    st.session_state.current_df_name = "Initial DataFrame"
    
if "show_code" not in st.session_state:
    st.session_state.show_code = True
    
if "execution_times" not in st.session_state:
    st.session_state.execution_times = []
    
if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "security_mode" not in st.session_state:
    st.session_state.security_mode = "medium"

# Cache decorator for performance optimization
@st.cache_data(ttl=3600, show_spinner=False)
def format_code(code):
    """Format Python code with syntax highlighting."""
    try:
        highlighted = highlight(code, PythonLexer(), HtmlFormatter(style='default'))
        return f'<div class="stCodeBlock">{highlighted}</div>'
    except Exception as e:
        # Fallback if syntax highlighting fails
        return f'<div class="stCodeBlock"><pre>{code}</pre></div>'



# Function to convert DataFrame to string representation (works for both Pandas and Polars)
def dataframe_to_string(df, full=False):
    """Convert either a Pandas or Polars DataFrame to a string representation.
    If full=True, returns the complete DataFrame, otherwise returns first 5 rows."""
    import pandas as pd
    
    # Check if it's a Polars DataFrame
    if hasattr(df, 'to_pandas'):
        # Convert Polars to Pandas first
        pandas_df = df.to_pandas()
        if full:
            return pandas_df.to_string()
        else:
            return pandas_df.head(5).to_string()
    # Check if it's a Pandas DataFrame
    elif isinstance(df, pd.DataFrame):
        if full:
            return df.to_string()
        else:
            return df.head(5).to_string()
    else:
        # Fallback for other types
        if full:
            return str(df)
        else:
            return str(df.head(5))



@st.cache_data(ttl=3600, show_spinner=False)
def get_dataframe_profile(df):
    """Generate a quick profile of the dataframe."""
    try:
        profile = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isna().sum().sum(),
            "missing_percentage": round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
            "memory_usage": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),  # MB
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime']).columns),
        }
        return profile
    except Exception as e:
        return {}


def is_suspicious_query(query):
    """Check if the query contains potentially suspicious patterns."""
    suspicious_patterns = [
        r'hack', r'exploit', r'bypass', r'injection',
        r'delete.*data', r'drop.*table', r'system command',
        r'connect.*external', r'send.*data.*to',
        r'password', r'credentials', r'token', r'api key',
        r'os\.', r'sys\.', r'subprocess', r'exec\(', r'eval\('
    ]
    
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in suspicious_patterns)


def log_execution(user_id, query, result_status):
    """Log execution for security tracking."""
    logging.info(f"User: {user_id}, Query: {query[:100]}..., Status: {result_status}")


def safe_execute_code(code, df):
    """Safely execute Python code with the dataframe using enhanced security measures."""
    # Convert Polars DataFrame to Pandas if needed
    if not isinstance(df, pd.DataFrame):
        try:
            # Attempt to convert Polars DataFrame to Pandas
            df = df.to_pandas()
        except Exception:
            # If conversion fails, raise an error
            raise ValueError("Unable to convert DataFrame to Pandas format")
    
    # Create a safe local environment
    local_vars = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
    
    result = {
        "output": None,
        "error": None,
        "dataframe": None,
        "figure": None
    }
    
    try:
        # Define comprehensive patterns for unsafe operations
        dangerous_patterns = [
            r'os\.', r'sys\.', r'subprocess', r'exec\(', r'eval\(',
            r'import\s+(?!pandas|numpy|matplotlib|seaborn)', r'__import__',
            r'open\(', r'file\(', r'read|write', r'glob', 
            r'shutil', r'requests\.', r'urllib', r'socket\.',
            r'while\s+True', r'for\s+.*\s+in\s+range\(\s*\d{7,}\s*\)',  # Resource exhaustion
            r'df\.drop\(\s*([\'"][\'"]|1)\s*\)', # Dropping all data
            r'globals\(\)', r'locals\(\)', r'getattr\(', r'setattr\(', r'delattr\('
        ]
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                raise ValueError(f"Code contains potentially unsafe operations: {pattern}")
        
        # Preprocess code to remove or replace problematic commands
        # Replace plt.show() with pass to avoid warnings
        code = re.sub(r'plt\.show\(\)', 'pass', code)
        
        # Ensure code creates a figure if visualization-related functions are called
        if re.search(r'(plt\.|sns\.)(plot|bar|hist|scatter|box|line|heat|pair)', code) and not 'plt.figure' in code:
            code = "plt.figure(figsize=(10, 6))\n" + code
        
        # Add resource limitations
        # Check dataframe size
        if df.shape[0] * df.shape[1] > 10000000:  # Approx limit for larger operations
            if any(op in code.lower() for op in ['groupby', 'apply', 'merge', 'join']):
                sample_size = min(10000, int(df.shape[0] * 0.1))
                code = f"# Using sample due to dataframe size\ndf = df.sample({sample_size}) if len(df) > {sample_size} else df\n" + code
        
        # Add print statements to capture intermediate values
        modified_code = "import io\nfrom contextlib import redirect_stdout\n"
        modified_code += "\n".join(code.split('\n'))
        
        # Capture stdout
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            # Execute the code with timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timed out after 30 seconds")
            
            # Set timeout (30 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                exec(modified_code, local_vars)
                signal.alarm(0)  # Reset alarm
            except Exception as e:
                signal.alarm(0)  # Reset alarm
                raise e
        
        # Get the output
        output = buffer.getvalue()
        if output:
            result["output"] = output
        
        # Check if a new dataframe was created
        for var_name, var_value in local_vars.items():
            if var_name != "df" and isinstance(var_value, pd.DataFrame):
                # Security check - limit large DataFrame returns
                if var_value.shape[0] > 1000:
                    result["dataframe"] = var_value.head(1000)
                    result["output"] = (result["output"] or "") + "\n[Note: Large DataFrame truncated to 1000 rows for display]"
                else:
                    result["dataframe"] = var_value
                break
        
        # Check if a figure was created and capture it
        if plt.get_fignums():
            fig = plt.gcf()
            plt.tight_layout()  # Improve layout
            result["figure"] = fig
        else:
            # If no figure was created but code has plotting commands, try to create one
            if re.search(r'(plt\.|sns\.)(plot|bar|hist|scatter|box|line|heat|pair)', code):
                plt.figure(figsize=(10, 6))
                exec(modified_code, local_vars)
                if plt.get_fignums():
                    fig = plt.gcf()
                    plt.tight_layout()
                    result["figure"] = fig
            # Close any unused figures to avoid memory leaks
            plt.close('all')
        
        return result
    except TimeoutError as e:
        result["error"] = "Code execution timed out. This may be due to an infinite loop or excessive processing."
        plt.close('all')
        return result
    except Exception as e:
        result["error"] = str(e)
        # Make sure to close any partial figures on error
        plt.close('all')
        return result


def render_dataframe_info(df):
    """Render information about the current dataframe."""
    profile = get_dataframe_profile(df)
    
    st.markdown("### Dataset Overview")
    
    # Metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", profile.get("rows", 0))
    with col2:
        st.metric("Columns", profile.get("columns", 0))
    with col3:
        st.metric("Missing Values", f"{profile.get('missing_percentage', 0)}%")
    with col4:
        st.metric("Size", f"{profile.get('memory_usage', 0)} MB")
    
    # Column type breakdown
    st.markdown("#### Column Types")
    col_types = {
        "Numeric": profile.get("numeric_columns", 0),
        "Categorical": profile.get("categorical_columns", 0),
        "Datetime": profile.get("datetime_columns", 0),
        "Other": profile.get("columns", 0) - profile.get("numeric_columns", 0) - 
                profile.get("categorical_columns", 0) - profile.get("datetime_columns", 0)
    }
    
    # Create a horizontal bar chart for column types
    fig, ax = plt.subplots(figsize=(10, 2))
    bars = ax.barh(list(col_types.keys()), list(col_types.values()))
    ax.bar_label(bars)
    ax.set_xlabel('Count')
    ax.set_title('Column Types Distribution')
    st.pyplot(fig)


def render_chat_interface(df, client):
    """Render the chat interface for interacting with data using direct OpenAI API calls."""
    # Render existing messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            if "content" in message:
                if isinstance(message["content"], dict):
                    if "security_warning" in message["content"] and message["content"]["security_warning"]:
                        st.markdown(f"<div class='status-danger' style='width: 100%;'>{message['content']['explanation']}</div>", 
                                   unsafe_allow_html=True)
                    elif "explanation" in message["content"]:
                        # Filter out matplotlib warnings
                        explanation = message["content"]["explanation"]
                        explanation = re.sub(r'<string>:.*?cannot show the figure\.', '', explanation)
                        explanation = re.sub(r'INFO:httpx:.*?"HTTP/1\.1 200 OK"', '', explanation)
                        explanation = explanation.strip()
                        st.write(explanation)
                    
                    if "code" in message["content"] and message["content"]["code"]:
                        for code in message["content"]["code"]:
                            st.markdown(format_code(code), unsafe_allow_html=True)
                else:
                    # Simple text content
                    st.write(message["content"])
            
            if "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])
            
            if "execution_time" in message:
                st.markdown(f"<span class='status-info' style='float:right'>⏱️ {message['execution_time']:.2f}s</span>", unsafe_allow_html=True)
    
    # Chat input and options
    user_question = st.chat_input("Ask me anything about your data!", disabled=False)
    
    if st.button("Clear Chat 🗑️", type="primary", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("")
    
    with col2:
        cols = st.multiselect("Focus on specific columns (optional)", df.columns)
    
    # Process the user question
    if user_question:
        # Check for security issues first
        if is_suspicious_query(user_question):
            st.chat_message("user").write(user_question)
            with st.chat_message("assistant"):
                st.markdown(f"<div class='status-danger' style='width: 100%;'>⚠️ Your query contains potentially sensitive or unsafe patterns. Please rephrase your request to focus on data analysis tasks.</div>", 
                           unsafe_allow_html=True)
            # Don't add this to chat history
            return
        
        # Enhance the prompt with selected columns if specified
        if cols:
            user_question = f"Focus on columns {', '.join(cols)}. {user_question}"
        
        # Add to message history
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        
        # Add to chat history for later reference
        st.session_state.chat_history.append({
            "query": user_question,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataframe": st.session_state.current_df_name
        })
        
        # Display the user message
        with st.chat_message("user"):
            st.write(user_question)
        
        # Generate and display the response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data... 🔍"):
                start_time = time.time()
                
                try:
                    # Generate dataframe description 
                    # Generate dataframe description 
                    df_sample = dataframe_to_string(df, full=True)
                    #df_sample = df.head(5).to_string()
                    df_info = f"""
                    DataFrame Information:
                    - Shape: {df.shape}
                    - Columns: {', '.join(list(df.columns))}
                    - Data types: {', '.join([f"{col}: {df[col].dtype}" for col in df.columns])}
                    - Sample data (first 5 rows): 
                    {df_sample}
                    """
                    
                    # Format the prompt for the model
                    system_message = "You are a Data Scientist Expert. Whenever a user provides a question, access the dataframe they provide, analyze it, and answer based on the data in the dataframe. If there is information missing or the query requires additional code, provide the user with the Python code they can run on their own to retrieve the answer. If the user asks for visualizations or plots, provide them with the code to generate the desired plot. Your goal is to be helpful, clear, and actionable, empowering the user to solve data-related problems efficiently."
                    model_id = "gpt-4o-mini" # Default to gpt-4o-mini, but can be changed to other models
                    
                    prompt = f"""
                    {df_info}
                    
                    Please respond to this question about the data:
                    {user_question}
                    
                    If code is needed, provide Python code using pandas, matplotlib, and other common data science libraries.
                    Always wrap code in ```python and ``` tags.
                    """
                    
                    # Call API with streaming
                    stream = True
                    try:
                        # Check if client supports streaming (model-dependent)
                        completion = client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.4,
                            stream=stream
                        )
                        
                        # Use placeholder for streaming
                        response_placeholder = st.empty()
                        collected_messages = []
                        
                        if stream:
                            # Streaming response
                            response_text = ""
                            for chunk in completion:
                                if chunk.choices[0].delta.content is not None:
                                    content_chunk = chunk.choices[0].delta.content
                                    response_text += content_chunk
                                    collected_messages.append(content_chunk)
                                    # Update the placeholder with the current text
                                    response_placeholder.markdown(response_text)
                        else:
                            # Non-streaming response
                            response_text = completion.choices[0].message.content
                            response_placeholder.markdown(response_text)
                            
                    except Exception as api_error:
                        # Fallback to non-streaming if streaming fails
                        st.warning(f"Streaming failed, falling back to standard mode: {str(api_error)}")
                        completion = client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.4,
                            stream=False
                        )
                        response_text = completion.choices[0].message.content
                        
                    # Extract code blocks if any
                    code_blocks = []
                    if "```python" in response_text:
                        code_pattern = r"```python(.*?)```"
                        code_blocks = re.findall(code_pattern, response_text, re.DOTALL)
                        # Clean up the code blocks
                        code_blocks = [block.strip() for block in code_blocks]
                    
                    # Execute any visualization code
                    figure = None
                    if code_blocks:
                        for code_block in code_blocks:
                            # Check if this is visualization code
                            if re.search(r'(plt\.|sns\.)(plot|bar|hist|scatter|box|line|heat|pair)', code_block):
                                execution_result = safe_execute_code(code_block, df)
                                if "figure" in execution_result and execution_result["figure"] is not None:
                                    figure = execution_result["figure"]
                                    break
                    
                    # Calculate execution time
                    execution_time = time.time() - start_time
                    
                    # Construct the response object
                    response_obj = {
                        "explanation": response_text,
                        "code": code_blocks,
                    }
                    
                    if figure is not None:
                        response_obj["figure"] = figure
                        st.pyplot(figure)
                    
                    # Display code blocks with syntax highlighting
                    if code_blocks:
                        for code in code_blocks:
                            st.markdown(format_code(code), unsafe_allow_html=True)
                    
                    # Display execution time
                    st.markdown(f"<span class='status-info' style='float:right'>⏱️ {execution_time:.2f}s</span>", unsafe_allow_html=True)
                    
                    # Add to chat history
                    message_data = {
                        "role": "assistant", 
                        "content": response_obj,
                        "execution_time": execution_time
                    }
                    
                    if figure is not None:
                        message_data["figure"] = figure
                    
                    st.session_state.chat_messages.append(message_data)
                    st.session_state.execution_times.append(execution_time)
                    
                except Exception as e:
                    error_message = f"⚠️ Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_message,
                        "execution_time": time.time() - start_time
                    })


def render_performance_metrics():
    """Render performance metrics about query execution times."""
    if not st.session_state.execution_times:
        return
    
    with st.expander("📊 Performance Metrics", expanded=False):
        times = st.session_state.execution_times
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Response Time", f"{np.mean(times):.2f}s")
        with col2:
            st.metric("Fastest Response", f"{np.min(times):.2f}s")
        with col3:
            st.metric("Slowest Response", f"{np.max(times):.2f}s")
        
        # Plot response time history
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(range(1, len(times) + 1), times, marker='o')
        ax.set_title('Query Response Times')
        ax.set_xlabel('Query Number')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)


def main():
    """Main application entry point with simplified OpenAI-only interface."""
    st.title("👻Chat with Data using OpenAI")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Model configuration sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # API Key input
        if 'openai_api_key' not in st.session_state or not st.session_state.openai_api_key:
            openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
            if openai_api_key:
                st.session_state.openai_api_key = openai_api_key
                st.success("OpenAI API key saved!")
                st.rerun()
            else:
                st.error("⚠️ OpenAI API key is required.")
    
        # Model selection (simplified)
        st.header("⚙️ Model Settings")
        model_option = st.selectbox(
            "Select OpenAI Model",
            ["gpt-4o-mini", "gpt-4o"],
            index=0
        )
        
        if "selected_model" not in st.session_state or st.session_state.selected_model != model_option:
            st.session_state.selected_model = model_option
    
    # Security Mode Selector in sidebar
    with st.sidebar:
        st.header("🔒 Security Settings")
        if "security_mode" not in st.session_state:
            st.session_state.security_mode = "medium"
            
        st.session_state.security_mode = st.radio(
            "Security Mode", 
            ["medium", "high"], 
            index=0 if st.session_state.security_mode == "medium" else 1
        )
    
    # Check if DataFrame exists
    if 'df' not in st.session_state or st.session_state.df is None:
        uploaded_file = st.file_uploader("📤 Upload your dataset (CSV, Excel, etc.)", type=["csv", "xlsx", "xls"])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                
                st.success(f"Dataset loaded: {uploaded_file.name}")
                st.dataframe(st.session_state.df.head())
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        st.warning("Please upload a dataset to get started.")
        st.stop()
    
    # Get available dataframes
    dataframes = {
        "Initial DataFrame": st.session_state.df,
        "Processed DataFrame": st.session_state.get('df_processed'),
        "Encoded DataFrame": st.session_state.get('df_encoded'),
        "Scaled DataFrame": st.session_state.get('df_scaled')
    }
    
    # Filter out None values
    available_dataframes = {k: v for k, v in dataframes.items() if v is not None}
    
    if not available_dataframes:
        st.warning("No dataframes available for analysis.")
        st.stop()
    
    with st.expander('Chat Settings'):
        # DataFrame selector
        if "current_df_name" not in st.session_state:
            st.session_state.current_df_name = "Initial DataFrame"
            
        st.session_state.current_df_name = st.selectbox(
            "Select DataFrame", 
            list(available_dataframes.keys()),
            index=list(available_dataframes.keys()).index(st.session_state.current_df_name) 
            if st.session_state.current_df_name in available_dataframes else 0
        )
    
    # Main content area
    selected_df = available_dataframes[st.session_state.current_df_name]
    
    # Convert to pandas if needed
    if not isinstance(selected_df, pd.DataFrame):
        selected_df = selected_df.to_pandas()
    
    # Show dataframe info
    with st.expander("Dataset Overview", expanded=False):
        render_dataframe_info(selected_df)
    
    # Render performance metrics if available
    if "execution_times" in st.session_state and st.session_state.execution_times:
        render_performance_metrics()
    
    # Check if API key is set before rendering chat interface
    if 'openai_api_key' in st.session_state and st.session_state.openai_api_key:
        # Initialize OpenAI client if not already done
        if "openai_client" not in st.session_state:
            with st.spinner("Initializing OpenAI client..."):
                st.session_state.openai_client = OpenAI(api_key=st.session_state.openai_api_key)
        
        # Use the selected model from the sidebar
        if "selected_model" in st.session_state:
            model_id = st.session_state.selected_model
        else:
            model_id = "gpt-4o-mini"  # Default model
        
        # Render the chat interface
        if st.session_state.openai_client:
            render_chat_interface(selected_df, st.session_state.openai_client)
        else:
            st.error("Failed to initialize OpenAI client. Please check your API key.")
    else:
        st.error("⚠️ Please set your OpenAI API key in the sidebar to continue.")


if __name__ == "__main__":
    main()