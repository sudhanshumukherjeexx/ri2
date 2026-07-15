import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import json
import requests
from datetime import datetime
from io import StringIO
import logging
from contextlib import redirect_stdout
import textwrap
import os
from huggingface_hub import InferenceClient


try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


try:
    import seaborn as sns
except AttributeError:
    # cm.register_cmap - workaround
    import matplotlib as mpl
    if not hasattr(mpl.cm, 'register_cmap'):
        # Add a dummy function to prevent the error
        mpl.cm.register_cmap = lambda *args, **kwargs: None
    import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure page settings
st.set_page_config(page_title="Data Chat (Enhanced)", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* General styling improvements */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Code block styling */
    .code-block {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin: 1.2rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        font-family: monospace;
        white-space: pre-wrap;
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

if "using_direct_api" not in st.session_state:
    st.session_state.using_direct_api = False

# format Python code
def format_code(code):
    """Format Python code with basic styling."""
    return f'<div class="code-block">{code}</div>'

# function to directly query Hugging Face models
def query_huggingface_model(prompt, api_token, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """Query a Hugging Face model directly without using LangChain."""
    try:
        client = InferenceClient(model=model_name, token=api_token)
        
        # Format the prompt for instruction following models
        formatted_prompt = f"""
                    
                    <s>[INST] 
                    
                    You are a experience data scientist assistant with 10 years of experience. I will provide you with a DataFrame 
                    description and a query about it. Please help me analyze the data. When I ask for visualizations or data transformations, 
                    provide the Python code to achieve this.
                    {prompt} 
    
                    [/INST]</s>
                    
                    """
        
        response = client.text_generation(
            formatted_prompt,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            return_full_text=False
        )
        
        return response
    except Exception as e:
        logging.error(f"Error querying Hugging Face model: {str(e)}")
        return f"Error: {str(e)}"

# Function to create the pandas dataframe agent
def create_agent_for_df(df, api_token, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    """Create a LangChain pandas dataframe agent, with fallback to direct API calls."""
    # Convert to Pandas DataFrame if needed
    if not isinstance(df, pd.DataFrame):
        try:
            # Attempt to convert to Pandas
            st.info("Converting to Pandas DataFrame for analysis...")
            df = df.to_pandas()
        except Exception as e:
            st.error(f"Unable to convert DataFrame to Pandas format: {str(e)}")
            return None
        
    # We'll use a simple flag to indicate if we're using the agent or the direct API
    st.session_state.using_direct_api = False
    
    try:
        # Return a simple wrapper function that mimics the agent interface
        # but actually uses direct API calls to Hugging Face
        def direct_api_wrapper(query):
            df_info = f"""
                        DataFrame Information:
                        - Shape: {df.shape}
                        - Columns: {', '.join(list(df.columns))}
                        - Data types: {', '.join([f"{col}: {df[col].dtype}" for col in df.columns])}
                        - Sample data (first 5 rows): 
                        {df.head().to_string()}

                        User query: {query}
                        """
            return query_huggingface_model(df_info, api_token, model_name)
        
        # Set the flag
        st.session_state.using_direct_api = True
        
        # Return the wrapper function
        return direct_api_wrapper
        
    except Exception as e:
        logging.error(f"Error initializing LLM: {str(e)}")
        return None

# Fix for the Dataset Overview showing zeros
def get_dataframe_profile(df):
    """Generate a quick profile of the dataframe."""
    try:
        # Convert to Pandas DataFrame if needed
        if not isinstance(df, pd.DataFrame):
            try:
                # Attempt to convert to Pandas
                df = df.to_pandas()
            except Exception as e:
                raise ValueError(f"Unable to convert DataFrame to Pandas format: {str(e)}")
            
        # Calculate total cells in DataFrame
        total_cells = df.shape[0] * df.shape[1]
        
        # Calculate missing values
        missing_count = df.isna().sum().sum()
        
        # Calculate missing percentage with safeguard against division by zero
        missing_percentage = (missing_count / total_cells * 100) if total_cells > 0 else 0
        
        profile = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": missing_count,
            "missing_percentage": round(missing_percentage, 2),
            "memory_usage": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),  # MB
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime']).columns),
        }
        
        # Make sure all values are valid (not NaN)
        for key, value in profile.items():
            if pd.isna(value):
                profile[key] = 0
        
        return profile
    except Exception as e:
        logging.error(f"Error in get_dataframe_profile: {str(e)}")
        # Return fallback values
        return {
            "rows": df.shape[0] if hasattr(df, 'shape') else 0,
            "columns": df.shape[1] if hasattr(df, 'shape') else 0,
            "missing_values": 0,
            "missing_percentage": 0,
            "memory_usage": 0,
            "numeric_columns": 0,
            "categorical_columns": 0,
            "datetime_columns": 0
        }

# Function to check for suspicious patterns
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

# Function to log execution for security
def log_execution(user_id, query, result_status):
    """Log execution for security tracking."""
    logging.info(f"User: {user_id}, Query: {query[:100]}..., Status: {result_status}")

# Function to process agent response
def process_agent_response(response):
    """Process the response from the agent or direct API."""
    # Handle different response types
    if isinstance(response, str):
        # Direct API response
        result = {
            "explanation": response,
            "code": [],
            "figure": None,
            "output": None,
            "dataframe": None
        }
    else:
        # Try to handle any other response type
        try:
            result = {
                "explanation": str(response),
                "code": [],
                "figure": None,
                "output": None,
                "dataframe": None
            }
        except Exception as e:
            logging.error(f"Error processing response: {str(e)}")
            return {
                "explanation": f"Error processing response: {str(e)}",
                "code": [],
                "figure": None,
                "output": None,
                "dataframe": None
            }
    
    # Extract code blocks if any
    explanation = result["explanation"]
    if "```python" in explanation or "```" in explanation:
        code_pattern = r"```(?:python)?(.*?)```"
        code_blocks = re.findall(code_pattern, explanation, re.DOTALL)
        # Clean up the code blocks
        result["code"] = [block.strip() for block in code_blocks]
    
    # Execute the last code block if present to capture visualizations
    if result["code"]:
        try:
            # Check if there's visualization code in the last block
            last_code = result["code"][-1]
            if re.search(r'(plt\.|sns\.)(plot|bar|hist|scatter|box|line|heat|pair)', last_code):
                # Execute the code in a clean environment
                locals_dict = {"plt": plt, "sns": sns, "pd": pd, "np": np}
                exec(last_code, globals(), locals_dict)
                
                # Capture figure if available
                if plt.get_fignums():
                    result["figure"] = plt.gcf()
        except Exception as e:
            logging.error(f"Error executing visualization code: {str(e)}")
    
    return result

# Function to generate a response using the agent or direct API
def generate_response_with_agent(agent, query, show_code=True):
    """Generate response using the agent function."""
    start_time = time.time()
    
    try:
        # Security check for suspicious queries
        if is_suspicious_query(query):
            return {
                "explanation": "⚠️ This query contains potentially sensitive or unsafe patterns. Please rephrase your request.",
                "security_warning": True
            }, 0
        
        # Run the agent (or direct API function)
        if st.session_state.using_direct_api:
            response = agent(query)
        else:
            # This would be for the actual LangChain agent if implemented
            response = agent.run(query)
        
        # Process the response
        result = process_agent_response(response)
        
        # If show_code is False, remove code blocks from the explanation
        if not show_code and "```" in result["explanation"]:
            result["explanation"] = re.sub(r"```python.*?```", "", result["explanation"], flags=re.DOTALL)
            result["explanation"] = re.sub(r"```.*?```", "", result["explanation"], flags=re.DOTALL)
        
        # Log successful execution
        log_execution("user", query, "execution_complete")
        
        execution_time = time.time() - start_time
        st.session_state.execution_times.append(execution_time)
        
        return result, execution_time
        
    except Exception as e:
        execution_time = time.time() - start_time
        st.session_state.execution_times.append(execution_time)
        error_message = f"Error generating response: {str(e)}"
        
        # Log the execution error
        log_execution("user", query, f"execution_error: {str(e)}")
        
        return {"explanation": error_message, "code": []}, execution_time




# Function to render the chat interface using the agent or direct API
def render_chat_interface(df, api_token, model_name):
    """Render the chat interface for interacting with the data."""
    
    # Create the agent if it doesn't exist yet or if the dataframe has changed
    if "current_agent" not in st.session_state or "current_agent_df_name" not in st.session_state or st.session_state.current_agent_df_name != st.session_state.current_df_name:
        with st.spinner("Initializing data chat interface..."):
            st.session_state.current_agent = create_agent_for_df(df, api_token, model_name)
            st.session_state.current_agent_df_name = st.session_state.current_df_name
    
    # Check if agent creation failed
    if st.session_state.current_agent is None:
        st.error("Failed to initialize the data chat interface. Please check your API token and model selection.")
        return
    
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
                        explanation = explanation.strip()
                        st.write(explanation)
                    
                    if "code" in message["content"] and message["content"]["code"] and st.session_state.show_code:
                        for code in message["content"]["code"]:
                            st.markdown(format_code(code), unsafe_allow_html=True)
                    
                    if "output" in message["content"] and message["content"]["output"]:
                        # Filter out matplotlib warnings from output
                        output = message["content"]["output"]
                        output = re.sub(r'.*?Matplotlib is currently using agg.*?\n', '', output)
                        if output.strip():
                            st.text(output)
                    
                    if "dataframe" in message["content"] and message["content"]["dataframe"] is not None:
                        st.dataframe(message["content"]["dataframe"])
                else:
                    # Filter out matplotlib warnings
                    content = str(message["content"])
                    content = re.sub(r'<string>:.*?cannot show the figure\.', '', content)
                    content = content.strip()
                    st.write(content)
            
            if "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])
            
            if "execution_time" in message:
                st.markdown(f"<span class='status-info' style='float:right'>⏱️ {message['execution_time']:.2f}s</span>", unsafe_allow_html=True)
    
    # Chat input and options
    user_question = st.chat_input("Ask me anything about your data!", disabled=not api_token)
    
    if st.button("Clear Chat 🗑️", type="primary", use_container_width=True):
        st.session_state.chat_messages = []
        
        # Reset the interface
        st.session_state.current_agent = create_agent_for_df(df, api_token, model_name)
        st.session_state.current_agent_df_name = st.session_state.current_df_name
        
        st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_code = st.checkbox("Show generated code", value=st.session_state.show_code)
        if show_code != st.session_state.show_code:
            st.session_state.show_code = show_code
    
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
                try:
                    response, execution_time = generate_response_with_agent(
                        st.session_state.current_agent, 
                        user_question, 
                        show_code=st.session_state.show_code
                    )
                    
                    message_data = {
                        "role": "assistant", 
                        "content": response,
                        "execution_time": execution_time
                    }
                    
                    # Check if there's a security warning
                    if "security_warning" in response and response["security_warning"]:
                        st.markdown(f"<div class='status-danger' style='width: 100%;'>{response['explanation']}</div>", 
                                  unsafe_allow_html=True)
                    else:
                        # Display explanation
                        if "explanation" in response:
                            st.write(response["explanation"])
                        
                        # Display code if requested
                        if st.session_state.show_code and "code" in response and response["code"]:
                            for code in response["code"]:
                                st.markdown(format_code(code), unsafe_allow_html=True)
                        
                        # Display output if any
                        if "output" in response and response["output"]:
                            st.text(response["output"])
                        
                        # Display resulting dataframe if any
                        if "dataframe" in response and response["dataframe"] is not None:
                            st.dataframe(response["dataframe"])
                        
                        # Display figure if any
                        if "figure" in response and response["figure"] is not None:
                            st.pyplot(response["figure"])
                            message_data["figure"] = response["figure"]
                    
                    # Display execution time
                    st.markdown(f"<span class='status-info' style='float:right'>⏱️ {execution_time:.2f}s</span>", unsafe_allow_html=True)
                    
                    # Add a button to save this query as a favorite (only for non-security warnings)
                    if not ("security_warning" in response and response["security_warning"]):
                        if st.button("⭐ Save as favorite"):
                            if user_question not in st.session_state.favorites:
                                st.session_state.favorites.append(user_question)
                                st.success("Added to favorites!")
                    
                    # Add to chat history
                    st.session_state.chat_messages.append(message_data)
                    
                except Exception as e:
                    error_message = f"⚠️ Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_message,
                        "execution_time": 0
                    })
    
    

# Function to render performance metrics
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





# Main application entry point
def main():
    """Main application entry point."""
    st.title("🤗 Chat with Data using Hugging Face Models")
    
    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Check for Hugging Face API token
    if 'huggingface_api_token' not in st.session_state:
        st.session_state.huggingface_api_token = ""
    
    # Sidebar for API token and settings
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🔑 API Settings")

        # Model selection (could be expanded with more options)
        model_name = st.selectbox(
            "Select Model",
            [
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "google/gemma-7b-it",
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
                "tiiuae/falcon-40b-instruct",
            ],
            index=0
            )

    api_token = os.environ.get("HF_API_KEY")
    if api_token:
        st.session_state.huggingface_api_token = api_token
        
    with col2:    
        # Security settings
        st.subheader("⚙️ Settings")
        st.session_state.security_mode = st.radio(
            "Security Mode", 
            ["medium", "high"], 
            index=0 if st.session_state.security_mode == "medium" else 1
        )
    
    # Check if Data is uploaded
    if 'df' not in st.session_state:
        # File uploader widget
        uploaded_file = st.file_uploader("📂 Upload a CSV or Excel file", 
                                          type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # file type
                if uploaded_file.name.endswith('.csv'):
                    # use Polars if available
                    if HAS_POLARS:
                        try:
                            st.session_state.df = pl.read_csv(uploaded_file)
                            st.info("File loaded using Polars")
                        except Exception as e:
                            # Error with Polars, fallback to Pandas
                            logging.warning(f"Error loading with Polars: {str(e)}. Falling back to Pandas.")
                            st.session_state.df = pd.read_csv(uploaded_file)
                    else:
                        st.session_state.df = pd.read_csv(uploaded_file)
                else:  
                    # use Polars if available
                    if HAS_POLARS:
                        try:
                            st.session_state.df = pl.read_excel(uploaded_file)
                            st.info("File loaded using Polars")
                        except Exception as e:
                            # if error with Polars, fallback to Pandas
                            logging.warning(f"Error loading with Polars: {str(e)}. Falling back to Pandas.")
                            st.session_state.df = pd.read_excel(uploaded_file)
                    else:
                        st.session_state.df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading the file: {str(e)}")
        
        st.info("👆 Upload a dataset to begin analyzing with AI models")
        st.stop()
    
    # get available dataframes
    dataframes = {
        "Initial DataFrame": st.session_state.df
    }
    
    # add other dataframes if they exist
    if 'df_processed' in st.session_state and st.session_state.df_processed is not None:
        dataframes["Processed DataFrame"] = st.session_state.df_processed
    
    # filter out None values
    available_dataframes = {k: v for k, v in dataframes.items() if v is not None}
    
    if not available_dataframes:
        st.warning("No dataframes available for analysis.")
        st.stop()
    
    # dataFrame selector
    if len(available_dataframes) >= 1:
        # Just use the initial dataframe without showing a dropdown
        st.session_state.current_df_name = "Initial DataFrame"
        selected_df = available_dataframes["Initial DataFrame"]
        st.info("Using Initial DataFrame")
    
    

    # content area
    selected_df = available_dataframes[st.session_state.current_df_name]    
    
    # Render performance metrics
    render_performance_metrics()
    
    # Render chat interface
    if st.session_state.huggingface_api_token:
        render_chat_interface(selected_df, st.session_state.huggingface_api_token, model_name)
    else:
        st.warning("⚠️ Please enter your Hugging Face API token in the sidebar to start chatting with your data.")

# Run the main application
if __name__ == "__main__":
    main()