import streamlit as st
from utilities.utils import *
from utilities.load_data import LoadData
import pandas as pd
from dotenv import load_dotenv
import os

# page config
st.set_page_config(
    page_title="R.I.D.E.",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# .env & api key
load_dotenv()
default_openai_api_key = os.getenv('OPENAI_API_KEY')


# session state handling
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None
if 'is_user_provided_key' not in st.session_state:
    st.session_state.is_user_provided_key = False
if 'data_source' not in st.session_state:
    st.session_state.data_source = "Upload Data"

# api key to be used
def update_api_key():
    input_key = st.session_state.api_key_input
    if input_key:
        st.session_state.openai_api_key = input_key
        st.session_state.is_user_provided_key = True
    else:
        st.session_state.openai_api_key = None
        st.session_state.is_user_provided_key = False

url = "https://sudhanshumukherjeexx.github.io/ride-docs"
st.sidebar.markdown("🗎 **View Documentation:** [click here](%s)" % url)

# sidebar
sidebar_style = """
<style>
    .small-font {
        font-size: 16px;
        font-weight: normal;
    }
</style>
"""
st.sidebar.markdown(sidebar_style, unsafe_allow_html=True)
st.sidebar.markdown('<div class="small-font">File Formats: CSV | EXCEL | PARQUET</div>', unsafe_allow_html=True)


# dataset - user upload or default
if 'df' not in st.session_state:
    data_source = st.sidebar.radio("Do you want to upload your own data or use a default dataset?", 
                                 ["Upload Data", "Use Default Datasets"],
                                 key="data_source")

    if data_source == "Upload Data":
        uploaded_file = st.sidebar.file_uploader("📂 Choose a file")
        
        # api key
        st.sidebar.text_input("OpenAI API Key", 
                            type="password",
                            key="api_key_input",
                            on_change=update_api_key)

        if uploaded_file is not None:
            file_type = uploaded_file.name.split('.')[-1]
            try:
                df, load_time = LoadData(uploaded_file, file_type)
                st.session_state['df'] = df
                st.success(f"File loaded in {load_time:.4f} seconds.")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error("An error occurred during file loading.")

    # defualt dataset choice
    else:  
        st.sidebar.subheader("Select a Sample Dataset")
        sample_data_options = ["iris", "titanic", "boston", "penguins", "winequality", "adult", "heart", 
                             "weatherHistory", "retail_sales_dataset", "polar_scatter_data", "polar_bar_data", 
                             "FIPS", "LAT_LONG", "2sample_ttest_data", "anova_data", "chi_square_data",
                             "mann_whitney_data", "wilcoxon_data", "kruskal_wallis_data"]
        st.session_state.dataset_name = st.sidebar.selectbox("Choose a dataset", sample_data_options)
        
        # my api key for user test
        st.sidebar.text_input("OpenAI API Key (Optional)", 
                            type="password",
                            key="api_key_input",
                            on_change=update_api_key)

        if st.sidebar.button("Load Dataset"):
            dataset_path = f"datasets/{st.session_state.dataset_name}.csv"
            try:
                df, load_time = LoadData(dataset_path, 'csv')
                st.session_state['df'] = df
                st.sidebar.success(f"Default dataset '{st.session_state.dataset_name}' loaded.")
                if not st.session_state.is_user_provided_key:
                    st.session_state.openai_api_key = default_openai_api_key
            except FileNotFoundError:
                st.sidebar.error(f"Dataset '{st.session_state.dataset_name}' not found in the datasets folder.")
            except Exception as e:
                st.sidebar.error(f"An error occurred while loading the dataset: {str(e)}")


# dataset upload messages
if 'df' in st.session_state:
    st.success("Dataset successfully loaded! You can now proceed with data exploration and analysis.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Please upload a file to get started or choose from default datasets..", icon="🛸")
    with col2:
        st.info("OpenAI API key helps you unlock advanced features.", icon="🔥")
    with col3:
        url = "https://sudhanshumukherjeexx.github.io/ride-docs"
        st.info("🗎 **View Documentation**: Includes a YouTube walkthrough of workflow and details [here](%s)." % url)

# sidebar gif
HORIZONTAL_RED = "images/ride_gif2.gif"
sidebar_logo = HORIZONTAL_RED 
st.logo(sidebar_logo, icon_image=HORIZONTAL_RED)




# ride logo
st.image("images/ride_gif1.png", use_container_width=True)

st.markdown("<h4 style='text-align: center;'><i>No Code. No Hassle. Just Results</i>.</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload your dataset, explore analytics, visualize trends, and leverage machine learning. All in a No-Code environment, backed by AI.</p>", unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    st.info("**Get Started:** Upload your dataset or choose from default datasets.", icon="📁")
with col4:
    st.info("**Unlock AI Features:** Enter your OpenAI API key to enable powerful AI-driven functionalities.", icon="👽")
with col5:
    st.info("**Test AI Capabilities:** Use default datasets to explore and test AI features.", icon="🧪")


# footer
st.divider()
footer = """
<div style='text-align: center; padding: 20px;'>
    <p style='font-weight: bold;'>Developed and Maintained by Sudhanshu Mukherjee</p>
    <a href="https://www.linkedin.com/in/sudhanshumukherjeexx/" target="_blank">
        <img src="data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+TGlua2VkSW48L3RpdGxlPjxwYXRoIGQ9Ik0yMC40NDcgMjAuNDUyaC0zLjU1NHYtNS41NjljMC0xLjMyOC0uMDI3LTMuMDM3LTEuODUyLTMuMDM3LTEuODUzIDAtMi4xMzYgMS40NDUtMi4xMzYgMi45Mzl2NS42NjdIOS4zNTFWOWgzLjQxNHYxLjU2MWguMDQ2Yy40NzctLjkgMS42MzctMS44NSAzLjM3LTEuODUgMy42MDEgMCA0LjI2NyAyLjM3IDQuMjY3IDUuNDU1djYuMjg2ek01LjMzNyA3LjQzM2MtMS4xNDQgMC0yLjA2My0uOTI2LTIuMDYzLTIuMDY1IDAtMS4xMzguOTItMi4wNjMgMi4wNjMtMi4wNjMgMS4xNCAwIDIuMDY0LjkyNSAyLjA2NCAyLjA2MyAwIDEuMTM5LS45MjUgMi4wNjUtMi4wNjQgMi4wNjV6bTEuNzgyIDEzLjAxOUgzLjU1NVY5aDMuNTY0djExLjQ1MnpNMjIuMjI1IDBIMS43NzFDLjc5MiAwIDAgLjc3NCAwIDEuNzI5djIwLjU0MkMwIDIzLjIyNy43OTIgMjQgMS43NzEgMjRoMjAuNDUxQzIzLjIgMjQgMjQgMjMuMjI3IDI0IDIyLjI3MVYxLjcyOUMyNCAuNzc0IDIzLjIgMCAyMi4yMjIgMGguMDAzeiIvPjwvc3ZnPg==" width="30" height="30" alt="LinkedIn"/>
    </a>
    <a href="https://github.com/sudhanshumukherjeexx/" target="_blank">
        <img src="data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+R2l0SHViPC90aXRsZT48cGF0aCBkPSJNMTIgLjI5N2MtNi42MyAwLTEyIDUuMzczLTEyIDEyIDAgNS4zMDMgMy40MzggOS44IDguMjA1IDExLjM4NS42LjExMy44Mi0uMjU4LjgyLS41NzcgMC0uMjg1LS4wMS0xLjA0LS4wMTUtMi4wNC0zLjMzOC43MjQtNC4wNDItMS42MS00LjA0Mi0xLjYxQzQuNDIyIDE4LjA3IDMuNjMzIDE3LjcgMy42MzMgMTcuN2MtMS4wODctLjc0NC4wODQtLjcyOS4wODQtLjcyOSAxLjIwNS4wODQgMS44MzggMS4yMzYgMS44MzggMS4yMzYgMS4wNyAxLjgzNSAyLjgwOSAxLjMwNSAzLjQ5NS45OTguMTA4LS43NzYuNDE3LTEuMzA1Ljc2LTEuNjA1LTIuNjY1LS4zLTUuNDY2LTEuMzMyLTUuNDY2LTUuOTMgMC0xLjMxLjQ2NS0yLjM4IDEuMjM1LTMuMjItLjEzNS0uMzAzLS41NC0xLjUyMy4xMDUtMy4xNzYgMCAwIDEuMDA1LS4zMjIgMy4zIDEuMjMuOTYtLjI2NyAxLjk4LS4zOTkgMy0uNDA1IDEuMDIuMDA2IDIuMDQuMTM4IDMgLjQwNSAyLjI4LTEuNTUyIDMuMjg1LTEuMjMgMy4yODUtMS4yMy42NDUgMS42NTMuMjQgMi44NzMuMTIgMy4xNzYuNzY1Ljg0IDEuMjMgMS45MSAxLjIzIDMuMjIgMCA0LjYxLTIuODA1IDUuNjI1LTUuNDc1IDUuOTIuNDIuMzYuODEgMS4wOTYuODEgMi4yMiAwIDEuNjA2LS4wMTUgMi44OTYtLjAxNSAzLjI4NiAwIC4zMTUuMjEuNjkuODI1LjU3QzIwLjU2NSAyMi4wOTIgMjQgMTcuNTkyIDI0IDEyLjI5N2MwLTYuNjI3LTUuMzczLTEyLTEyLTEyIi8+PC9zdmc+" width="30" height="30" alt="GitHub"/>
    </a>
    <a href="mailto:smukherjee3@umassd.edu" target="_blank">
        <img src="data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48dGl0bGU+R21haWw8L3RpdGxlPjxwYXRoIGQ9Ik0yNCA1LjQ1N3YxMy45MDljMCAuOTA0LS43MzIgMS42MzYtMS42MzYgMS42MzZoLTMuODE5VjExLjczTDEyIDE2LjY0IDUuNDU1IDExLjczdjkuMjczSDEuNjM2QS43MzQuNzM0IDAgMCAxIC45IDIwLjUzNlY1LjQ1N2MwLTIuMDIzIDIuMzA5LTMuMTc4IDMuOTI3LTEuOTY0TDUuNDU1IDQuMDkxIDEyIDkgMTguNTQ1IDQuMDlsLjYyNy0uNTk3YzEuNjE5LTEuMjE0IDMuOTI4LS4wNiAzLjkyOCAxLjk2NHoiLz48L3N2Zz4=" width="30" height="30" alt="Email"/>
    </a>
</div>

"""
st.markdown(footer, unsafe_allow_html=True)
st.markdown("<div style='text-align: center;'>Copyright (c) 2025 Sudhanshu Mukherjee</div>", unsafe_allow_html=True)