## 📘 RIDE User Manual – Panel 10: 🧠 Chat with Data (OpenAI + Hugging Face)

---
### 🎯 **Purpose of the Panel**

The **Chat with Data** panel empowers users to explore, analyze, and visualize their dataset through **natural language prompts**. It integrates with:

- **OpenAI GPT Models** (e.g., `gpt-4o`, `gpt-4o-mini`)
- **Hugging Face Transformers** (e.g., Mixtral, Mistral, Gemma, Falcon)

Users can ask questions about:
- Dataset structure
- Summary statistics
- Missing values
- Trends, correlations
- Visualizations (bar, line, scatter, heatmaps)
- Python code generation for transformations or EDA

---
### 🧭 **User Workflow**

1. **Upload Dataset**
    - Upload CSV or Excel file.
    - The dataset is automatically read and shown in a preview.

2. **Choose Model Provider**
    - 🧠 OpenAI (requires OpenAI API key)
    - 🤗 Hugging Face (requires Hugging Face API token)

3. **Select a DataFrame**
    - Choose from initial, encoded, imputed, or scaled versions.

4. **Ask Questions**
    - Examples:
        - “Which column has the most missing values?”
        - “Plot a bar chart of product categories.”
        - “How is age correlated with income?”

5. **Receive Answers**
    - Get responses in plain English.
    - Includes Python code blocks if applicable.
    - Auto-executed plots rendered inline.

6. **Track Execution**
    - Track query response times and performance metrics.

---
## 🤖 How It Works

### 🔐 OpenAI Mode

- Integrates with `openai.ChatCompletion`.
- Adds security filters to reject suspicious code (e.g., `eval()`, `subprocess`).
- Uses `safe_execute_code()` to interpret and safely execute Python code blocks.
- Tracks execution time and logs user queries.
- Caches formatting using `st.cache_data`.

### 🤗 Hugging Face Mode

- Uses `InferenceClient` from `huggingface_hub`.
- Formats prompts using `Mixtral`-style `[INST]` tokens.
- Directly queries LLMs with the full DataFrame context.
- Parses responses, extracts Python code blocks and executes them for visualizations.

---
## 🧠 AI Safety and Security

| Safety Feature                   | Description                                          |
| -------------------------------- | ---------------------------------------------------- |
| **Suspicious Pattern Detection** | Blocks unsafe queries (e.g., `drop table`, `open()`) |
| **Resource Limit Guarding**      | Automatically samples large datasets                 |
| **Timeout Mechanism**            | Halts long-running code after 30 seconds             |
| **Restricted Libraries**         | Only pandas, numpy, matplotlib, seaborn permitted    |
