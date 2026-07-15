## 📘 RIDE User Manual – Panel 1: **Data Overview**

---

### 🔍 **Purpose of this Panel**

The **Data Overview panel** is designed to give users an instant, interactive understanding of their dataset. It provides descriptive statistics, data types, shape, missing value counts, and correlation analysis—all wrapped with optional AI-driven explanations.

---

### 🧭 **User Workflow**

1. **Add API Key (on Home Page)**  
    The user needs to first input their OpenAI API key for GPT-powered insights to work.
    
2. **Upload Dataset**  
    The dataset must be uploaded. The tool expects a tabular format (e.g., CSV, Excel).
    
3. **View Header & Dataset Cards**  
    A GIF and a visual introduction describe what this panel does.
    
4. **Explore Tabs:**
    
    - 📊 **Basic Statistics**
        
        - Preview of top rows of the dataset.
            
        - Descriptive statistics using `.describe()`.
            
        - Info about shape, features, data types, and missing values.
            
        - Option to trigger **AI analysis** of summary statistics.
            
    - 🔄 **Correlations**
        
        - Correlation matrix for numeric features.
            
        - Plotly heatmap visualization.
            
        - Optional GPT-based insights for both matrix and heatmap.
            

---

### 💻 Features Breakdown

|Feature|Description|
|---|---|
|Dataset Preview|Shows the first few rows of the uploaded dataset.|
|Statistical Summary|Displays count, mean, std, min, max, and quartiles for each numeric column.|
|Shape and Columns|Shows number of rows/columns and feature names.|
|Data Types|Indicates each column’s datatype (int, float, object, etc.).|
|Missing Values Count|Identifies which columns have missing data and how much.|
|Correlation Matrix|Matrix showing linear relationships between numeric features.|
|Correlation Heatmap|Heatmap using Plotly to visualize strong/weak relationships.|
|AI-Powered Insights|GPT-generated explanations for summary stats and heatmap analysis.|

---

### 🤖 AI Features

- **LLM (GPT-4o-mini)** analyzes:
    
    - Statistical distribution
        
    - Data trends
        
    - Feature relationships
        
- **Image-based GPT analysis** of the correlation heatmap using the `AnalyzeImage()` function.