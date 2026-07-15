## 📘 RIDE User Manual – Panel 6: **Feature Scaling & Transformation**

---
### 📊 **Purpose of the Panel**

This panel allows users to **normalize, standardize, or transform features** for better performance in machine learning models. Proper scaling and transformation:

- Makes algorithms converge faster.
- Prevents features with larger magnitudes from dominating the model.
- Handles skewness and non-normal distributions.

---
### 🧭 **User Workflow**

1. **Upload Dataset**  
    Choose from:
    - Initial DataFrame
    - After Missing Value Imputation
    - After Feature Encoding

2. **Select Scaling or Transformation Method**  
    Choose from 8 methods split into two categories:    
    - **Feature Scaling**
    - **Feature Transformation**

3. **Choose Features to Scale**  
    Select one or more numeric columns.

4. **View Results**  
    - Preview scaled data
    - See summary stats before and after scaling
    - Compare original vs transformed distributions

5. **Download Scaled Data**  
    Download the transformed dataset for modeling.

---
### 💻 Features Breakdown

| Feature                   | Description                                                        |
| ------------------------- | ------------------------------------------------------------------ |
| Source Selector           | Choose between raw, imputed, or encoded datasets.                  |
| Scaling Method Selector   | Includes both scaling and transformation strategies.               |
| Numeric Column Detection  | Uses utility to auto-select numeric features.                      |
| Before/After Summary      | Side-by-side stats before and after scaling.                       |
| Distribution Comparison   | Interactive histogram to compare original vs scaled distributions. |
| Download Transformed Data | Exports the result as a CSV.                                       |

---
## 🔧 Scaling & Transformation Methods

### 📏 Feature Scaling

| Method                        | Description                           | Best For                                                   |
| ----------------------------- | ------------------------------------- | ---------------------------------------------------------- |
| **Min-Max Scaling**           | Rescales values to [0, 1] range.      | When bounded input is required (e.g., image pixel values). |
| **Standardization (Z-score)** | Centers data with μ=0 and σ=1.        | When data needs normalization for gradient-based models.   |
| **Robust Scaler**             | Uses IQR (Q3 - Q1), ignores outliers. | When outliers are present and shouldn't dominate scaling.  |
| **Max AbsScaler**             | Scales by max absolute value.         | When dealing with **sparse** data (e.g., TF-IDF).          |

### 🔄 Feature Transformation

|Method|Description|Best For|
|---|---|---|
|**Quantile Transformer**|Converts feature to a uniform distribution.|When feature values are heavily skewed.|
|**Log Transformer**|Applies `log(1 + x)`.|Right-skewed distributions (e.g., income, prices).|
|**Power Transformer (Box-Cox)**|Normalizes data using λ parameter.|Positive-only data with non-normal shape.|
|**Power Transformer (Yeo-Johnson)**|Modified Box-Cox, supports negatives.|Mixed-sign numeric data needing normalization.|

---
## 🧠 Why This Panel Matters

- Scaling ensures model **convergence** and **fair weight distribution**.
- Transformation can reduce **skewness** and make data more **Gaussian**, which is preferred by many statistical and ML models.