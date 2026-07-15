## 📘 RIDE User Manual – Panel 7: **Statistical Data Exploration**

---
### 📊 **Purpose of the Panel**

The **Distribution Diagnostics Panel** allows users to:
- Understand data shape via **skewness** and **kurtosis**
- Test for **normality** with Q-Q plots and statistical tests
- **Detect outliers** using the IQR method

This is crucial for identifying data anomalies, choosing suitable transformations, and selecting proper machine learning models.

---
### 🧭 **User Workflow**

1. **Upload Dataset**  
    Choose from:
    - Initial DataFrame
    - After Missing Value Imputation
    - After Feature Encoding
    - After Feature Scaling

2. **Navigate the Tabs**:
    - **📊 Distribution & Q-Q Plots**: Visualize histograms + Q-Q plots, run normality tests.
    - **🏔️ Kurtosis**: Check how “peaked” or “flat” distributions are.
    - **↗️ Skewness**: Explore asymmetry in distributions.
    - **🔍 Outliers**: Detect extreme values using IQR logic.

3. **Interactive Visuals**:
    - Select numeric features.
    - Compare histograms and Q-Q plots.
    - Get metric summaries and AI insights.

---
### 💻 Features Breakdown

|Feature|Description|
|---|---|
|Data Source Selection|Choose which version of dataset to analyze.|
|Distribution Analysis|Histogram + KDE + Q-Q Plot.|
|Statistical Tests|Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling.|
|Skewness Panel|Visual and numeric skewness for all columns.|
|Kurtosis Panel|Measures tailedness; interprets platykurtic vs leptokurtic.|
|Outlier Detection|Uses IQR method; gives upper/lower bounds, values, and percentage.|
|AI Explanation|GPT-generated summaries for kurtosis and skewness.|

---
## 🔍 What Each Technique Tells You

### 🟩 Skewness

| Type    | Interpretation                  | When It Matters                   |
| ------- | ------------------------------- | --------------------------------- |
| **= 0** | Symmetric distribution (Normal) | Good baseline for modeling        |
| **> 0** | Right-skewed (long right tail)  | Log transformation may help       |
| **< 0** | Left-skewed (long left tail)    | Square-root or Box-Cox might help |

---
### 🟪 Kurtosis

| Type    | Interpretation             | When It Matters                      |
| ------- | -------------------------- | ------------------------------------ |
| **≈ 0** | Normal-tailed (mesokurtic) | Safe for most parametric tests       |
| **> 0** | Heavy-tailed (leptokurtic) | Higher chance of outliers            |
| **< 0** | Light-tailed (platykurtic) | More uniform, less prone to extremes |

---
### 🟥 Q-Q Plot

- **Compares quantiles** of sample distribution to a normal distribution.
- **S-shaped curve**: Skewed data.
- **Straight line**: Normal distribution.

---
### 🟨 Statistical Tests for Normality

| Test                   | Description                                                        | When It’s Used                   |
| ---------------------- | ------------------------------------------------------------------ | -------------------------------- |
| **Shapiro-Wilk**       | Most powerful for n < 5000                                         | Small to medium datasets         |
| **Kolmogorov-Smirnov** | Compares empirical vs normal distribution (uses μ and σ from data) | For larger datasets              |
| **Anderson-Darling**   | Strong test across all sizes                                       | Offers critical value comparison |

---
### 🧮 Outlier Detection (IQR Method)

- **IQR = Q3 - Q1**
- **Outliers**:
    - Below: Q1 − 1.5 × IQR
    - Above: Q3 + 1.5 × IQR
- Users can **tune k (IQR multiplier)** to make detection more or less sensitive.
