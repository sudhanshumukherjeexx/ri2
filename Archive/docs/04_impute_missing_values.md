
## 📘 RIDE User Manual – Panel 4: **Impute Missing Values**

---
### 🧠 **Purpose of the Panel**

This panel allows users to handle missing values in their dataset using **nine distinct strategies** ranging from simple deletion to statistical and ML-inspired methods. The goal is to maintain dataset integrity while preparing for modeling or analysis.

---
### 🧭 **User Workflow**

1. **Upload Dataset**  
    Automatically initializes `df_processed` from the uploaded dataset.
    
2. **Review Current Data**  
    The current dataset (with missing values) is displayed.
    
3. **Missing Values Summary**  
    A visual summary highlights all features with missing values.
    
4. **Imputation Settings**
    - Select a **column** with missing data.
    - Choose an **imputation method**.
    - If applicable, input a specific replacement value.

5. **Apply & View Results**
    - Press the **“Apply Imputation”** button.
    - View/download the processed data.

---
### 💻 Features Breakdown

|Feature|Description|
|---|---|
|Column Selection|Dropdown to choose a column with missing values.|
|Imputation Method Picker|Dropdown with 9 methods to handle missing values.|
|Value Input (Conditional)|User enters value when using “Replace with Specific Value.”|
|Imputation Logic Execution|Custom logic executed depending on the chosen method.|
|Feedback & Result Viewer|Instant success/error message, optional data preview, and downloadable result.|

---

## 🧪 Imputation Methods & Significance

|Method|Explanation|When to Use|
|---|---|---|
|**1. Drop Missing Values**|Removes all rows where the selected column has a missing value. ✅ Simple but risky if data loss is significant.|Use only when missing rows are few and ignorable.|
|**2. Replace with Specific Value**|Replaces missing values with a fixed value entered by the user. ✅ Good for categorical defaults or domain-specific values.|Use when you have contextual knowledge (e.g., 0 = No Response).|
|**3. Forward Fill (ffill)**|Fills each missing value with the **last known value above** it. ✅ Useful for time-series or ordered data.|Use when data has a logical time flow.|
|**4. Backward Fill (bfill)**|Fills missing values with the **next known value below** it. ✅ Similar to ffill but forward-looking.|Use in reverse-time data or last-to-first ordering.|
|**5. Distribution Sampling**|Randomly samples values from the column's distribution (normal approximation). ✅ Keeps variability and prevents overfitting.|Use for continuous, numeric columns with normal-like distributions.|
|**6. Mean Imputation**|Fills missing values with the mean of the column. ✅ Easy, but may reduce variance.|Use with normally distributed, symmetric numeric data.|
|**7. Median Imputation**|Fills with the median of the column. ✅ Better than mean for skewed data.|Use with skewed or outlier-heavy numeric features.|
|**8. Nearest Neighbors**|Estimates missing values based on similarity to nearest rows (distance-based). ✅ Sophisticated but computationally heavier.|Use when strong similarity exists among rows.|
|**9. Interpolation**|Fills values by interpolating between surrounding known values. ✅ Natural fit for numeric, ordered data.|Best for numeric sequences or time-series data.|
