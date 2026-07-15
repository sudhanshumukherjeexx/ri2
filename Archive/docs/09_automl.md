## 📘 RIDE User Manual – Panel 9: 🤖 AutoML (Machine Learning Explorer)

---
### 🚀 Purpose of the Panel

The **AutoML Panel** helps users perform **automated machine learning** (AutoML) tasks for:
- Regression
- Classification
- Clustering  
### Without writing a single line of code, users can:
- Select a dataset
- Pick features and a target
- Choose between step-by-step or built-in preprocessing
- Run advanced models
- Explore evaluation metrics for the models
- Export predictions

It integrates a powerful backend to run model pipelines, manage parallel execution, and rank models by performance.

---
### 🧭 User Workflow

1. **Upload Dataset & Choose Variant** Choose from Initial, Imputed, Scaled, or Encoded DataFrames.
2. **Select Task**
    - Regression Analysis
    - Classification Analysis
    - Clustering Analysis
    - (Time Series – coming soon)

3. **Choose Columns**
    - Select features and a target variable.
    - Handles missing targets intelligently.

4. **Preprocessing Mode**
    - Choose between:
        - ⚙️ Step-by-step preprocessing via previous panels
        - 🔄 One-click automatic pipeline (`eval.py` handles everything inside the model)

5. **Run Analysis**
    - Press a button to start training.
    - Behind the scenes, dozens of models are evaluated.

6. **Results**
    - View top-performing models
    - See metrics like R², Adjusted R², RMSE, Accuracy, F1, ROC-AUC
    - Download results and predictions

7. **Clustering**
    - Choose standard or PCA-based clustering
    - Visualize clusters in 2D space
    - Download labeled datasets

---
### 🤖 How It Works Internally – `eval.py`

- **Model Registry**: Dynamically filters usable models based on dataset size.
- **Preprocessing**:
    - Numeric → Imputed with mean, scaled with `StandardScaler`
    - Low-cardinality categorical → OneHotEncoded
    - High-cardinality categorical → OrdinalEncoded
- **Parallel Execution**: Uses `ProcessPoolExecutor` to run models in parallel.
- **Memory Management**: Skips memory-intensive models for large datasets.
- **Metric Calculation**:
    - Classification: Accuracy, Balanced Accuracy, F1 Score, ROC-AUC
    - Regression: R², Adjusted R², RMSE
- **Model Filtering**: Automatically excludes unsuitable models for your task.

---
## 🧪 Key Metrics Explained

### 🔢 Regression

| Metric          | Formula                                       | Description                                                                                                                                                                                                                                                     |
| --------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R²**          | $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$         | Measures the proportion of variance explained by the model. Values range from 0 to 1, with higher values indicating better fit. An R² of 0.7 means 70% of the outcome variation is explained by the model.                                                      |
| **Adjusted R²** | $1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$      | Modified version of R² that penalizes for adding unnecessary predictors. Prevents overfitting by accounting for model complexity. Always lower than R², and may decrease when irrelevant features are added.                                                    |
| **RMSE**        | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ | Root Mean Square Error measures the average prediction error in the same units as the dependent variable. Lower values indicate better fit. Useful for understanding practical significance of model errors and comparing models with the same target variable. |

---
### ✅ Classification

| Metric                | Formula                                                      | Description                                   |
| --------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| **Accuracy**          | $\frac{TP + TN}{TP + TN + FP + FN}$                         | Measures the proportion of correctly classified instances (both positive and negative). Simple but potentially misleading with imbalanced datasets. A 99% accuracy could mean simply predicting the majority class in a dataset with 99:1 class ratio. |
| **Balanced Accuracy** | $\frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$ | Average of sensitivity and specificity, giving equal weight to both classes regardless of their size. Better for imbalanced datasets as it prevents models from achieving high scores by simply predicting the majority class. |
| **F1 Score**          | $2 \cdot \frac{precision \cdot recall}{precision + recall} = \frac{2TP}{2TP + FP + FN}$ | Harmonic mean of precision and recall, balancing both false positives and false negatives. Particularly useful when class imbalance exists and both false positives and negatives have significant consequences. Values range from 0 to 1. |
| **ROC-AUC**           | Area under the ROC curve plotting True Positive Rate vs False Positive Rate | Measures model's ability to discriminate between classes across all possible classification thresholds. A value of 0.5 indicates random guessing, 1.0 is perfect classification. Less affected by class imbalance than accuracy. Shows model's ranking ability rather than its calibration. |

---
### 🔀 Clustering Metrics

| Metric                     | Description                                                     |
| -------------------------- | --------------------------------------------------------------- |
| **Inertia**                | Sum of squared distances within clusters (used in elbow method) |
| **Silhouette Score**       | Measures separation between clusters (ranges from -1 to 1)      |
| **PCA Explained Variance** | Shows how much variance each principal component retains        |
