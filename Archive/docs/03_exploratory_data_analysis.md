## 📘 RIDE User Manual – Panel 3: **Exploratory Data Analysis (EDA)**

---
### 📊 **Purpose of the Panel**

The **EDA Panel** helps users visualize patterns, distributions, and relationships in their data using a suite of 15+ interactive plot types. These visuals make it easier to understand underlying structures before performing modeling or transformation.

---
### 🧭 **User Workflow**

1. **Upload Dataset & Select Preprocessed Version**  
    User selects from:
    - Initial DataFrame
    - After Missing Value Imputation
    - After Feature Scaling
    - After Feature Encoding

2. **Header & Overview**  
    GIF and a markdown introduction help the user understand what the EDA panel does.
    
3. **Visual Grid**  
    Users are presented with a categorized set of plotting options:
    - **Basic Plots**
    - **Advanced Plots**
    - **Specialized Plots**
    - **Geospatial Visualizations**
    
4. **Click to Generate**  
    On clicking a plot type:
    - The corresponding visualization function is invoked.
    - The selected plot appears with a title and a back button.

---
### 💻 Features Breakdown

|Feature|Description|
|---|---|
|Data Source Selector|Allows switching between raw and processed versions of the dataset.|
|Plot Grid|Shows image previews and labels of 15+ supported plots.|
|One-click Plotting|Clicking a plot button immediately renders the visual.|
|Plot Functions|Powered by modular functions like `plot_histogram`, `plot_boxplot`, etc.|
|Back Navigation|Users can return to the grid easily via a back button.|

---
### 📊 Visualization Types Supported
#### 🔹 Basic Plots

| Plot Type        | Purpose / Use Case                                                                                                               |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Box Plot**     | Shows the spread and skewness of data via quartiles and outliers. ✅ Best for comparing feature distributions, spotting outliers. |
| **Histogram**    | Displays frequency distribution of numeric values. ✅ Helps detect skewness, modality, and data concentration.                    |
| **Scatter Plot** | Visualizes relationships between two continuous variables. ✅ Useful for detecting correlations and clusters.                     |
| **Bar Chart**    | Compares quantities of categorical variables. ✅ Ideal for viewing counts or aggregations across categories.                      |
| **Pie Chart**    | Shows proportional breakdown of categorical variables. ✅ Good for visualizing share/percentage composition.                      |
| **Line Plot**    | Displays trends across time or ordered observations. ✅ Best for time-series or sequential data analysis.                         |

---
#### 🔸 Advanced Plots

|Plot Type|Purpose / Use Case|
|---|---|
|**2D Hist Contour**|Combines density-based histogram with contour lines. ✅ Useful when you want to detect dense regions in bivariate data.|
|**Contour Plot**|Displays 3D surface on a 2D plane using contours. ✅ Ideal for understanding gradient/response surfaces.|
|**Violin Plot**|Combines box plot and kernel density plot. ✅ Great for understanding the distribution shape and symmetry.|
|**3D Scatter**|Visualizes 3D relationships between 3 continuous variables. ✅ Useful for dimensional analysis and observing clusters.|
|**3D Line**|Connects points in 3D space. ✅ Helpful for time-based sequences across three dimensions.|

---
#### 🎯 Specialized Plots

| Plot Type         | Purpose / Use Case                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Polar Scatter** | Plots data on a circular axis (angle and radius). ✅ Best for directional or cyclical data (e.g., wind direction, time). |
| **Polar Bar**     | Like a bar chart in a circular layout. ✅ Useful for periodic or circular metrics (e.g., sales across 12 months).        |

---
#### 🌍 Geospatial Visualizations

| Plot Type          | Purpose / Use Case                                                                                                                     |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Scatter Map**    | Plots points on a geographical map. ✅ Excellent for visualizing locations (e.g., store distribution, incident reports).                |
| **Choropleth Map** | Colors geographic areas based on a variable. ✅ Great for showing demographic density, income levels, or COVID spread.                  |
| **Bubble Map**     | Uses bubbles to show values over geographical coordinates. ✅ Perfect for showing magnitude across locations (e.g., sales, population). |

---

### 🤖 AI Integration

While this panel does not directly use GPT, it is fully compatible with previous and subsequent AI-assisted panels where generated insights complement these visualizations.
