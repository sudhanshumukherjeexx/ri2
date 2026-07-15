## 📘 RIDE User Manual – Panel 2: **Data Cleanup and Conversion**

---
### 🔄 **Purpose of the Panel**

This panel enables users to **clean their dataset** by:

- Removing duplicate rows
    
- Converting feature columns to appropriate data types  
    These preprocessing steps are crucial for maintaining data quality and preparing datasets for deeper analysis or modeling.

---
### 🧭 **User Workflow**

1. **Upload Dataset**  
    Users first upload a dataset (handled globally in the app).
    
2. **View Panel Intro**  
    A GIF and explanation help the user understand what this panel does.
    
3. **Use Two Tabs:**
    
    - **Handle Duplicates**: Automatically detects and removes duplicates.
        
    - **Convert Data Types**:
        
        - User selects a column and target data type (INT, FLOAT, BOOLEAN, STRING, DATETIME).
            
        - The backend intelligently parses and casts the column.
            
        - Before/after comparison of data types is shown for validation.
            

---
### 💻 Features Breakdown

| Feature                  | Description                                                                |
| ------------------------ | -------------------------------------------------------------------------- |
| Duplicate Row Detection  | Identifies and displays the count of duplicated rows in the dataset.       |
| Remove Duplicates Button | Allows users to remove duplicate rows in one click.                        |
| Data Type Conversion     | Provides dropdowns for selecting a column and a target data type.          |
| Smart Parsing Logic      | Automatically handles tricky conversions like datetime or boolean strings. |
| Type Comparison View     | Displays side-by-side comparison of original and updated data types.       |

---
### 🧠 How Conversions Are Handled Internally

- **INT/FLOAT**:
    
    - Strips non-numeric characters using regex.
        
    - Converts strings only if not already numeric.
        
- **BOOLEAN**:
    
    - Converts “true/yes/1/y” strings to `True`, others to `False`.
        
- **DATETIME**:
    
    - Tries multiple common date formats.
        
    - Uses `polars.str.to_datetime()` with fallback parsing.