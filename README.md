# Automated Data Science Toolkit (ADST)

A Streamlit-based mini data analysis toolkit that helps users upload CSV/XLSX files, inspect data quality, clean missing values, explore EDA dashboards, build simple Linear Regression models, and export analysis reports.

## Project Goals

Many users have raw datasets but do not know how to clean, explore, or analyze them properly. ADST aims to provide a guided workflow:

```text
Upload Data
- Inspect Dataset
- Clean Data
- Explore EDA Dashboard
- Train Basic Linear Regression Model
- Export Report
```

## Main Features

1. Multi-file Upload
   - Upload one or multiple CSV/XLSX files.
   - Suitable for e-commerce datasets with multiple related tables.

2. Data Catalog 
   - Show rows, columns, missing cells, duplicate rows, numeric columns, and categorical columns for each table.

3. Relationship Builder
   - Suggest possible relationships between tables based on shared column names.
   - Allow users to merge two tables manually.

4. Auto Cleaning
   - Remove duplicate rows.
   - Detect missing values.
   - Handle missing values based on missing rate and column type.
   - Show cleaning logs for transparency.

5. EDA Dashboard
Current EDA fetures include:
   - Data Preview.
   - Descriptive statistics.
   - Distribution charts.
   - Boxplots.
   - Scatter plots with OLS trendline.
   - Correlation heatmap.
   - Top correlated features.

6. Linear Regression
   - Train / Validation / Test split: 70 / 15 / 15.
   - Show R², RMSE, MAE.
   - Show model coefficients and Actual vs Predicted chart.

7. Export Report
   - Export HTML report.
   - Download cleaned or merged dataset.

## Recommend Use Case
ADST can be used with general CSV/XLSX datasets, but it is especially suitable for e-commerce datasets.

Example e-commerce analysis tasks:
   - Analyze order value.
   - Explore customer behavior.
   - Study product performance.
   - Investigate review scores.
   - Examine payment patterns.
   - Build a simple model to estimate a numeric target.

## Project Structure
```
Automated-Data-Science-Toolkit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── cleaner.py
│   ├── relationships.py
│   ├── eda.py
│   ├── ecommerce.py
│   ├── models.py
│   └── report.py
├── assets/
│   └── screenshots/
└── sample_data/
    └── README.md
```

## File Explanation
app.py 
Main Streamlit interface. It controls the layout, sideabar, tabs, and user workflow.

src/loader.py
Handles single-file and multi-file upload logic.

src/cleaner.py
Contains data quality checks, missing value inspection, auto-cleaning logic, and correlation matrix calculation.

src/relationships.py
Suggests possible relationships between uploaded tables and handles table merging.

src/models.py
Contains the Linear Regression training pipeline and model evaluation metrics.

src/report.py
Generates the exportable HTML analysis report.

requirements.txt
Lists the Python libraries needed to run the project.

## Installation
1. Clone the repository
```
git clone https://github.com/tnamgia76-debug/Automated-Data-Science-Toolkit.git
cd Automated-Data-Science-Toolkit
```
2. Create a virtual environment
For Windows:
```
python -m venv venv
venv\Scripts\activate
```
For macOS/Linux:
```
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run the Streamlit app
```
streamlit run app.py
```

## Required Libraries
Main libraries used in this project:
```
streamliit
pandas
numpy 
plotly
scikit-learn
openpyxl
statsmodels (Needed for the OLS trendline Ploty scatter plots)
```
## Current Limitations

This project is still an MVP/demo version.

Current limitations:

The app does not automatically understand the business meaning of every column.
Multi-table merging requires user confirmation.
Linear Regression is the only machine learning model currently supported.
The cleaning logic is rule-based and may not fit every dataset.
Large datasets may require sampling for faster EDA.
The app is not yet designed as a production-scale SaaS platform.

## Future Improvements

Planned improvements:

Add an e-commerce insight tab.
Add model suitability warnings before training.
Add outlier detection.
Add better missing-value strategy selection.
Add export to PDF.
Add charts into the exported report.
Add user authentication for deployed versions.
Improve performance for large datasets.
Add deployment guide.

## Notes on Reliability

ADST is designed to assist analysis, not replace human judgment.

The app can suggest cleaning actions, relationships, correlations, and model results, but users should still review the output carefully before making decisions.

Correlation and Linear Regression results should not be interpreted as proof of causality.

## Author

Developed as a personal data science and Streamlit project.
