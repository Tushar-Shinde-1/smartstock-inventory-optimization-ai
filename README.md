# Predictive Inventory Optimization for Footwear Wholesale Distribution

This project implements a complete machine learning pipeline (following the CRISP-ML(Q) methodology) to forecast product demand, optimize safety stock, and generate business insights for a footwear wholesale distributor.

## Prerequisites

Make sure you have Python 3.10+ installed along with the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`

You can install them via pip if needed:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## How to Run the Pipeline

To execute the project and generate the findings from scratch, run the following commands sequentially from the project root directory.

### Step 1: Data Preparation
Clean the raw data, execute feature engineering (lag, rolling windows, temporal), perform scaling/encoding, and perform a time-based train-test split.
```bash
python create_preparation.py
```
*Outputs:* `Prepared_Dataset.csv`, `Train_Dataset.csv`, `Test_Dataset.csv`, `preprocessing_artifacts.pkl`, and a Jupyter notebook (`03_Data_Preparation.ipynb`).

### Step 2: Model Training
Train and evaluate models (Linear Regression, Random Forest, XGBoost). This performs hyperparameter tuning, 5-fold cross-validation, and selects the best model.
```bash
python run_modeling.py
```
*Outputs:* `best_model.pkl` (the trained model) and `dashboard/model_results.json`.

*(Note: You can also use `python create_modeling.py` to generate an interactive `04_Modeling.ipynb` notebook detailing this process.)*

### Step 3: Generate Insights (SmartStock)
Leverage the best model to forecast demand for the recent 30 days, compute Safety Stock / Reorder Points, detect deadstock, and summarize actionable restock recommendations.
```bash
python generate_insights.py
```
*Outputs:* `dashboard/smartstock_insights.json`.

### Step 4: Create Evaluation Documentation
Generate the formal CRISP-ML(Q) Phase 5 Evaluation document.
```bash
python create_evaluation_doc.py
```
*Outputs:* `05_Evaluation_Documentation.docx`.

### Step 5: Start the Analytics Dashboard
The project includes an HTML/JS dashboard that consumes the generated JSON files to present the project's insights interactively.
To view it, start a simple HTTP server from within the `dashboard` directory:

```bash
cd dashboard
python -m http.server 8000
```
Then, open your web browser and navigate to: [http://localhost:8000](http://localhost:8000)
