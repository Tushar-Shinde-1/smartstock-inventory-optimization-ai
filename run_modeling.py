"""
CRISP-ML(Q) Phase 4: Run Modeling Pipeline
Trains LR, RF, XGBoost → Tunes → Cross-validates → Exports results to JSON for dashboard
"""
import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

print("=" * 60)
print("  CRISP-ML(Q) Phase 4: Model Training Pipeline")
print("=" * 60)

# ============================================================
# DATA PREPARATION (inline)
# ============================================================
print("\n[1/8] Loading and preparing data...")
df = pd.read_csv('FootWare_Wholesale_Sales_Dataset.csv')

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df['Margin (%)'] = df['Margin (%)'].str.replace('%', '').astype(float) / 100
df['Tax (GST % )'] = df['Tax (GST % )'].str.replace('%', '').astype(float) / 100

df.drop([c for c in ['Net Profit (₹)', 'Total Revenue (₹)', 'Net Tax (₹)'] if c in df.columns], axis=1, inplace=True)
df.drop([c for c in ['Margin (%)', 'Tax (GST % )', 'Tax Amount (₹)', 'Dealer Location', 'Profit (₹)'] if c in df.columns], axis=1, inplace=True)

df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Day_of_Month'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Quarter'] = df['Date'].dt.quarter
df['Year'] = df['Date'].dt.year
df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)

df = df.sort_values('Date').reset_index(drop=True)
daily = df.groupby('Date')['Quantity Sold'].sum().reset_index()
daily.columns = ['Date', 'Daily_Total_Demand']
for lag in [1, 7, 14, 30]:
    daily[f'Lag_{lag}'] = daily['Daily_Total_Demand'].shift(lag)
for w in [7, 14, 30]:
    daily[f'Rolling_{w}_Mean'] = daily['Daily_Total_Demand'].rolling(window=w).mean()
daily['Rolling_7_Std'] = daily['Daily_Total_Demand'].rolling(window=7).std()

merge_cols = ['Date','Lag_1','Lag_7','Lag_14','Lag_30','Rolling_7_Mean','Rolling_14_Mean','Rolling_30_Mean','Rolling_7_Std']
df = df.merge(daily[merge_cols], on='Date', how='left')
lag_cols = ['Lag_1','Lag_7','Lag_14','Lag_30','Rolling_7_Mean','Rolling_14_Mean','Rolling_30_Mean','Rolling_7_Std']
df = df.dropna(subset=lag_cols).reset_index(drop=True)

encoders = {}
for col in ['Product', 'Brand', 'Dealer']:
    le = LabelEncoder()
    df[col + '_Encoded'] = le.fit_transform(df[col])
    encoders[col] = le

features_to_scale = ['Unit Price (₹)', 'Size', 'Stock Availability'] + lag_cols
scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

feature_columns = [
    'Product_Encoded', 'Brand_Encoded', 'Dealer_Encoded',
    'Size', 'Unit Price (₹)', 'Stock Availability',
    'Day_of_Week', 'Day_of_Month', 'Month', 'Quarter', 'Year',
    'Week_of_Year', 'Is_Weekend', 'Is_Month_Start', 'Is_Month_End',
    'Lag_1', 'Lag_7', 'Lag_14', 'Lag_30',
    'Rolling_7_Mean', 'Rolling_14_Mean', 'Rolling_30_Mean', 'Rolling_7_Std'
]

split_date = '2025-07-01'
train_mask = df['Date'] < split_date
test_mask = df['Date'] >= split_date
X_train = df.loc[train_mask, feature_columns].copy()
X_test = df.loc[test_mask, feature_columns].copy()
y_train = df.loc[train_mask, 'Quantity Sold'].copy()
y_test = df.loc[test_mask, 'Quantity Sold'].copy()

print(f"   Records: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)} | Features: {len(feature_columns)}")

# Helper
def get_metrics(y_true, y_pred):
    return {
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        'R2': round(r2_score(y_true, y_pred), 4),
        'MAPE': round(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100, 2)
    }

results = {}

# ============================================================
# MODEL 1: Linear Regression
# ============================================================
print("\n[2/8] Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_train_p = lr.predict(X_train)
lr_test_p = lr.predict(X_test)
results['Linear Regression'] = {
    'train': get_metrics(y_train, lr_train_p),
    'test': get_metrics(y_test, lr_test_p),
    'coefficients': {f: round(c, 4) for f, c in zip(feature_columns, lr.coef_)}
}
print(f"   Test R²: {results['Linear Regression']['test']['R2']}")

# ============================================================
# MODEL 2: Random Forest
# ============================================================
print("\n[3/8] Training Random Forest (default)...")
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_train_p = rf.predict(X_train)
rf_test_p = rf.predict(X_test)
results['Random Forest'] = {
    'train': get_metrics(y_train, rf_train_p),
    'test': get_metrics(y_test, rf_test_p),
}
print(f"   Test R²: {results['Random Forest']['test']['R2']}")

# ============================================================
# MODEL 3: XGBoost
# ============================================================
print("\n[4/8] Training XGBoost (default)...")
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_train, y_train)
xgb_train_p = xgb.predict(X_train)
xgb_test_p = xgb.predict(X_test)
results['XGBoost'] = {
    'train': get_metrics(y_train, xgb_train_p),
    'test': get_metrics(y_test, xgb_test_p),
}
print(f"   Test R²: {results['XGBoost']['test']['R2']}")

# ============================================================
# TUNING: Random Forest
# ============================================================
print("\n[5/8] Tuning Random Forest (RandomizedSearchCV, 30 iter)...")
rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions={
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5, 0.7]
    },
    n_iter=30, cv=5, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
rf_tuned_train = rf_best.predict(X_train)
rf_tuned_test = rf_best.predict(X_test)
results['Random Forest (Tuned)'] = {
    'train': get_metrics(y_train, rf_tuned_train),
    'test': get_metrics(y_test, rf_tuned_test),
    'best_params': {k: str(v) for k, v in rf_search.best_params_.items()},
    'feature_importance': {f: round(float(i), 4) for f, i in zip(feature_columns, rf_best.feature_importances_)}
}
print(f"   Test R²: {results['Random Forest (Tuned)']['test']['R2']} | Best params: {rf_search.best_params_}")

# ============================================================
# TUNING: XGBoost
# ============================================================
print("\n[6/8] Tuning XGBoost (RandomizedSearchCV, 40 iter)...")
xgb_search = RandomizedSearchCV(
    XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    param_distributions={
        'n_estimators': [200, 300, 500, 700],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0]
    },
    n_iter=40, cv=5, scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_train, y_train)
xgb_best = xgb_search.best_estimator_
xgb_tuned_train = xgb_best.predict(X_train)
xgb_tuned_test = xgb_best.predict(X_test)
results['XGBoost (Tuned)'] = {
    'train': get_metrics(y_train, xgb_tuned_train),
    'test': get_metrics(y_test, xgb_tuned_test),
    'best_params': {k: str(v) for k, v in xgb_search.best_params_.items()},
    'feature_importance': {f: round(float(i), 4) for f, i in zip(feature_columns, xgb_best.feature_importances_)}
}
print(f"   Test R²: {results['XGBoost (Tuned)']['test']['R2']} | Best params: {xgb_search.best_params_}")

# ============================================================
# CROSS-VALIDATION
# ============================================================
print("\n[7/8] Running 5-Fold Cross-Validation...")
cv_results = {}
for name, model in [('Linear Regression', LinearRegression()), ('Random Forest (Tuned)', rf_best), ('XGBoost (Tuned)', xgb_best)]:
    mae_scores = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_results[name] = {
        'mae_mean': round(float(mae_scores.mean()), 4),
        'mae_std': round(float(mae_scores.std()), 4),
        'mae_folds': [round(float(s), 4) for s in mae_scores],
        'r2_mean': round(float(r2_scores.mean()), 4),
        'r2_std': round(float(r2_scores.std()), 4),
        'r2_folds': [round(float(s), 4) for s in r2_scores]
    }
    print(f"   {name}: MAE={cv_results[name]['mae_mean']}±{cv_results[name]['mae_std']}, R²={cv_results[name]['r2_mean']}±{cv_results[name]['r2_std']}")

# ============================================================
# EXPORT RESULTS
# ============================================================
print("\n[8/8] Exporting results...")

# Predictions for scatter plots (sample 200 points for dashboard)
np.random.seed(42)
sample_idx = np.random.choice(len(y_test), min(200, len(y_test)), replace=False)
sample_idx.sort()

predictions = {
    'actual': [int(v) for v in y_test.values[sample_idx]],
    'Linear Regression': [round(float(v), 2) for v in lr_test_p[sample_idx]],
    'Random Forest (Tuned)': [round(float(v), 2) for v in rf_tuned_test[sample_idx]],
    'XGBoost (Tuned)': [round(float(v), 2) for v in xgb_tuned_test[sample_idx]]
}

# Residuals for histograms
def residual_hist(y_true, y_pred, bins=30):
    residuals = y_true - y_pred
    counts, edges = np.histogram(residuals, bins=bins)
    return {
        'counts': [int(c) for c in counts],
        'edges': [round(float(e), 2) for e in edges]
    }

residual_data = {
    'Linear Regression': residual_hist(y_test.values, lr_test_p),
    'Random Forest (Tuned)': residual_hist(y_test.values, rf_tuned_test),
    'XGBoost (Tuned)': residual_hist(y_test.values, xgb_tuned_test)
}

# Determine best model
best_name = max(
    ['Linear Regression', 'Random Forest (Tuned)', 'XGBoost (Tuned)'],
    key=lambda n: results[n]['test']['R2']
)

# Dataset info
dataset_info = {
    'total_records': len(df),
    'train_records': len(X_train),
    'test_records': len(X_test),
    'num_features': len(feature_columns),
    'feature_columns': feature_columns,
    'target': 'Quantity Sold',
    'target_stats': {
        'mean': round(float(y_test.mean()), 2),
        'std': round(float(y_test.std()), 2),
        'min': int(y_test.min()),
        'max': int(y_test.max())
    },
    'date_range': {'train': 'Jan 2023 – Jun 2025', 'test': 'Jul 2025 – Dec 2025'},
    'products': sorted(df['Product'].unique().tolist()),
    'brands': sorted(df['Brand'].unique().tolist()),
}

output = {
    'dataset': dataset_info,
    'model_results': results,
    'cv_results': cv_results,
    'predictions': predictions,
    'residuals': residual_data,
    'best_model': best_name
}

with open('dashboard/model_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Save best model pickle
model_map = {'Linear Regression': lr, 'Random Forest (Tuned)': rf_best, 'XGBoost (Tuned)': xgb_best}
with open('best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model_map[best_name],
        'name': best_name,
        'encoders': encoders,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'metrics': results[best_name]['test']
    }, f)

print(f"\n{'='*60}")
print(f"  ✅ ALL DONE!")
print(f"  Best Model: {best_name}")
print(f"  Test R²:    {results[best_name]['test']['R2']}")
print(f"  Test MAE:   {results[best_name]['test']['MAE']}")
print(f"  Results:    dashboard/model_results.json")
print(f"  Model:      best_model.pkl")
print(f"{'='*60}")
