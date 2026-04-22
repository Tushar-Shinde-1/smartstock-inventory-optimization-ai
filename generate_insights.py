"""
SmartStock — Generate Business Intelligence Insights
Demand Forecast | Safety Stock & ROP | Deadstock | Daily Recommendations
"""
import pandas as pd
import numpy as np
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 60)
print("  SmartStock — Generating Business Insights")
print("=" * 60)

# ============================================================
# LOAD DATA + TRAINED MODEL
# ============================================================
print("\n[1/6] Loading data and trained model...")
import os
import requests

supabase_url = os.environ.get('SUPABASE_URL')
supabase_key = os.environ.get('SUPABASE_KEY')

if supabase_url and supabase_key:
    print("   Fetching data from live Supabase 'footwear_database' table...")
    endpoint = f"{supabase_url}/rest/v1/footwear_database"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}"
    }
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        df_raw = pd.DataFrame(response.json())
        print(f"   Successfully fetched {len(df_raw)} records from Supabase.")
    else:
        print(f"   Error fetching from Supabase: {response.text}")
        print("   Falling back to local CSV...")
        df_raw = pd.read_csv('FootWare_Wholesale_Sales_Dataset.csv')
else:
    print("   Supabase credentials not found. Reading from local CSV...")
    df_raw = pd.read_csv('FootWare_Wholesale_Sales_Dataset.csv')

try:
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y')
except ValueError:
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])  # Fallback if format changed


with open('best_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

best_model = artifacts['model']
best_name = artifacts['name']
feature_columns = artifacts['feature_columns']
print(f"   Best model: {best_name}")
print(f"   Records: {len(df_raw)}")

# ============================================================
# PREPARE FULL DATA (same as training pipeline)
# ============================================================
print("\n[2/6] Preparing data...")
df = df_raw.copy()
df['Margin (%)'] = df['Margin (%)'].str.replace('%', '').astype(float) / 100
df['Tax (GST % )'] = df['Tax (GST % )'].str.replace('%', '').astype(float) / 100
df.drop([c for c in ['Net Profit (₹)','Total Revenue (₹)','Net Tax (₹)'] if c in df.columns], axis=1, inplace=True)
df.drop([c for c in ['Margin (%)','Tax (GST % )','Tax Amount (₹)','Dealer Location','Profit (₹)'] if c in df.columns], axis=1, inplace=True)

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

# ============================================================
# INSIGHT 1: DEMAND FORECAST PER PRODUCT
# ============================================================
print("\n[3/6] Generating demand forecasts...")

# Use last 30 days of data as the "current period" for forecasting
last_date = df['Date'].max()
recent = df[df['Date'] >= (last_date - pd.Timedelta(days=30))].copy()
recent['Predicted_Demand'] = best_model.predict(recent[feature_columns])

# Aggregate forecasts per product
product_forecast = []
for product in df_raw['Product'].unique():
    prod_data = recent[recent['Product'] == product]
    if len(prod_data) == 0:
        continue
    avg_actual = df_raw[df_raw['Product'] == product]['Quantity Sold'].mean()
    avg_predicted = prod_data['Predicted_Demand'].mean()
    total_predicted = prod_data['Predicted_Demand'].sum()
    std_demand = df_raw[df_raw['Product'] == product]['Quantity Sold'].std()
    if pd.isna(std_demand): std_demand = 0.0

    product_forecast.append({
        'product': product,
        'avg_daily_demand': round(float(avg_actual), 1),
        'predicted_avg_demand': round(float(avg_predicted), 1),
        'total_predicted_30d': round(float(total_predicted), 0),
        'demand_std': round(float(std_demand), 2),
        'records': int(len(prod_data))
    })

# Per product-brand breakdown
product_brand_forecast = []
for product in df_raw['Product'].unique():
    for brand in df_raw[df_raw['Product'] == product]['Brand'].unique():
        mask = (recent['Product'] == product) & (recent['Brand'] == brand)
        if mask.sum() == 0:
            continue
        subset = recent[mask]
        raw_mask = (df_raw['Product'] == product) & (df_raw['Brand'] == brand)
        raw_sub = df_raw[raw_mask]
        
        std_demand_brand = raw_sub['Quantity Sold'].std()
        if pd.isna(std_demand_brand): std_demand_brand = 0.0

        product_brand_forecast.append({
            'product': product,
            'brand': brand,
            'avg_daily_demand': round(float(raw_sub['Quantity Sold'].mean()), 1),
            'predicted_demand': round(float(subset['Predicted_Demand'].mean()), 1),
            'demand_std': round(float(std_demand_brand), 2),
            'avg_price': round(float(raw_sub['Unit Price (₹)'].mean()), 0),
            'current_stock': round(float(raw_sub['Stock Availability'].mean()), 0),
        })

print(f"   Product-level forecasts: {len(product_forecast)}")
print(f"   Product-Brand forecasts: {len(product_brand_forecast)}")

# ============================================================
# INSIGHT 2: SAFETY STOCK & REORDER POINT
# ============================================================
print("\n[4/6] Calculating Safety Stock & Reorder Points...")

# Assumptions
LEAD_TIME_DAYS = 7  # Supplier lead time in days
SERVICE_LEVEL_Z = 1.65  # Z-score for 95% service level

inventory_optimization = []
for item in product_brand_forecast:
    avg_demand = item['avg_daily_demand']
    std_demand = item['demand_std']
    predicted = item['predicted_demand']

    # Safety Stock = Z × σ_demand × √(Lead_Time)
    safety_stock = round(SERVICE_LEVEL_Z * std_demand * np.sqrt(LEAD_TIME_DAYS), 1)

    # Reorder Point = (Avg Daily Demand × Lead Time) + Safety Stock
    reorder_point = round(avg_demand * LEAD_TIME_DAYS + safety_stock, 1)

    # Economic Order Quantity (simplified Wilson formula)
    # EOQ = √(2 × D × S / H)  where D=annual demand, S=order cost, H=holding cost
    annual_demand = avg_demand * 365
    order_cost = 500  # assumed fixed order cost ₹500
    holding_cost = item['avg_price'] * 0.2  # 20% of price
    if holding_cost > 0:
        eoq = round(np.sqrt(2 * annual_demand * order_cost / holding_cost), 0)
    else:
        eoq = round(avg_demand * 30, 0)

    current_stock = item['current_stock']
    stock_status = 'Critical' if current_stock < reorder_point * 0.5 else (
        'Reorder' if current_stock < reorder_point else (
        'Healthy' if current_stock < reorder_point * 3 else 'Overstock'))

    inventory_optimization.append({
        'product': item['product'],
        'brand': item['brand'],
        'avg_daily_demand': avg_demand,
        'predicted_demand': predicted,
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'eoq': int(eoq),
        'current_stock': int(current_stock),
        'stock_status': stock_status,
        'lead_time': LEAD_TIME_DAYS,
    })

print(f"   Inventory items analyzed: {len(inventory_optimization)}")

# ============================================================
# INSIGHT 3: SLOW-MOVING & DEADSTOCK
# ============================================================
print("\n[5/6] Identifying slow-moving and deadstock items...")

last_90 = df_raw[df_raw['Date'] >= (df_raw['Date'].max() - pd.Timedelta(days=90))]
last_30 = df_raw[df_raw['Date'] >= (df_raw['Date'].max() - pd.Timedelta(days=30))]

stock_health = []
all_avg = df_raw.groupby(['Product', 'Brand'])['Quantity Sold'].mean()
threshold_slow = all_avg.quantile(0.25)
threshold_dead = all_avg.quantile(0.10)

for product in df_raw['Product'].unique():
    for brand in df_raw[df_raw['Product'] == product]['Brand'].unique():
        mask_all = (df_raw['Product'] == product) & (df_raw['Brand'] == brand)
        mask_90 = (last_90['Product'] == product) & (last_90['Brand'] == brand)
        mask_30 = (last_30['Product'] == product) & (last_30['Brand'] == brand)

        total_sold = int(df_raw[mask_all]['Quantity Sold'].sum())
        avg_sold = round(float(df_raw[mask_all]['Quantity Sold'].mean()), 1)
        sold_90d = int(last_90[mask_90]['Quantity Sold'].sum()) if mask_90.sum() > 0 else 0
        sold_30d = int(last_30[mask_30]['Quantity Sold'].sum()) if mask_30.sum() > 0 else 0
        avg_stock = round(float(df_raw[mask_all]['Stock Availability'].mean()), 0)

        # Inventory Turnover = Total Sold / Avg Stock
        turnover = round(total_sold / max(avg_stock, 1), 2)

        # Days of Supply = Avg Stock / Avg Daily Sales
        days_supply = round(avg_stock / max(avg_sold / 30, 0.1), 0) if avg_sold > 0 else 999

        # Classification
        if avg_sold <= threshold_dead:
            health = 'Deadstock'
            action = 'Liquidate — deep discount or bundle'
        elif avg_sold <= threshold_slow:
            health = 'Slow-Moving'
            action = 'Promote — run targeted discounts'
        elif turnover > 8:
            health = 'Fast-Moving'
            action = 'Maintain — ensure consistent stock'
        else:
            health = 'Normal'
            action = 'Monitor — standard replenishment'

        stock_health.append({
            'product': product,
            'brand': brand,
            'total_sold': total_sold,
            'avg_daily_sales': avg_sold,
            'sold_last_90d': sold_90d,
            'sold_last_30d': sold_30d,
            'avg_stock': int(avg_stock),
            'turnover_ratio': turnover,
            'days_of_supply': int(min(days_supply, 999)),
            'health': health,
            'action': action,
        })

health_summary = {}
for item in stock_health:
    h = item['health']
    health_summary[h] = health_summary.get(h, 0) + 1

print(f"   Stock health breakdown: {health_summary}")

# ============================================================
# INSIGHT 4: DAILY RECOMMENDATIONS
# ============================================================
print("\n[6/6] Generating daily recommendations...")

recommendations = []
for opt in inventory_optimization:
    product = opt['product']
    brand = opt['brand']
    current = opt['current_stock']
    rop = opt['reorder_point']
    ss = opt['safety_stock']
    eoq = opt['eoq']

    # Find health for this item
    health_item = next((h for h in stock_health if h['product'] == product and h['brand'] == brand), None)
    health = health_item['health'] if health_item else 'Normal'
    turnover = health_item['turnover_ratio'] if health_item else 5

    # Decision logic
    if health == 'Deadstock':
        alert = 'Deadstock'
        priority = 'Low'
        action = f'Run clearance sale or bundle with fast-movers'
        order_qty = 0
        color = '#ef4444'
    elif current <= ss:
        alert = 'Critical — Below Safety Stock'
        priority = 'Urgent'
        action = f'Order {eoq} units immediately (Express shipping)'
        order_qty = eoq
        color = '#ef4444'
    elif current <= rop:
        alert = 'Reorder Now'
        priority = 'High'
        action = f'Place order for {eoq} units'
        order_qty = eoq
        color = '#f59e0b'
    elif current >= rop * 3 and turnover < 4:
        alert = 'Overstock Warning'
        priority = 'Medium'
        action = f'Consider 10-15% discount to accelerate sales'
        order_qty = 0
        color = '#a855f7'
    else:
        alert = 'Healthy'
        priority = 'Normal'
        days_until_rop = round((current - rop) / max(opt['avg_daily_demand'], 0.5), 0)
        if pd.isna(days_until_rop): days_until_rop = 0
        action = f'Stock OK — ~{int(days_until_rop)} days until reorder'
        order_qty = 0
        color = '#22c55e'

    recommendations.append({
        'product': product,
        'brand': brand,
        'current_stock': current,
        'reorder_point': rop,
        'safety_stock': ss,
        'alert': alert,
        'priority': priority,
        'action': action,
        'order_qty': order_qty,
        'color': color,
    })

# Summary
alert_summary = {}
for r in recommendations:
    a = r['alert']
    alert_summary[a] = alert_summary.get(a, 0) + 1

print(f"   Recommendation breakdown: {alert_summary}")

# ============================================================
# MODEL EVALUATION METRICS
# ============================================================
# Load model results
with open('dashboard/model_results.json', 'r') as f:
    model_data = json.load(f)

evaluation = {
    'best_model': model_data['best_model'],
    'test_metrics': model_data['model_results'][model_data['best_model']]['test'],
    'cv_results': model_data['cv_results'],
    'model_comparison': {
        name: data['test'] for name, data in model_data['model_results'].items()
    },
    'evaluation_criteria': {
        'accuracy': 'R² and MAE on unseen test data',
        'generalization': '5-fold cross-validation stability',
        'robustness': 'Performance across different product categories',
        'business_impact': 'Inventory cost reduction through optimized stocking'
    },
    'service_level': '95%',
    'lead_time_days': LEAD_TIME_DAYS,
    'z_score': SERVICE_LEVEL_Z,
}

# ============================================================
# EXPORT
# ============================================================
output = {
    'product_forecast': product_forecast,
    'product_brand_forecast': product_brand_forecast,
    'inventory_optimization': inventory_optimization,
    'stock_health': stock_health,
    'health_summary': health_summary,
    'recommendations': recommendations,
    'alert_summary': alert_summary,
    'evaluation': evaluation,
    'generated_at': str(pd.Timestamp.now()),
}

with open('dashboard/smartstock_insights.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"  [SUCCESS] SmartStock insights generated!")
print(f"  Output: dashboard/smartstock_insights.json")
print(f"  Forecasts: {len(product_brand_forecast)} product-brand combos")
print(f"  Inventory: {len(inventory_optimization)} items with SS & ROP")
print(f"  Health: {health_summary}")
print(f"  Alerts: {alert_summary}")
print(f"{'='*60}")
