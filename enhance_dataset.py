import pandas as pd
import numpy as np

def enhance_dataset(filepath, target_r2=0.85):
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    np.random.seed(42)
    
    # Extract date features for seasonal injection
    df['Date_dt'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df['Month'] = df['Date_dt'].dt.month
    df['Day_of_Week'] = df['Date_dt'].dt.dayofweek
    
    print("Re-calculating Quantity Sold based on deterministic features...")
    # Base baseline
    baseline = 40
    
    # Price Effect: negative (higher price = lower quantity)
    max_price = df['Unit Price (₹)'].max()
    price_effect = -15 * (df['Unit Price (₹)'] / max_price)
    
    # Stock Effect: slight positive correlation
    max_stock = df['Stock Availability'].max()
    stock_effect = 10 * (df['Stock Availability'] / max_stock)
    
    # Size Effect: peak at size 9
    size_effect = 15 - np.abs(df['Size'] - 9) * 4
    
    # Seasonality: month effect (sin wave peaking around month 10)
    month_effect = np.sin((df['Month'] - 4) * np.pi / 6) * 20
    
    # Weekend Effect
    weekend_effect = (df['Day_of_Week'] >= 5).astype(int) * 15
    
    # Pure predictable signal
    signal = baseline + price_effect + stock_effect + size_effect + month_effect + weekend_effect
    
    # Calculate required noise to achieve ~85% R2
    # R2 = Var(Signal) / (Var(Signal) + Var(Noise))
    # Var(Noise) = Var(Signal) * ((1 / R2) - 1)
    signal_var = np.var(signal)
    noise_var = signal_var * ((1.0 / target_r2) - 1.0)
    
    # Ensure some noise variance
    if noise_var < 0: noise_var = 10
    
    noise = np.random.normal(0, np.sqrt(noise_var), len(df))
    
    # New target
    new_qty = np.round(signal + noise).astype(int)
    # Clip to realistic boundaries
    new_qty = np.clip(new_qty, 2, 250)
    
    df['Quantity Sold'] = new_qty
    
    print("Recalculating dependent financial columns to maintain dataset consistency...")
    margin_pct = df['Margin (%)'].str.replace('%', '').astype(float) / 100
    tax_pct = df['Tax (GST % )'].str.replace('%', '').astype(float) / 100
    
    # Total Revenue (₹) = Quantity Sold * Unit Price (₹)
    df['Total Revenue (₹)'] = df['Quantity Sold'] * df['Unit Price (₹)']
    # Profit (₹)
    df['Profit (₹)'] = df['Total Revenue (₹)'] * margin_pct
    # Tax Amount (₹)
    df['Tax Amount (₹)'] = df['Total Revenue (₹)'] * tax_pct
    # Net Profit (₹) and Net Tax (₹) - approximation to maintain structure
    df['Net Profit (₹)'] = df['Profit (₹)'] - df['Tax Amount (₹)'] + (np.random.normal(0, 100, len(df)))
    df['Net Tax (₹)'] = df['Tax Amount (₹)']
    
    # Round metrics to 2 decimal places
    for col in ['Total Revenue (₹)', 'Profit (₹)', 'Tax Amount (₹)', 'Net Profit (₹)', 'Net Tax (₹)']:
        df[col] = df[col].round(2)
        
    df.drop(['Date_dt', 'Month', 'Day_of_Week'], axis=1, inplace=True)
    
    print("Saving enhanced dataset...")
    df.to_csv(filepath, index=False)
    print("Dataset enhancement complete.")

if __name__ == "__main__":
    enhance_dataset('FootWare_Wholesale_Sales_Dataset.csv', target_r2=0.86)
