# Real-Time Deployment & Continuous Data Strategy
## SmartStock Predictive Inventory Optimization

To take this project from a local Jupyter Notebook/Script pipeline into a production application capable of handling continuous daily inputs, we need to transition from a static CSV file to an automated Cloud Architecture. 

Here is a comprehensive strategy to deploy this project **100% freely** and automate the daily data ingestion and prediction generation.

---

## 1. Architecture Overview (The "Free" Stack)

To keep costs at $0, we will split the application into three layers:

1. **Database Layer (Data Storage):** [Supabase](https://supabase.com/) (Free Tier)
   * Instead of reading from `FootWare_Wholesale_Sales_Dataset.csv`, new daily transactions will be inserted into a free PostgreSQL database.
2. **Compute & Automation Layer (The Brains):** [GitHub Actions](https://github.com/features/actions) (Free Tier)
   * We will use GitHub Actions' CRON jobs to automatically wake up every night, run the Python scripts, generate the JSON insights, and push the results.
3. **Frontend / Dashboard Layer (The UI):** [Vercel](https://vercel.com/) or [Netlify](https://netlify.com/) (Free Tier)
   * The HTML/JS dashboard will be hosted on a fast, global CDN. It will simply read the generated JSON data.

---

## 2. Diagram: The Daily Automation Pipeline

Here is how data legally flows through your system every single day:

```mermaid
sequenceDiagram
    participant Store as Store POS / ERP
    participant DB as Supabase (Database)
    participant Action as GitHub Actions (CRON)
    participant Model as XGBoost Model (.pkl)
    participant Frontend as Vercel (Dashboard)

    Note over Store, DB: 1. Continuous Data Collection
    Store->>DB: Send real-time sales throughout the day
    
    Note over Action, Frontend: 2. Nightly Batch Processing (e.g., 1:00 AM)
    Action->>DB: Fetch last 30 days of Sales Data
    Action->>Model: Run `generate_insights.py`
    Model-->>Action: Output `smartstock_insights.json`
    Action->>Frontend: Push new Insights (Triggers Deployment)
    
    Note over Frontend: 3. Business Ready
    Frontend->>Frontend: Dashboard Updates automatically by morning
```

---

## 3. Step-by-Step Implementation Guide

### Phase 1: Migrate Data to a Database
Currently, your `run_modeling.py` and `generate_insights.py` read via `pd.read_csv()`.
1. Create a free account on **Supabase**.
2. Create a table named `sales_transactions` matching the columns of your dataset.
3. Update your Python code to query the database using `psycopg2` or `SQLAlchemy` instead of `pd.read_csv()`.
   ```python
   # Example of new data connection
   from sqlalchemy import create_engine
   engine = create_engine('postgresql://user:pass@supabase-url.com:5432/postgres')
   df_raw = pd.read_sql("SELECT * FROM sales_transactions", engine)
   ```

### Phase 2: Setup Point-of-Sale (POS) integrations
* Whenever a sale happens in the real world, the retail POS (Point of Sale) system needs to make a simple HTTP POST request to your Supabase Database to insert the new record. 
* This means your data is continuously updating in "real-time."

### Phase 3: Setup GitHub Actions for Daily Inference
Since your pipeline builds `Lag_7` and `Rolling_30_Mean` features, pure "real-time" inference on every single click isn't needed nor efficient. Inventory forecasting is best done via **nightly batch inference**.

1. In your repository, create a file at `.github/workflows/daily_forecast.yml`.
2. Configure it to run on a schedule:
   ```yaml
   name: Daily SmartStock Forecast
   on:
     schedule:
       - cron: '0 1 * * *' # Runs at 1:00 AM every single day
   jobs:
     build:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.10'
         - name: Install dependencies
           run: pip install pandas numpy scikit-learn xgboost
         - name: Generate New Insights
           env:
             DATABASE_URL: ${{ secrets.DATABASE_URL }}
           run: python generate_insights.py
         - name: Commit and Push new JSON
           run: |
             git config --global user.name "GitHub Action Bot"
             git config --global user.email "action@github.com"
             git add dashboard/smartstock_insights.json
             git commit -m "Automated daily insight generation"
             git push
   ```
* **What happens here**: At 1 AM, the free GitHub server pulls the newest data from Supabase, loads your `best_model.pkl`, generates tomorrow's prediction, and saves the new JSON directly to your code base.

### Phase 4: Deploy the Dashboard
1. Go to **Vercel.com** and connect your GitHub repository.
2. Tell Vercel that your root directory is `dashboard/`.
3. Every time GitHub Actions pushes the new `smartstock_insights.json` file in Phase 3, Vercel will automatically redeploy your dashboard.
4. Your business stakeholders can visit `your-custom-name.vercel.app` at 8:00 AM every morning and see the exact inventory alerts and predictions based on yesterday's continuous data.

---

## 4. Handling Model Drift (Automated Retraining)

Over time, consumer trends change and your XGBoost model will become less accurate. You don't need to retrain the model every single day, but rather every month.

You can set up a second GitHub Action (`.github/workflows/monthly_training.yml`) that runs on the 1st of every month:
1. Fetches all historical data + the new month of data.
2. Runs `python run_modeling.py`.
3. Overwrites `best_model.pkl`.
4. Tests the R² Score. If the score is adequate, it commits the new model to your repository so it's ready for the daily inference. 

By following this architecture, you get an Enterprise-grade automated pipeline without paying for expensive cloud servers or endpoints.
