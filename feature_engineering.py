import pandas as pd
import numpy as np

print("--- 1. LOADING DATASET ---")
df = pd.read_csv("cleaned_customer_data.csv")

print("--- 2. ENGINEERING FEATURES ---")

# 1. Digital Channels
channels = ['card_active', 'mobile_app_active', 'web_active', 'ussd_active']
df['total_active_channels'] = df[channels].sum(axis=1)

# 2. Debt-to-Income
df['debt_to_income'] = np.where(
    df['monthly_income_ngn'] > 0,
    df['loan_outstanding_ngn'] / (df['monthly_income_ngn'] * 12),
    0
)

# 3. Total Financial Value
df['total_financial_value'] = df['account_balance_ngn'] + df['transaction_value_30d_ngn']

# 4. Recency (Days since last transaction) 
date_col = 'last_transaction_date' if 'last_transaction_date' in df.columns else df.columns[df.columns.str.contains('date', case=False)].tolist()[0]
df[date_col] = pd.to_datetime(df[date_col])
snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
df['recency_days'] = (snapshot_date - df[date_col]).dt.days

# 5. Frequency (Transaction count) 
freq_col = 'transaction_count_30d' if 'transaction_count_30d' in df.columns else df.columns[df.columns.str.contains('count|freq', case=False)].tolist()[0]
df['frequency_score'] = df[freq_col]


df.to_csv('engineered_customer_data.csv', index=False)
print("Success! Engineered dataset saved with all 5 features.")
