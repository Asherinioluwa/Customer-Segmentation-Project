import pandas as pd

file_path = "synthetic_customer_segmentation_data.csv.xlsx"
df = pd.read_excel(file_path)

# 1. Fill activity/transaction gaps with 0
zero_fill_cols = ['transactions_30d', 'transaction_value_30d_ngn', 'days_since_last_transaction']
for col in zero_fill_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 2. Fill financial/score gaps with the median
median_fill_cols = ['monthly_income_ngn', 'credit_score']
for col in median_fill_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

df.to_csv('cleaned_customer_data.csv', index=False)
print("Data cleaning complete! Saved as 'cleaned_customer_data.csv'.")