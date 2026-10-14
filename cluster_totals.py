import pandas as pd

print("--- LOADING CLUSTERED DATA ---")
df = pd.read_csv("final_clustered_customers.csv")

# First, show the exact size of each cluster
print("\n--- CUSTOMER COUNT PER CLUSTER ---")
print(df['cluster'].value_counts().sort_index())

# Select financial and volume metrics to sum up
features_to_sum = [
    'total_financial_value', 
    'account_balance_ngn', 
    'monthly_income_ngn',
    'frequency_score'
]

print("\n--- TOTAL (SUM) METRICS PER CLUSTER ---")
cluster_totals = df.groupby('cluster')[features_to_sum].sum()

pd.options.display.float_format = '{:,.2f}'.format
print(cluster_totals)

# Save to a new CSV file
cluster_totals.to_csv("cluster_total_summary.csv")
print("\nSuccess! Totals summary saved to 'cluster_total_summary.csv'.")