import pandas as pd

print("--- LOADING CLUSTERED DATA ---")
df = pd.read_csv("final_clustered_customers.csv")

# Select key features to profile
features = [
    'total_financial_value', 
    'debt_to_income', 
    'total_active_channels', 
    'recency_days', 
    'frequency_score',
    'monthly_income_ngn',
    'account_balance_ngn'
]

print("\n--- AVERAGE (MEAN) METRICS PER CLUSTER ---")
cluster_profile = df.groupby('cluster')[features].mean()
print(cluster_profile.to_string())

cluster_profile.to_csv("cluster_summary_profile.csv")
print("\nSuccess! Summary profile also saved to 'cluster_summary_profile.csv'.")