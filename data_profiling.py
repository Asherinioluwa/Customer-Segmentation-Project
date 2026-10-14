import pandas as pd

print("--- LOADING RAW DATASET ---")

df = pd.read_excel('synthetic_customer_segmentation_data.csv.xlsx')

print("\n" + "="*50)
print("1. DATASET SHAPE & OVERVIEW")
print("="*50)
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}")

print("\n" + "="*50)
print("2. MISSING VALUES BREAKDOWN")
print("="*50)
missing_count = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing_count, 'Percentage (%)': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])
if missing_count.sum() == 0:
    print("Zero missing values found across all columns.")

print("\n" + "="*50)
print("3. NUMERICAL COLUMNS (Checking for wild min/max outliers)")
print("="*50)
print(df.describe().T[['min', 'max', 'mean', 'std']])

print("\n" + "="*50)
print("4. CATEGORICAL COLUMNS (Checking for typos & unique values)")
print("="*50)
for col in df.select_dtypes(include=['object', 'bool']).columns:
    print(f"\n--- {col} ({df[col].nunique()} unique values) ---")
    print(df[col].value_counts().head(5)) # Shows top 5 most common entries

print("\n--- PROFILING SCRIPT FINISHED ---")