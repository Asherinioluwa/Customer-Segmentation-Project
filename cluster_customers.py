import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("--- 1. LOADING RAW DATA ---")
df = pd.read_csv("cleaned_customer_data.csv")

print("--- 2. CREATING FEATURES ---")
channels = ["card_active", "mobile_app_active", "web_active", "ussd_active"]
df["total_active_channels"] = df[channels].sum(axis=1)

df["debt_to_income"] = np.where(
    df["monthly_income_ngn"] > 0,
    df["loan_outstanding_ngn"] / (df["monthly_income_ngn"] * 12),
    0,
)

df["total_financial_value"] = (
    df["account_balance_ngn"] + df["transaction_value_30d_ngn"]
)
df["recency_days"] = df["days_since_last_transaction"]
df["frequency_score"] = df["transactions_30d"]

print("--- 3. PREPARING FEATURES & SCALING ---")
features = [
    "total_financial_value",
    "debt_to_income",
    "total_active_channels",
    "recency_days",
    "frequency_score",
]

X = df[features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("--- 4. CALCULATING ELBOW METHOD (JUSTIFYING K=4) ---")
inertia = []
K_range = range(1, 11)

for k in K_range:
  kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
  kmeans_temp.fit(X_scaled)
  inertia.append(kmeans_temp.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(
    K_range,
    inertia,
    marker="o",
    linestyle="-",
    color="#1f77b4",
    linewidth=2,
    markersize=6,
)
plt.axvline(
    x=4,
    color="#d62728",
    linestyle="--",
    linewidth=2,
    label="Selected Optimal K = 4",
)
plt.title("Elbow Method for Optimal K Selection", fontsize=11, fontweight="bold")
plt.xlabel("Number of Clusters (K)", fontsize=9)
plt.ylabel("Inertia", fontsize=9)
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

print("--- 5. RUNNING FINAL CLUSTERING (K=4) ---")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

from matplotlib.colors import BoundaryNorm

print("--- 6. GENERATING PCA SCATTER PLOT PROJECTION ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# Set up discrete colormap and boundaries for 4 clusters (0, 1, 2, 3)
cmap = plt.get_cmap("viridis", 4)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

plt.figure(figsize=(9, 6))
scatter = plt.scatter(
    df["PC1"],
    df["PC2"],
    c=df["cluster"],
    cmap=cmap,
    norm=norm,
    alpha=0.6,
    s=15,
)
plt.title(
    "Customer Segmentation Projection (PCA 2D Scatter Plot)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel(
    f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%"
    " Variance)",
    fontsize=10,
)
plt.ylabel(
    f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%"
    " Variance)",
    fontsize=10,
)

# Clean discrete colorbar with labels centered on the 4 blocks
cbar = plt.colorbar(scatter, ticks=[0, 1, 2, 3])
cbar.set_label("Cluster Groups")

plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

print("--- 7. SAVING RESULTS ---")
df.to_csv("final_clustered_customers.csv", index=False)
print("Success! Final model complete and saved to 'final_clustered_customers.csv'.")

print("\nCustomer count per cluster:")
print(df["cluster"].value_counts())