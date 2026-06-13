
"""
Task 02: K-Means Clustering on Mall Customer Dataset
Groups retail customers based on purchase history
(Annual Income & Spending Score).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
# Download from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print(df.info())

# ---------------------------------------------------------
# 2. Select features for clustering
# ---------------------------------------------------------
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 3. Elbow Method to find optimal K
# ---------------------------------------------------------
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.xticks(K_range)
plt.grid(True)
plt.savefig('elbow_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 4. Fit K-Means with chosen K (5 is typical for this dataset)
# ---------------------------------------------------------
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ---------------------------------------------------------
# 5. Visualize clusters
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']

for cluster in range(optimal_k):
    cluster_points = X[df['Cluster'] == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                s=60, c=colors[cluster % len(colors)],
                label=f'Cluster {cluster}')

# Plot centroids (inverse-transform back to original scale)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1],
            s=250, c='black', marker='X', label='Centroids')

plt.title('Customer Segments based on Income & Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig('cluster_plot.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------------------------------------------------------
# 6. Cluster summary
# ---------------------------------------------------------
summary = df.groupby('Cluster')[['Annual Income (k$)', 'Spending Score (1-100)']].mean()
summary['Count'] = df['Cluster'].value_counts()
print("\nCluster Summary:")
print(summary)

df.to_csv('clustered_customers.csv', index=False)
print("\nSaved: elbow_plot.png, cluster_plot.png, clustered_customers.csv")