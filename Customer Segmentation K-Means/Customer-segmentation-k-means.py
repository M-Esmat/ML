# customer_segmentation.py

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv(r'D:\projects\z datasets\Mall-Customer-Segmentation-Data\Mall_Customers.csv')

# Display basic information about the dataset
print("Dataset Head:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nChecking for missing values:")
print(df.isnull().sum())

print("\nDataset Info:")
df.info()

print("\nDataset Statistics:")
print(df.describe())

# Selecting relevant features
X = df.iloc[:, [3, 4]].values

# Plot distributions of the numerical features
plt.figure(figsize=(15, 6))
for i, feature in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)'], start=1):
    plt.subplot(1, 3, i)
    sns.histplot(df[feature], bins=20, kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Compute Within-Cluster Sum of Squares (WCSS) for different cluster numbers
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Graph for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Optimal number of clusters determined from the elbow method: k = 5
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
Y = kmeans.fit_predict(X)

# Visualize the clusters and centroids
plt.figure(figsize=(8, 8))
colors = ['green', 'red', 'yellow', 'violet', 'blue']
for cluster in range(5):
    plt.scatter(X[Y == cluster, 0], X[Y == cluster, 1], s=50, c=colors[cluster], label=f'Cluster {cluster + 1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
