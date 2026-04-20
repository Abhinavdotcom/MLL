# ============================================================
# CONCEPT: Clustering — KMeans, K-Medoids, DBSCAN
# Includes: Elbow method, Silhouette score, PCA visualisation
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
)
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# --- Load Data ---
data = pd.read_csv('your_dataset.csv')
data = data.dropna(axis=1, how='all')

# --- Clean numeric column ---
data['numeric_col'] = data['numeric_col'].astype(str).str.replace(',', '', regex=True)
data['numeric_col'] = pd.to_numeric(data['numeric_col'], errors='coerce')

# --- Optional: Create label for ARI/NMI evaluation ---
data = data.dropna(subset=['numeric_col']).copy()
try:
    data['Target_Class'] = pd.qcut(data['numeric_col'], q=3, labels=['Low', 'Medium', 'High'])
except ValueError:
    data['Target_Class'] = pd.cut(data['numeric_col'], bins=3, labels=['Low', 'Medium', 'High'])
y = data['Target_Class']
y_encoded = LabelEncoder().fit_transform(y.astype(str))

# --- Feature Matrix ---
remove_cols = ['numeric_col', 'Target_Class']
X = data.drop(columns=remove_cols, errors='ignore')

for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object', 'string']).columns:
    mode_val = X[col].mode(dropna=True)
    X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')

for col in X.select_dtypes(include=['object', 'string']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_scaled = StandardScaler().fit_transform(X)

X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# ---- KMeans: Find Best K (Elbow + Silhouette) ----
k_values = list(range(2, min(9, X_train.shape[0])))
wcss, sil_scores = [], []
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train)
    wcss.append(km.inertia_)
    try:
        sil_scores.append(silhouette_score(X_train, km.labels_))
    except Exception:
        sil_scores.append(np.nan)

best_k = k_values[int(np.nanargmax(np.array(sil_scores, dtype=float)))]
print("Best K:", best_k)

# ---- KMeans Fit ----
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# ---- K-Medoids (custom implementation) ----
D = pairwise_distances(X_scaled, metric='euclidean')
rng = np.random.default_rng(42)

def k_medoids_labels(X_dist, k, max_iter=50):
    n = X_dist.shape[0]
    medoid_indices = rng.choice(n, size=k, replace=False)
    for _ in range(max_iter):
        labels = np.argmin(X_dist[:, medoid_indices], axis=1)
        new_medoid_indices = medoid_indices.copy()
        for i in range(k):
            cluster_idx = np.where(labels == i)[0]
            if len(cluster_idx) == 0:
                new_medoid_indices[i] = rng.choice(n)
                continue
            sub_dist = X_dist[np.ix_(cluster_idx, cluster_idx)]
            new_medoid_indices[i] = cluster_idx[np.argmin(sub_dist.sum(axis=1))]
        if np.array_equal(new_medoid_indices, medoid_indices):
            break
        medoid_indices = new_medoid_indices
    return np.argmin(X_dist[:, medoid_indices], axis=1)

labels_kmedoids = k_medoids_labels(D, best_k)

# ---- DBSCAN: Auto eps using KNN distance ----
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_train)
distances, _ = nn.kneighbors(X_train)
knn_dist = np.sort(distances[:, -1])
eps = float(np.percentile(knn_dist, 90)) if len(knn_dist) > 0 else 0.5
eps = max(eps, 0.1)

labels_dbscan = DBSCAN(eps=eps, min_samples=5).fit_predict(X_scaled)

# ---- Clustering Evaluation Report ----
def clustering_report(name, X_use, labels, y_true=None):
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels != -1]) if -1 in unique_labels else len(unique_labels)
    silhouette = db_index = ch_index = ari = nmi = np.nan
    if n_clusters >= 2:
        try:
            silhouette = silhouette_score(X_use, labels)
            db_index   = davies_bouldin_score(X_use, labels)
            ch_index   = calinski_harabasz_score(X_use, labels)
        except Exception:
            pass
    if y_true is not None:
        try:
            ari = adjusted_rand_score(y_true, labels)
            nmi = normalized_mutual_info_score(y_true, labels)
        except Exception:
            pass
    print(f"\n===== {name} =====")
    print(f"Clusters: {n_clusters}  |  Silhouette: {round(float(silhouette),4)}  |  DB Index: {round(float(db_index),4)}  |  CH Index: {round(float(ch_index),4)}")
    if y_true is not None:
        print(f"ARI: {round(float(ari),4)}  |  NMI: {round(float(nmi),4)}")
    return {'Model': name, 'Clusters': n_clusters, 'Silhouette': silhouette,
            'Davies_Bouldin': db_index, 'Calinski_Harabasz': ch_index, 'ARI': ari, 'NMI': nmi}

r1 = clustering_report('K-MEANS',    X_scaled, labels_kmeans,   y_encoded)
r2 = clustering_report('K-MEDOIDS',  X_scaled, labels_kmedoids, y_encoded)
r3 = clustering_report('DBSCAN',     X_scaled, labels_dbscan,   y_encoded)

print("\n===== COMPARISON TABLE =====")
print(pd.DataFrame([r1, r2, r3]))

# ---- PCA 2D Visualisation ----
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)

for labels, title in [(labels_kmeans, 'K-Means'), (labels_kmedoids, 'K-Medoids'), (labels_dbscan, 'DBSCAN')]:
    plt.figure(figsize=(7, 5))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', s=12)
    plt.title(f'{title} Clusters (PCA 2D)')
    plt.tight_layout()
    plt.show()
