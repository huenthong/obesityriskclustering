import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Obesity Risk Clustering App')

# Load the scaled cleaned CSV file
df = pd.read_csv('pca_df.csv')

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Select clustering algorithm
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

# Input parameters and fine-tuning based on selected clustering model
if cluster_model == 'KMeans':
    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)
    init_method = st.selectbox('Initialization method', ['k-means++', 'random'])
    max_iter = st.slider('Maximum iterations', min_value=100, max_value=1000, value=300)
    
    # Fine-tuning KMeans
    param_grid = {"n_clusters": [n_clusters]}
    kmeans_model = GridSearchCV(KMeans(init=init_method, max_iter=max_iter, n_init=10), param_grid, cv=3)
    kmeans_model.fit(df_scaled)
    labels = kmeans_model.best_estimator_.predict(df_scaled)
    st.write(f'Best KMeans Parameters: {kmeans_model.best_params_}')

elif cluster_model == 'MeanShift':
    # Random sampling for MeanShift
    bandwidth_values = np.linspace(0.8, 1.5, num=10)
    n_iter = 5
    best_score = -1
    best_bandwidth = None
    for i in range(n_iter):
        bandwidth = random.choice(bandwidth_values)
        meanshift = MeanShift(bandwidth=bandwidth)
        labels = meanshift.fit_predict(df_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(df_scaled, labels)
            if score > best_score:
                best_score = score
                best_bandwidth = bandwidth
    clustering = MeanShift(bandwidth=best_bandwidth)
    labels = clustering.fit_predict(df_scaled)
    st.write(f'Best Bandwidth: {best_bandwidth}')

elif cluster_model == 'DBSCAN':
    # Random sampling for DBSCAN
    eps_values = np.linspace(0.1, 0.5, num=10)
    min_samples_values = np.arange(3, 12)
    n_iter = 5
    best_score = -1
    best_eps = None
    best_min_samples = None
    for i in range(n_iter):
        eps = random.choice(eps_values)
        min_samples = random.choice(min_samples_values)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(df_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(df_scaled, labels)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_min_samples = min_samples
    clustering = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    labels = clustering.fit_predict(df_scaled)
    st.write(f'Best eps: {best_eps}, Best min_samples: {best_min_samples}')

elif cluster_model == 'Gaussian Mixture':
    # Fine-tuning Gaussian Mixture
    param_grid = {"n_components": [n_clusters], "covariance_type": ['full', 'tied', 'diag', 'spherical']}
    gmm_model = GridSearchCV(GaussianMixture(max_iter=300), param_grid, cv=3)
    gmm_model.fit(df_scaled)
    labels = gmm_model.best_estimator_.predict(df_scaled)
    st.write(f'Best Gaussian Mixture Parameters: {gmm_model.best_params_}')

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    # Random sampling for Agglomerative Clustering
    linkage_values = ['ward', 'complete', 'average', 'single']
    n_clusters_values = np.arange(2, 10)
    n_iter = 5
    best_score = -1
    best_n_clusters = None
    best_linkage = None
    for i in range(n_iter):
        n_clusters = random.choice(n_clusters_values)
        linkage = random.choice(linkage_values)
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity='euclidean')
        labels = agglomerative.fit_predict(df_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(df_scaled, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_linkage = linkage
    clustering = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_linkage, affinity='euclidean')
    labels = clustering.fit_predict(df_scaled)
    st.write(f'Best n_clusters: {best_n_clusters}, Best linkage: {best_linkage}')

elif cluster_model == 'Spectral Clustering':
    # Random sampling for Spectral Clustering
    affinity_values = ['nearest_neighbors', 'rbf']
    n_neighbors_values = np.arange(2, 20)
    n_iter = 5
    best_score = -1
    best_affinity = None
    best_n_neighbors = None
    for i in range(n_iter):
        affinity = random.choice(affinity_values)
        n_neighbors = random.choice(n_neighbors_values)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors)
        labels = spectral.fit_predict(df_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(df_scaled, labels)
            if score > best_score:
                best_score = score
                best_affinity = affinity
                best_n_neighbors = n_neighbors
    clustering = SpectralClustering(n_clusters=n_clusters, affinity=best_affinity, n_neighbors=best_n_neighbors)
    labels = clustering.fit_predict(df_scaled)
    st.write(f'Best affinity: {best_affinity}, Best n_neighbors: {best_n_neighbors}')

# Applying PCA for visualization
apply_pca = st.checkbox('Display PCA Visualization')

if apply_pca:
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    st.write('PCA Result:')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    if len(np.unique(labels)) > 5:  # Adjust legend if too many clusters
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Standard legend position
    st.pyplot(fig)

# Silhouette Score
if len(np.unique(labels)) > 1:
    silhouette_avg = silhouette_score(df_scaled, labels)
    st.write(f'Silhouette Score: {silhouette_avg:.2f}')

# Number of records in each cluster
st.subheader('Number of records in each cluster:')
cluster_counts = pd.Series(labels).value_counts().sort_index()
st.write(cluster_counts)

# Mean statistics for each cluster
st.subheader('Mean statistics for each cluster:')
cluster_mean_stats = pd.DataFrame(df).groupby(labels).mean()
st.write(cluster_mean_stats)

# Median statistics for each cluster
st.subheader('Median statistics for each cluster:')
cluster_median_stats = pd.DataFrame(df).groupby(labels).median()
st.write(cluster_median_stats)
