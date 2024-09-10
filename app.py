import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Setting the environment variable for avoiding the KMeans memory leak warning
#os.environ['OMP_NUM_THREADS'] = '9'

# Streamlit app title
st.title('Obesity Risk Clustering App')

# Load the scaled cleaned CSV file

df = pd.read_csv('encoded_clean_df.csv')


# Select clustering algorithm
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

# Input for number of clusters (for applicable methods)
if cluster_model in ['KMeans', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering']:
    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)

# Applying PCA for visualization
apply_pca = st.checkbox('Apply PCA for visualization')

# Perform clustering based on selected model
def perform_clustering(df, model, n_clusters=None):
    if model == 'KMeans':
        clustering = KMeans(n_clusters=n_clusters, n_init=10)
    elif model == 'MeanShift':
        clustering = MeanShift()
    elif model == 'DBSCAN':
        clustering = DBSCAN()
    elif model == 'Gaussian Mixture':
        clustering = GaussianMixture(n_components=n_clusters)
    elif model == 'Agglomerative Hierarchical Clustering':
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    elif model == 'Spectral Clustering':
        clustering = SpectralClustering(n_clusters=n_clusters)

    if model == 'Gaussian Mixture':
        labels = clustering.fit_predict(df)
    else:
        labels = clustering.fit_predict(df)

    return labels

# Perform clustering
if cluster_model != 'MeanShift':  # MeanShift does not require n_clusters
    labels = perform_clustering(df, cluster_model, n_clusters=n_clusters)
else:
    labels = perform_clustering(df, cluster_model)

# Display PCA visualization
if apply_pca:
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    st.write('PCA Result:')
    fig, ax = plt.subplots()
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    st.pyplot(fig)

# Silhouette Score
if len(np.unique(labels)) > 1:  # Silhouette score needs at least 2 clusters
    silhouette_avg = silhouette_score(df, labels)
    st.write(f'Silhouette Score: {silhouette_avg:.2f}')

# Number of records in each cluster
st.write('Number of records in each cluster:')
cluster_counts = pd.Series(labels).value_counts().sort_index()
st.write(cluster_counts)

# Mean statistics for each cluster
st.write('Mean statistics for each cluster:')
cluster_mean_stats = pd.DataFrame(df).groupby(labels).mean()
st.write(cluster_mean_stats)

# Median statistics for each cluster
st.write('Median statistics for each cluster:')
cluster_median_stats = pd.DataFrame(df).groupby(labels).median()
st.write(cluster_median_stats)

# Median statistics for each cluster
st.write('Median statistics for each cluster:')
cluster_median_stats = pd.DataFrame(df).groupby(labels).median()
st.write(cluster_median_stats)
