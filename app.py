import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Obesity Risk Clustering App')

# Load the scaled cleaned CSV file
df = pd.read_csv('processed_clean_df.csv')

# Select clustering algorithm
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

# Additional parameters based on the selected model
if cluster_model in ['KMeans', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering']:
    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)

if cluster_model == 'DBSCAN':
    eps = st.slider('Eps (maximum distance between samples)', min_value=0.1, max_value=5.0, value=0.5)
    min_samples = st.slider('Min samples (number of samples in a neighborhood for a point to be considered as a core point)', min_value=1, max_value=50, value=5)
    
if cluster_model == 'Spectral Clustering':
    n_neighbors = st.slider('Number of neighbors for Spectral Clustering', min_value=2, max_value=30, value=10)

# Apply PCA for visualization
apply_pca = st.checkbox('Apply PCA for visualization')

# Perform clustering based on selected model
if cluster_model == 'KMeans':
    clustering = KMeans(n_clusters=n_clusters, n_init=10)
elif cluster_model == 'MeanShift':
    clustering = MeanShift()
elif cluster_model == 'DBSCAN':
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
elif cluster_model == 'Gaussian Mixture':
    clustering = GaussianMixture(n_components=n_clusters)
elif cluster_model == 'Agglomerative Hierarchical Clustering':
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
elif cluster_model == 'Spectral Clustering':
    clustering = SpectralClustering(n_clusters=n_clusters, n_neighbors=n_neighbors)

labels = clustering.fit_predict(df)

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

