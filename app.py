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
df = pd.read_csv('pca_df.csv')

# Standardize data
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df)

# Select clustering algorithm
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

# Input for number of clusters (for applicable methods)
if cluster_model in ['KMeans', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering']:
    n_clusters = st.slider('Number of clusters', min_value=2, max_value=10, value=3)

# Input parameters based on selected clustering model
if cluster_model == 'KMeans':
    init_method = st.selectbox('Initialization method', ['k-means++', 'random'])
    max_iter = st.slider('Maximum iterations', min_value=100, max_value=1000, value=300)

elif cluster_model == 'MeanShift':
    bandwidth = st.slider('Bandwidth', min_value=0.8, max_value=1.5, value=1.18, step=0.1)

elif cluster_model == 'DBSCAN':
    eps = st.slider('Epsilon', min_value=0.1, max_value=0.5, value=0.32, step=0.1)
    min_samples = st.slider('Minimum samples', min_value=3, max_value=12, value=8)

elif cluster_model == 'Gaussian Mixture':
    covariance_type = st.selectbox('Covariance type', ['full', 'tied', 'diag', 'spherical'])
    max_iter = st.slider('Maximum iterations', min_value=100, max_value=1000, value=300)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    affinity = st.selectbox('Affinity', ['euclidean'])
    linkage = st.selectbox('Linkage', ['ward', 'complete', 'average', 'single'])

elif cluster_model == 'Spectral Clustering':
    affinity = st.selectbox('Affinity', ['nearest_neighbors', 'rbf'])
    n_neighbors = st.slider('Number of neighbors', min_value=2, max_value=20, value=10)

# Applying PCA for visualization
apply_pca = st.checkbox('Display PCA Visualization')

# Perform clustering based on selected model
if cluster_model == 'KMeans':
    clustering = KMeans(n_clusters=n_clusters, init=init_method, n_init=10, max_iter=max_iter)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'MeanShift':
    clustering = MeanShift(bandwidth=bandwidth)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'DBSCAN':
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'Gaussian Mixture':
    clustering = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, max_iter=max_iter)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = clustering.fit_predict(df_scaled)

elif cluster_model == 'Spectral Clustering':
    clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors)
    labels = clustering.fit_predict(df_scaled)

# Display PCA visualization
if apply_pca:
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = labels

    st.write('PCA Result:')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    # Handle long legends
    if len(np.unique(labels)) > 5:  # If more than 5 clusters, adjust the legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Standard legend position
    st.pyplot(fig)

# Silhouette Score
if len(np.unique(labels)) > 1:  # Silhouette score needs at least 2 clusters
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
