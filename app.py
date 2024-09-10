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
from sklearn.exceptions import NotFittedError

# Streamlit app title
st.title('Obesity Risk Clustering App')

# Load the scaled cleaned CSV file
df = pd.read_csv('processed_clean_df.csv')

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Select clustering algorithm
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

# Input parameters based on the selected model
params = {}

if cluster_model in ['KMeans', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering']:
    params['n_clusters'] = st.slider('Number of clusters', min_value=2, max_value=10, value=3)

if cluster_model in ['DBSCAN']:
    params['eps'] = st.slider('Epsilon (eps)', min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    params['min_samples'] = st.slider('Minimum samples (min_samples)', min_value=1, max_value=100, value=5)

if cluster_model == 'Spectral Clustering':
    params['affinity'] = st.selectbox('Affinity', ('nearest_neighbors', 'rbf', 'precomputed'))

# Applying PCA for visualization
apply_pca = st.checkbox('Apply PCA for visualization')

# Perform clustering based on selected model
try:
    if cluster_model == 'KMeans':
        clustering = KMeans(n_clusters=params['n_clusters'], n_init=10)
        labels = clustering.fit_predict(df_scaled)

    elif cluster_model == 'MeanShift':
        clustering = MeanShift()
        labels = clustering.fit_predict(df_scaled)

    elif cluster_model == 'DBSCAN':
        clustering = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
        labels = clustering.fit_predict(df_scaled)

    elif cluster_model == 'Gaussian Mixture':
        clustering = GaussianMixture(n_components=params['n_clusters'])
        labels = clustering.fit_predict(df_scaled)

    elif cluster_model == 'Agglomerative Hierarchical Clustering':
        clustering = AgglomerativeClustering(n_clusters=params['n_clusters'])
        labels = clustering.fit_predict(df_scaled)

    elif cluster_model == 'Spectral Clustering':
        clustering = SpectralClustering(n_clusters=params['n_clusters'], affinity=params['affinity'])
        labels = clustering.fit_predict(df_scaled)

    # Display PCA visualization
    if apply_pca:
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df_scaled)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = labels

        st.write('PCA Result:')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax, legend='full')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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

except NotFittedError as e:
    st.write(f'Error: {str(e)}')
except ValueError as e:
    st.write(f'Error: {str(e)}')



