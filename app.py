import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Function to perform preprocessing
def preprocess_data(df):
    # Feature engineering
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Adolescent', 'Adult', 'Elderly'])
    df['FAVC_CAEC'] = df['FAVC'] + "_" + df['CAEC']
    df['Healthy_Score'] = df['FCVC'] + df['FAF'] - df['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
    df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})

    # Drop redundant columns
    df.drop(columns=['Height', 'Weight'], inplace=True)

    return df

# Load and preprocess the dataset
df = pd.read_csv('ObesityDataSet.csv')
df = preprocess_data(df)

# Separate features (since there's no target in clustering)
X = df.select_dtypes(include=[np.number])

# Scale numerical features
numerical_features = X.select_dtypes(include=['number']).columns
scaler = RobustScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

# Streamlit app title
st.title('Obesity Risk Clustering App')

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
    eps = st.slider('Epsilon', min_value=0.1, max_value=0.5, value=0.5, step=0.1)
    min_samples = st.slider('Minimum samples', min_value=3, max_value=12, value=6)

elif cluster_model == 'Gaussian Mixture':
    covariance_type = st.selectbox('Covariance type', ['full', 'tied', 'diag', 'spherical'])
    max_iter = st.slider('Maximum iterations', min_value=100, max_value=1000, value=300)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    affinity = st.selectbox('Affinity', ['euclidean'])
    linkage = st.selectbox('Linkage', ['ward', 'complete', 'average', 'single'])

elif cluster_model == 'Spectral Clustering':
    affinity = st.selectbox('Affinity', ['nearest_neighbors', 'rbf'])
    n_neighbors = st.slider('Number of neighbors', min_value=2, max_value=20, value=10)

# Perform clustering based on selected model
if cluster_model == 'KMeans':
    clustering = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter, n_init=10)
    labels = clustering.fit_predict(X_scaled)

elif cluster_model == 'MeanShift':
    clustering = MeanShift(bandwidth=bandwidth)
    labels = clustering.fit_predict(X_scaled)

elif cluster_model == 'DBSCAN':
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(X_scaled)

elif cluster_model == 'Gaussian Mixture':
    clustering = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, max_iter=max_iter)
    labels = clustering.fit_predict(X_scaled)

elif cluster_model == 'Agglomerative Hierarchical Clustering':
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
    labels = clustering.fit_predict(X_scaled)

elif cluster_model == 'Spectral Clustering':
    clustering = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors)
    labels = clustering.fit_predict(X_scaled)

# Display PCA visualization
apply_pca_viz = st.checkbox('Display PCA Visualization')

if apply_pca_viz:
    pca_viz = PCA(n_components=2)
    pca_data = pca_viz.fit_transform(X_scaled)
    pca_df_viz = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
    pca_df_viz['Cluster'] = labels

    # Plot PCA results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df_viz, palette='viridis')
    plt.title('PCA of Clustering Results')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    st.pyplot()
