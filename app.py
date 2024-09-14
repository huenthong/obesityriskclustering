import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

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
    # Elbow Method
    n_clusters_range = range(1, 11)
    inertia = []
    for n in n_clusters_range:
        kmeans = KMeans(n_clusters=n, init=init_method, max_iter=max_iter, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot the elbow curve
    st.subheader('Elbow Method for Optimal Number of Clusters (K-Means)')
    plt.figure(figsize=(10, 8))
    plt.plot(n_clusters_range, inertia, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters (K-Means)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    st.pyplot()

    # Silhouette Score
    silhouette_scores = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init=init_method, max_iter=max_iter, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)

    # Plot Silhouette Scores
    st.subheader('Silhouette Score for Optimal Number of Clusters')
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    st.pyplot()

    # Run K-Means with the chosen number of clusters
    optimal_n_clusters = n_clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, init=init_method, max_iter=max_iter, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Display the number of clusters found
    st.write(f"Number of clusters found by K-Means: {optimal_n_clusters}")

    # PCA Visualization
    apply_pca_viz = st.checkbox('Display PCA Visualization')
    if apply_pca_viz:
        pca_viz = PCA(n_components=3)
        pca_data = pca_viz.fit_transform(X_scaled)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'])
        pca_df['KMeans_Cluster'] = labels

        # 2D Plot
        st.write('PCA Result (2D Plot):')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='KMeans_Cluster', palette='viridis', ax=ax)
        if len(np.unique(labels)) > 5:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
        else:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        # 3D Plot
        st.write('PCA Result (3D Plot):')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['KMeans_Cluster'], cmap='viridis', marker='o')
        color_bar = fig.colorbar(scatter, ax=ax, pad=0.1)
        color_bar.set_label('Cluster')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.title('K-Means Clustering Results on Principal Components (3D)')
        st.pyplot(fig)

    # Display cluster sizes
    st.subheader('Cluster Sizes (K-Means)')
    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    st.write(cluster_sizes)

elif cluster_model == 'MeanShift':
    # Estimate Bandwidth
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.2, n_samples=500)
    st.write(f"Estimated Bandwidth: {bandwidth}")

    # Run Mean Shift
    mean_shift = MeanShift(bandwidth=bandwidth)
    labels = mean_shift.fit_predict(X_scaled)

    # Display the number of clusters found
    n_clusters = len(np.unique(labels))
    st.write(f"Number of clusters found by Mean Shift: {n_clusters}")

    # PCA Visualization
    apply_pca_viz = st.checkbox('Display PCA Visualization')
    if apply_pca_viz:
        pca_viz = PCA(n_components=3)
        pca_data = pca_viz.fit_transform(X_scaled)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2', 'PC3'])
        pca_df['MeanShift_Cluster'] = labels

        # 2D Plot
        st.write('PCA Result (2D Plot):')
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=pca_df_viz, x='PC1', y='PC2', hue='Cluster', palette='Set1', ax=ax)
    # Handle long legends
    if len(np.unique(labels)) > 5:  # If more than 5 clusters, adjust the legend
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Standard legend position
    st.pyplot(fig)

# Number of records in each cluster
st.subheader('Number of records in each cluster:')
cluster_counts = pd.Series(labels).value_counts().sort_index()
st.write(cluster_counts)

# Mean statistics for each cluster
st.subheader('Mean statistics for each cluster:')
cluster_mean_stats = pd.DataFrame(X).groupby(labels).mean()
st.write(cluster_mean_stats)

# Median statistics for each cluster
st.subheader('Median statistics for each cluster:')
cluster_median_stats = pd.DataFrame(X).groupby(labels).median()
st.write(cluster_median_stats)

