import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Obesity Risk Clustering App with PCA')

# Load the dataset (replace with your dataset path)
df = pd.read_csv('ObesityDataSet.csv')

# Feature engineering (as per your previous steps)
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Adolescent', 'Adult', 'Elderly'])
df['FAVC_CAEC'] = df['FAVC'] + "_" + df['CAEC']
df['Healthy_Score'] = df['FCVC'] + df['FAF'] - df['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})

# Drop redundant columns
df.drop(columns=['Height', 'Weight'], inplace=True)

# Separate features (since there's no target in clustering)
X = df.select_dtypes(include=[np.number])

# Step 2: Scale numerical features
numerical_features = X.select_dtypes(include=['number']).columns
scaler = RobustScaler()
X_scaled = X.copy()
X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

# Step 3: Apply PCA
apply_pca = st.checkbox('Apply PCA (95% Variance Retained)')
if apply_pca:
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    # Convert PCA result to DataFrame
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    st.write(f"Explained variance by components: {explained_variance}")
    st.write(f"Cumulative explained variance: {cumulative_variance}")

    # Visualize PCA results
    st.subheader('PCA Components')
    st.write(pca_df.describe())

# Step 4: Elbow Method and Silhouette Scores for KMeans
elbow_silhouette_analysis = st.checkbox('Run Elbow Method and Silhouette Score Analysis')
if elbow_silhouette_analysis:
    st.subheader('Elbow Method and Silhouette Score for KMeans')

    # Perform PCA with 5 components for clustering
    pca = PCA(n_components=5)
    X_pca_5 = pca.fit_transform(X_scaled)
    scaled_pca_df = StandardScaler().fit_transform(X_pca_5)

    # Elbow Method (Inertia)
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_pca_df)
        inertia.append(kmeans.inertia_)

    # Plot Elbow Method
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Elbow Method plot
    axes[0].plot(k_range, inertia, marker='o')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method for Optimal Number of Clusters')

    # Silhouette Score
    silhouette_scores = []
    valid_k = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_pca_df)
        labels = kmeans.labels_

        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(scaled_pca_df, labels)
            silhouette_scores.append(silhouette_avg)
            valid_k.append(k)
        else:
            silhouette_scores.append(np.nan)

    # Filter out NaN values to avoid dimension mismatch
    filtered_valid_k = [k for k, s in zip(k_range, silhouette_scores) if not np.isnan(s)]
    filtered_silhouette_scores = [s for s in silhouette_scores if not np.isnan(s)]

    # Silhouette Score plot
    axes[1].plot(filtered_valid_k, filtered_silhouette_scores, marker='o')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score for Different Number of Clusters')

    plt.tight_layout()
    st.pyplot(fig)

# Perform clustering based on selected model after EDA
st.subheader('Proceed to Clustering Model Selection')
cluster_model = st.selectbox(
    'Select a clustering model',
    ('KMeans', 'MeanShift', 'DBSCAN', 'Gaussian Mixture', 'Agglomerative Hierarchical Clustering', 'Spectral Clustering')
)

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

# Applying PCA for visualization
apply_pca = st.checkbox('Display PCA Visualization')

# Perform clustering based on selected model
if cluster_model == 'KMeans':
    clustering = KMeans(n_clusters=n_clusters, init=init_method, max_iter=max_iter)
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
    if len(np.unique(labels)) > 5:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# Silhouette Score
if len(np.unique(labels)) > 1:
    silhouette_avg = silhouette_score(df_scaled, labels)
    st.write(f'Silhouette Score: {silhouette_avg:.2f}')

# Number of records in each cluster
unique, counts = np.unique(labels, return_counts=True)
cluster_count_df = pd.DataFrame(list(zip(unique, counts)), columns=['Cluster', 'Count'])
st.subheader('Number of records in each cluster:')
st.dataframe(cluster_count_df)

# Mean and Median statistics for each cluster
st.subheader('Cluster Mean Statistics:')
st.dataframe(df_final.groupby(labels).mean())

st.subheader('Cluster Median Statistics:')
st.dataframe(df_final.groupby(labels).median())


