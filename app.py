import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import silhouette_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app title
st.title('Obesity Risk Clustering App')

# Load the dataset (replace with your dataset path)
df = pd.read_csv('ObesityDataSet.csv')

# Step 1: Create a BMI feature
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

# Step 2: Categorize Age into bins
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100], labels=['Child', 'Adolescent', 'Adult', 'Elderly'])

# Step 3: Create an interaction feature between FAVC and CAEC (Consumption habits)
df['FAVC_CAEC'] = df['FAVC'] + "_" + df['CAEC']

# Step 4: Create a Healthy Habits Score (combining healthy lifestyle indicators)
df['Healthy_Score'] = df['FCVC'] + df['FAF'] - df['FAVC'].apply(lambda x: 1 if x == 'yes' else 0)

# Step 5: Convert binary categorical features to 1/0 (like family history and FAVC)
df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})

# Step 6: Drop redundant columns after creating engineered features
df.drop(columns=['Height', 'Weight'], inplace=True)  # BMI replaces these

# List of numerical features
numerical_features = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'Healthy_Score']

# Boxplot visualization for numerical features (before outlier removal)
st.subheader('Boxplot of Numerical Features (Before Outlier Removal)')
fig, axes = plt.subplots(2, 4, figsize=(15, 10))
axes = axes.flatten()
for i, feature in enumerate(numerical_features):
    sns.boxplot(x=df[feature], ax=axes[i])
    axes[i].set_title(f'Boxplot of {feature}')
st.pyplot(fig)

# Remove outliers using Z-score (values more than 3 standard deviations from the mean)
numerical_features_excl_age = [f for f in numerical_features if f != 'Age']
for feature in numerical_features_excl_age:
    df = df[(np.abs(stats.zscore(df[feature])) < 3)]

# Check and remove duplicate rows
df = df.drop_duplicates()

# Handling skewness (log transformation for skewed features)
for feature in numerical_features_excl_age:
    if df[feature].skew() > 1:
        df[feature] = np.log1p(df[feature])

# One-Hot Encode nominal categorical features
nominal_features = ['Gender', 'FAVC_CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'CAEC']
onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_nominal = onehot_encoder.fit_transform(df[nominal_features])
encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=onehot_encoder.get_feature_names_out(nominal_features))

# Label Encode ordinal features
ordinal_features = ['NObeyesdad', 'Age_Group']
label_encoder = LabelEncoder()
df['NObeyesdad'] = label_encoder.fit_transform(df['NObeyesdad'])
df['Age_Group'] = label_encoder.fit_transform(df['Age_Group'])

# Drop original nominal columns and concatenate encoded ones
df_numerical_only = df.drop(columns=nominal_features).reset_index(drop=True)
df_encoded = pd.concat([df_numerical_only, encoded_nominal_df], axis=1)

# Drop weakly correlated and highly correlated features
weak_corr_features = ['FAVC', 'FCVC', 'NCP', 'CH2O', 'TUE', 'Gender_Male', 'SMOKE_yes', 'SCC_yes']
df_clean = df_encoded.drop(columns=weak_corr_features)

# Final feature selection (further columns based on correlation)
further_columns_to_drop = ['FAVC_CAEC_yes_Sometimes', 'CAEC_Sometimes', 'Healthy_Score', 'CAEC_Frequently']
df_final = df_clean.drop(columns=further_columns_to_drop)

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_final)

# Display the shape of the dataset after preprocessing
st.write(f"Shape of the preprocessed dataset: {df_final.shape}")

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


