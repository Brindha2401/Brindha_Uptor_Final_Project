import numpy as np
import pandas as pd
import numpy as nps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv('brindha_107_final_project.csv')
print("\nColumns:", df.columns.tolist())

# Data Preprocessing

## 1. Handling Missing Values
# Check for missing values
print("Missing values before imputation:\n", df.isnull().sum())

# Impute missing numerical values with mean
df['use_count'].fillna(df['use_count'].mean())
df['reward_points_used'].fillna(df['reward_points_used'].mean())

# Impute missing categorical values with mode
df['gender'].fillna(df['gender'].mode()[0])
df['shared_account'].fillna(df['shared_account'].mode()[0])

print("\nMissing values after imputation:\n", df.isnull().sum())

## 2. Encoding Categorical Variables
# Initialize LabelEncoder
le = LabelEncoder()

# Encode 'gender' and 'shared_account'
df['gender'] = le.fit_transform(df['gender'])
df['shared_account'] = le.fit_transform(df['shared_account'])

## 3. Feature Scaling
# Initialize StandardScaler
scaler = StandardScaler()

# Scale numerical features
df[['use_count', 'reward_points_used']] = scaler.fit_transform(df[['use_count', 'reward_points_used']])

## 4. Outlier Detection and Removal
# Using Z-score to identify outliers

z_scores = np.abs(zscore(df[['use_count', 'reward_points_used']]))
df = df[(z_scores < 3).all(axis=1)]

# Exploratory Data Analysis (EDA)

## 1. Distribution of Numerical Features
plt.figure(figsize=(12, 6))
sns.histplot(df['use_count'], kde=True, color='blue', bins=30)
plt.title('use_count Distribution')
plt.xlabel('use_count')
plt.ylabel('Frequency')
plt.savefig("use_count_distribution.png")

## 2. Count Plot of Categorical Features
plt.figure(figsize=(12, 6))
sns.countplot(x='gender', data=df)
plt.title('gender Distribution')
plt.xlabel('gender')
plt.ylabel('Count')
plt.savefig("gender_distribution.png")

## 3. Correlation Heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig("correlation_heatmap.png")

## 4. Pairplot for Feature Relationships
sns.pairplot(df[['use_count', 'reward_points_used', 'gender', 'shared_account']], hue='shared_account', palette='Set1')
plt.suptitle('Pairplot of Features by shared_account', y=1.02)
plt.savefig("pairplot_features_by_shared_account.png")

## 5. Boxplot for use_count by shared_account
plt.figure(figsize=(12, 6))
sns.boxplot(x='shared_account', y='use_count', data=df)
plt.title('use_count Distribution by shared_account')
plt.xlabel('shared_account')
plt.ylabel('use_count')
plt.savefig("use_count_distribution_by_shared_account.png")

# Model Building

## 1. Supervised Learning: Logistic Regression

# Define features and target
X = df[['use_count', 'reward_points_used', 'gender']]
y = df['shared_account']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=200)

# Train the model
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

## 2. Unsupervised Learning: KMeans Clustering

# Initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
df['Cluster'] = kmeans.fit_predict(X)

# Visualize Clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x='use_count', y='reward_points_used', hue='Cluster', data=df, palette='Set1')
plt.title('KMeans Clusters of Users')
plt.xlabel('use_count')
plt.ylabel('reward_points_used')
plt.savefig("KMeans_clusters_of_users.png")
