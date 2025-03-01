# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import skew, kurtosis

# Load the preprocessed dataset
data = pd.read_csv('emdat_preprocessed.csv')

# Set visualization style
sns.set_style("darkgrid")

# Step 1: Basic Dataset Overview
print("\nDataset Shape:", data.shape)
print("\nColumn Data Types:\n", data.dtypes)
print("\nFirst 5 Rows:\n", data.head())

# Step 2: Check for Missing Values
print("\nMissing Values Per Column:\n", data.isnull().sum())

# Visualize missing values using a heatmap
plt.figure(figsize=(12, 5))
msno.heatmap(data)
plt.title("Missing Values Heatmap")
plt.show()

# Step 3: Summary Statistics
print("\nSummary Statistics:\n", data.describe())

# Step 4: Detect Outliers using Box Plots
plt.figure(figsize=(15, 6))
sns.boxplot(data=data.select_dtypes(include=['int64', 'float64']))
plt.xticks(rotation=90)
plt.title("Box Plot of Numerical Features (Outlier Detection)")
plt.show()

# Step 5: Skewness & Kurtosis Analysis
skewness = data.skew()
kurt = data.kurtosis()
print("\nFeature Skewness:\n", skewness)
print("\nFeature Kurtosis:\n", kurt)

# Step 6: Correlation Analysis
corr_matrix = data.corr()

# Display correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 7: Distribution of Key Features
num_cols = data.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_cols[:6]):  # Plot only first 6 numerical features for clarity
    plt.subplot(2, 3, i + 1)
    sns.histplot(data[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Step 8: Relationship Between Disaster Type and Severity Index (if exists)
if 'Severity_Index' in data.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Disaster Type_Flood', y='Severity_Index', data=data)
    plt.title("Impact of Disaster Type on Severity Index")
    plt.xticks(rotation=45)
    plt.show()

print("\nEDA Completed! Check the visualizations.")
