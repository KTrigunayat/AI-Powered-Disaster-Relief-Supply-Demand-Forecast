# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

# Step 1: Load the dataset with error handling for encoding issues
try:
    data = pd.read_csv('Dataset.csv', encoding='utf-8')
    print("File loaded successfully with UTF-8 encoding.")
except UnicodeDecodeError:
    data = pd.read_csv('Dataset.csv', encoding='latin1')
    print("File loaded successfully with Latin-1 encoding.")

# Display basic dataset info
print("\nDataset Shape:", data.shape)
print("Columns Available:", data.columns)

# Step 2: Convert date columns to datetime format (if applicable)
date_columns = ['Start Date', 'End Date']  # Adjust column names as needed
for col in date_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col], errors='coerce')

# Step 3: Handle missing values
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# Fill missing numerical values with median
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    data[col].fillna(data[col].median(), inplace=True)

# Fill missing categorical values with mode
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Step 4: Remove duplicate records
data.drop_duplicates(inplace=True)

# Step 5: Feature Engineering
# Create a 'Severity Index' as a combination of key impact factors
if {'Total Deaths','No. Injured','No. Affected','No. Homeless','Total Affected',"Total Damage ('000 US$)"}.issubset(data.columns):#Total Deaths,No. Injured,No. Affected,No. Homeless,Total Affected,Total Damage ('000 US$)
    data['Severity_Index'] = (
        data['Total Deaths'] * 0.5 + data['Total Damage ($USD)'] * 0.2 + data['No. Injured'] * 0.1 + data['No. Affected'] * 0.1 + data['No. Homeless'] * 0.1
    )

# Extract year from start date (if applicable)
if 'Start Date' in data.columns:
    data['Year'] = data['Start Date'].dt.year

# One-Hot Encode categorical variables (e.g., Disaster Type)
if 'Disaster Type' in data.columns:
    data = pd.get_dummies(data, columns=['Disaster Type'], drop_first=True)

# Step 6: Feature Scaling using StandardScaler
scaler = StandardScaler()
scaled_cols = num_cols.tolist()  # Convert column index to list
if 'Severity_Index' in data.columns:
    scaled_cols.append('Severity_Index')  # Include engineered feature

data[scaled_cols] = scaler.fit_transform(data[scaled_cols])

# Step 7: Dimensionality Reduction using PCA & SVD
pca = PCA(n_components=5)  # Adjust components based on variance analysis
data_pca = pca.fit_transform(data[scaled_cols])
print("\nExplained Variance by PCA Components:", pca.explained_variance_ratio_)

svd = TruncatedSVD(n_components=5)
data_svd = svd.fit_transform(data[scaled_cols])

# Step 8: Save the cleaned dataset
data.to_csv('emdat_preprocessed.csv', index=False)
print("\nData Preprocessing Completed! File Saved as 'emdat_preprocessed.csv'.")
