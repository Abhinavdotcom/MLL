# ============================================================
# CONCEPT: Exploratory Data Analysis (EDA) & Preprocessing
# Dataset: Any CSV with numerical and categorical columns
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- Load Dataset ---
df = pd.read_csv('your_dataset.csv')

# --- Basic Inspection ---
print(df.head())
print(df.tail())
print(df.info())
print(df.describe(include="all"))
print(df.shape)
print(df.dtypes)
print(df.columns)
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.nunique())

# --- Separate Column Types ---
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object', 'string']).columns

# --- Statistics ---
print(df[num_cols].mean())
print(df[num_cols].median())
print(df[num_cols].std())

# --- Handle Missing Values ---
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# --- Remove Duplicates ---
df.drop_duplicates(inplace=True)

# --- Feature Engineering (examples) ---
# df['New_Feature'] = df['Col_A'].str.split(',').str.len()
# df['Date_Col'] = pd.to_datetime(df['Date_Col'], format='%d/%m/%Y', errors='coerce')
# df['Year'] = df['Date_Col'].dt.year

# --- Feature / Target Split ---
X = df.drop(columns=['target_column'])
y = df['target_column']

# --- Feature Scaling ---
minmax_scaler = MinMaxScaler()
X_minmax = X.copy()
X_minmax[num_cols] = minmax_scaler.fit_transform(X_minmax[num_cols])

standard_scaler = StandardScaler()
X_zscore = X.copy()
X_zscore[num_cols] = standard_scaler.fit_transform(X_zscore[num_cols])

# --- Outlier Detection using IQR ---
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
df_clean = df[~outlier_mask]

# --- Visualizations ---
sns.set_style("whitegrid")

# Bar Chart
plt.figure(figsize=(8, 5))
df['categorical_col'].value_counts().head(8).plot(kind='bar', color='coral')
plt.title('Top 8 Categories')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Pie Chart
plt.figure(figsize=(7, 7))
df['categorical_col'].value_counts().head(5).plot(kind='pie', autopct='%1.1f%%')
plt.title('Top 5 Distribution')
plt.ylabel('')
plt.tight_layout()
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(df['numeric_col'].dropna(), bins=15, edgecolor='black', alpha=0.7)
plt.title('Histogram of Numeric Column')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Line Chart
plt.figure(figsize=(8, 5))
df['time_col'].value_counts().sort_index().plot(kind='line', marker='o', color='red')
plt.title('Trend Over Time')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
plt.scatter(df['col_x'], df['col_y'], alpha=0.5)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()

# Violin Plot
plt.figure(figsize=(10, 6))
top_cats = df['categorical_col'].value_counts().head(5).index
sns.violinplot(x='categorical_col', y='numeric_col', data=df[df['categorical_col'].isin(top_cats)])
plt.title('Violin Plot')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
