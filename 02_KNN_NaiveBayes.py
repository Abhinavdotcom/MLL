# ============================================================
# CONCEPT: KNN & Naive Bayes Classification
# Use case: Classify a target into multiple categories
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load & Clean ---
data = pd.read_csv('your_dataset.csv')
data = data.dropna(axis=1, how='all')

# --- Create classification target using quantile bins ---
data['Target_Class'] = pd.qcut(
    data['numeric_target_col'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# --- Features and Target ---
X = data.drop(columns=['numeric_target_col', 'Target_Class'])
y = data['Target_Class']

# --- Fill Missing Values ---
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].mean())
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# --- Label Encode Categorical Columns ---
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col])

# --- Train / Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Scale Features (important for KNN) ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- KNN Classifier ----
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2,
    weights='uniform'
)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("===== KNN RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))

# ---- Gaussian Naive Bayes ----
nb = GaussianNB(priors=None, var_smoothing=1e-9)
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

print("\n===== NAIVE BAYES RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))
