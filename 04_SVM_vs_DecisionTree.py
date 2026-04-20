# ============================================================
# CONCEPT: SVM vs Decision Tree Classification
# Includes: Outlier detection (IQR), MSE comparison
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

# --- Load Data ---
data = pd.read_csv('your_dataset.csv')

# --- Clean numeric column ---
data['numeric_col'] = data['numeric_col'].astype(str).str.replace(',', '', regex=True)
data['numeric_col'] = pd.to_numeric(data['numeric_col'], errors='coerce')

# --- Outlier Detection (IQR) ---
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
outlier_counts = {}
for col in numeric_cols:
    q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    iqr = q3 - q1
    outlier_counts[col] = ((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum()
print("Outliers per column:\n", pd.Series(outlier_counts))

# --- Create Classification Target ---
data = data.dropna(subset=['numeric_col']).copy()
try:
    data['Target_Class'] = pd.qcut(data['numeric_col'], q=3, labels=['Low', 'Medium', 'High'])
except ValueError:
    data['Target_Class'] = pd.cut(data['numeric_col'], bins=3, labels=['Low', 'Medium', 'High'])

# --- Features / Target ---
X = data.drop(columns=['numeric_col', 'Target_Class'])
y = data['Target_Class']

# --- Impute Missing ---
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object', 'string']).columns:
    mode_val = X[col].mode(dropna=True)
    X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')

# --- Label Encode ---
encoders = {}
for col in X.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# --- Split & Scale ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For MSE computation we need numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))
y_test_encoded = label_encoder.transform(y_test.astype(str))

# ---- SVM (RBF Kernel) ----
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_clf.fit(X_train_scaled, y_train)
svm_pred = svm_clf.predict(X_test_scaled)
svm_pred_encoded = label_encoder.transform(svm_pred.astype(str))

print("\n===== SVM RESULTS =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("MSE:", mean_squared_error(y_test_encoded, svm_pred_encoded))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print("Classification Report:\n", classification_report(y_test, svm_pred))

# ---- Decision Tree ----
dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=5, random_state=42)
dt_clf.fit(X_train_scaled, y_train)
dt_pred = dt_clf.predict(X_test_scaled)
dt_pred_encoded = label_encoder.transform(dt_pred.astype(str))

print("\n===== DECISION TREE RESULTS =====")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print("MSE:", mean_squared_error(y_test_encoded, dt_pred_encoded))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Classification Report:\n", classification_report(y_test, dt_pred))

# ---- Comparison Table ----
comparison = pd.DataFrame({
    'Model': ['SVM (RBF)', 'Decision Tree'],
    'Accuracy': [accuracy_score(y_test, svm_pred), accuracy_score(y_test, dt_pred)],
    'MSE': [mean_squared_error(y_test_encoded, svm_pred_encoded),
            mean_squared_error(y_test_encoded, dt_pred_encoded)]
})
print("\n===== MODEL COMPARISON =====")
print(comparison)
