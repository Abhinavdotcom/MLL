# ============================================================
# CONCEPT: Ensemble Methods
# Includes: Bagging, AdaBoost (Boosting), Stacking, Random Forest
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# --- Load Data ---
data = pd.read_csv('your_dataset.csv')

# --- Clean numeric column ---
data['numeric_col'] = data['numeric_col'].astype(str).str.replace(',', '', regex=True)
data['numeric_col'] = pd.to_numeric(data['numeric_col'], errors='coerce')

# --- Outlier Detection (IQR) ---
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    iqr = q3 - q1

# --- Create Classification Target ---
data = data.dropna(subset=['numeric_col']).copy()
try:
    data['Target_Class'] = pd.qcut(data['numeric_col'], q=3, labels=['Low', 'Medium', 'High'])
except ValueError:
    data['Target_Class'] = pd.cut(data['numeric_col'], bins=3, labels=['Low', 'Medium', 'High'])

# --- Features / Target ---
X = data.drop(columns=['numeric_col', 'Target_Class'])
y = data['Target_Class']

# --- Impute ---
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object', 'string']).columns:
    mode_val = X[col].mode(dropna=True)
    X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')

# --- Encode ---
for col in X.select_dtypes(include=['object', 'string']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# --- Split & Scale ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Evaluation Helper ---
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"\n===== {name} RESULTS =====")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_te, preds))
    print("Classification Report:\n", classification_report(y_te, preds))
    return acc

# ---- Bagging ----
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
    n_estimators=100,
    random_state=42
)

# ---- AdaBoost (Boosting) ----
boosting = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)

# ---- Stacking ----
stacking = StackingClassifier(
    estimators=[
        ('dt',  DecisionTreeClassifier(max_depth=6, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42))
    ],
    final_estimator=LogisticRegression(max_iter=200),
    passthrough=False
)

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42)

# --- Run All ---
acc_bag   = evaluate_model('BAGGING',          bagging,  X_train_scaled, X_test_scaled, y_train, y_test)
acc_boost = evaluate_model('BOOSTING (ADABOOST)', boosting, X_train_scaled, X_test_scaled, y_train, y_test)
acc_stack = evaluate_model('STACKING',         stacking, X_train_scaled, X_test_scaled, y_train, y_test)
acc_rf    = evaluate_model('RANDOM FOREST',    rf,       X_train_scaled, X_test_scaled, y_train, y_test)

# --- Comparison Table ---
comparison = pd.DataFrame({
    'Model': ['Bagging', 'Boosting', 'Stacking', 'Random Forest'],
    'Accuracy': [acc_bag, acc_boost, acc_stack, acc_rf]
})
print('\n===== MODEL COMPARISON =====')
print(comparison)
