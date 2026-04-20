# ============================================================
# CONCEPT: Regression — Linear, Ridge, Lasso + Logistic Classification
# Includes: Log-transform target, MAE, MSE, RMSE, R2
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

# --- Load Data ---
data = pd.read_csv('your_dataset.csv')
data = data.dropna(axis=1, how='all')

# --- Clean numeric target column ---
data['numeric_col'] = data['numeric_col'].astype(str).str.replace(',', '', regex=True)
data['numeric_col'] = pd.to_numeric(data['numeric_col'], errors='coerce')

# --- Outlier Detection (IQR) ---
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
    iqr = q3 - q1

# --- Feature Engineering ---
# data['Date_Col'] = pd.to_datetime(data['Date_Col'], format='%d/%m/%Y', errors='coerce')
# data['Year']  = data['Date_Col'].dt.year
# data['Month'] = data['Date_Col'].dt.month

data = data.dropna(subset=['numeric_col']).copy()

# --- Create classification target ---
try:
    data['Target_Class'] = pd.qcut(data['numeric_col'], q=3, labels=['Low', 'Medium', 'High'])
except ValueError:
    data['Target_Class'] = pd.cut(data['numeric_col'], bins=3, labels=['Low', 'Medium', 'High'])

# --- Log-transform regression target (stabilises variance) ---
data['Log_Target'] = np.log1p(data['numeric_col'])

# --- Feature Matrix ---
remove_cols = ['numeric_col', 'Log_Target', 'Target_Class']
X = data.drop(columns=remove_cols, errors='ignore').copy()
y_reg = data['Log_Target']
y_clf = data['Target_Class']

# --- Impute ---
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object', 'string']).columns:
    mode_val = X[col].mode(dropna=True)
    X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')

# --- Encode ---
encoders = {}
for col in X.select_dtypes(include=['object', 'string']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# --- Split & Scale (shared split for fair comparison) ---
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- Regression Evaluation Helper ----
def evaluate_regression_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_te)
    y_true = np.expm1(y_te)
    y_pred = np.clip(np.expm1(pred_log), a_min=0, a_max=None)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    print(f"\n===== {name} RESULTS =====")
    print(f"MAE: {mae:.4f}  |  MSE: {mse:.4f}  |  RMSE: {rmse:.4f}  |  R2: {r2:.4f}")
    return {'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2_Score': r2}

# ---- Classification Evaluation Helper ----
def evaluate_logistic_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"\n===== {name} RESULTS =====")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_te, preds))
    print("Classification Report:\n", classification_report(y_te, preds))
    return {'Model': name, 'Accuracy': acc}

# ---- Run Regression Models ----
linear_report = evaluate_regression_model('LINEAR REGRESSION', LinearRegression(),
    X_train_scaled, X_test_scaled, y_reg_train, y_reg_test)

ridge_report = evaluate_regression_model('RIDGE REGRESSION', Ridge(alpha=1.0, random_state=42),
    X_train_scaled, X_test_scaled, y_reg_train, y_reg_test)

lasso_report = evaluate_regression_model('LASSO REGRESSION', Lasso(alpha=0.001, max_iter=5000, random_state=42),
    X_train_scaled, X_test_scaled, y_reg_train, y_reg_test)

print("\n===== REGRESSION COMPARISON =====")
print(pd.DataFrame([linear_report, ridge_report, lasso_report]))

# ---- Run Logistic Regression (Classification) ----
logistic_report = evaluate_logistic_model('LOGISTIC REGRESSION',
    LogisticRegression(max_iter=2000, random_state=42),
    X_train_scaled, X_test_scaled, y_clf_train, y_clf_test)

print("\n===== FINAL SUMMARY =====")
regression_df = pd.DataFrame([linear_report, ridge_report, lasso_report])
print("Best Regression Model (by R2):")
print(regression_df.sort_values('R2_Score', ascending=False).head(1))
print("\nLogistic Regression:")
print(pd.DataFrame([logistic_report]))
