# ============================================================
# CONCEPT: Decision Tree Classification + K-Fold Cross Validation
# Includes: CART (Gini), Entropy Tree, Feature Importance
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Load & Clean ---
data = pd.read_csv('your_dataset.csv')
data = data.dropna(axis=1, how='all')

# --- Create classification target ---
try:
    data['Target_Class'] = pd.qcut(data['numeric_target_col'], q=3, labels=['Low', 'Medium', 'High'])
except ValueError:
    data['Target_Class'] = pd.cut(data['numeric_target_col'], bins=3, labels=['Low', 'Medium', 'High'])

# --- Features and Target ---
X = data.drop(columns=['numeric_target_col', 'Target_Class'])
y = data['Target_Class']

# --- Fill Missing Values ---
for col in X.select_dtypes(include=['int64', 'float64']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    mode_val = X[col].mode(dropna=True)
    X[col] = X[col].fillna(mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown')

# --- Label Encode ---
encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# --- Train / Test Split & Scaling ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Feature Importance using Decision Tree ---
feature_tree = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=6)
feature_tree.fit(X_train_scaled, y_train)
feature_importance = pd.Series(feature_tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top 10 Features:\n", feature_importance.head(10))

plt.figure(figsize=(10, 5))
feature_importance.head(10).plot(kind='bar', color='teal')
plt.title('Top 10 Feature Importances')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- CART Classifier (Gini) ---
cart_clf = DecisionTreeClassifier(criterion='gini', max_depth=6, min_samples_split=5, random_state=42)
cart_clf.fit(X_train_scaled, y_train)

# --- Entropy Classifier ---
entropy_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=5, random_state=42)
entropy_clf.fit(X_train_scaled, y_train)

# --- Visualize Both Trees ---
plt.figure(figsize=(22, 10))
plt.subplot(1, 2, 1)
plot_tree(cart_clf, feature_names=X.columns, class_names=[str(c) for c in cart_clf.classes_], filled=True, rounded=True, fontsize=7)
plt.title('Decision Tree (Gini Index)')
plt.subplot(1, 2, 2)
plot_tree(entropy_clf, feature_names=X.columns, class_names=[str(c) for c in entropy_clf.classes_], filled=True, rounded=True, fontsize=7)
plt.title('Decision Tree (Entropy)')
plt.tight_layout()
plt.show()

# --- Evaluate ---
y_pred_cart = cart_clf.predict(X_test_scaled)
y_pred_entropy = entropy_clf.predict(X_test_scaled)

print("\n===== CART (Gini) RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_cart))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_cart))
print("Classification Report:\n", classification_report(y_test, y_pred_cart))

print("\n===== Entropy Tree RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred_entropy))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report:\n", classification_report(y_test, y_pred_entropy))

# --- K-Fold Cross Validation ---
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all_scaled = scaler.fit_transform(X)

cv_cart = cross_val_score(cart_clf, X_all_scaled, y, cv=kfold, scoring='accuracy')
cv_entropy = cross_val_score(entropy_clf, X_all_scaled, y, cv=kfold, scoring='accuracy')

print("\n===== K-FOLD CROSS VALIDATION (5-Fold) =====")
print("CART CV Scores:", np.round(cv_cart, 4), "| Mean:", round(cv_cart.mean(), 4))
print("Entropy CV Scores:", np.round(cv_entropy, 4), "| Mean:", round(cv_entropy.mean(), 4))
