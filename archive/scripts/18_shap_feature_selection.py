import pandas as pd
import os
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
SHAP_FEATURES_OUT = os.path.join(DATA_DIR, "selected_features_shap.csv")

print("="*80)
print("SHAP FEATURE SELECTION")
print("="*80)

print(f"Loading {FEATURES_IN}...")
df = pd.read_csv(FEATURES_IN, index_col=0)

y = df['severity'].values
X_df = df.drop(columns=['severity']).fillna(0)
X = X_df.values

X = X.astype(np.float32)
y = y.astype(np.int64)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Compute class weights for XGBoost
classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
sample_weights = np.zeros(len(y), dtype=np.float32)
for c in classes:
    sample_weights[y == c] = weights[int(c)]

print("Training XGBoost to compute SHAP values...")
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1)
model.fit(X_scaled, y, sample_weight=sample_weights)

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
# Sample to save time if dataset is huge, but 45k is fast enough for TreeExplainer
shap_values = explainer.shap_values(X_scaled)

# For multi-class, shap_values is shape (n_samples, n_features, n_classes)
# We average the absolute SHAP values across all classes and all samples
mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))

# Create a DataFrame of feature importances
feature_importance = pd.DataFrame({
    'Feature': X_df.columns,
    'SHAP_Importance': mean_abs_shap
})

# Sort by importance
feature_importance = feature_importance.sort_values(by='SHAP_Importance', ascending=False)
print("\nTop 20 Features by SHAP Importance:")
print(feature_importance.head(20))

# Select Top 30 features
TOP_N = 30
top_features = feature_importance.head(TOP_N)['Feature'].tolist()

print(f"\nSelecting top {TOP_N} features and saving to new CSV...")
new_df = df[['severity'] + top_features]
new_df.to_csv(SHAP_FEATURES_OUT)

print(f"Saved to {SHAP_FEATURES_OUT}")
print("Done!")
