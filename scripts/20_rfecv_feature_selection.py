import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
RFECV_FEATURES_OUT = os.path.join(DATA_DIR, "selected_features_rfecv.csv")
PLOT_OUT = os.path.join(DATA_DIR, "rfecv_performance.png")

print("="*80)
print("RECURSIVE FEATURE ELIMINATION WITH CROSS-VALIDATION (RFECV)")
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

print("Setting up RFECV with XGBoost...")
# Using a slightly shallower tree to speed up the massive CV search
estimator = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)

# step=5 means it drops the 5 weakest features every iteration
# scoring='f1_macro' ensures we are optimizing for the metric that matters most
rfecv = RFECV(
    estimator=estimator,
    step=5,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring='f1_macro',
    min_features_to_select=5,
    n_jobs=-1
)

print("Running exhaustive RFECV Search (This may take a few minutes)...")
try:
    rfecv.fit(X_scaled, y)
except Exception as e:
    print(f"Error during fit: {e}")

print(f"\nOptimal number of features found: {rfecv.n_features_}")

# Plot number of features VS. cross-validation scores
plt.figure(figsize=(10, 6), facecolor='#0f1117')
plt.style.use('dark_background')
plt.xlabel("Number of features selected", color='white')
plt.ylabel("Cross validation score (F1 Macro)", color='white')

# Ensure we plot correctly regardless of scikit-learn version
if hasattr(rfecv, 'cv_results_'):
    scores = rfecv.cv_results_['mean_test_score']
else:
    scores = rfecv.grid_scores_

plt.plot(range(5, len(scores) * 5 + 5, 5)[:len(scores)], scores, color='#26c6da', linewidth=2)
plt.axvline(rfecv.n_features_, color='#ef5350', linestyle='--', label=f'Optimal: {rfecv.n_features_} features')
plt.title('RFECV: Feature Count vs F1-Score', color='white', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=200, facecolor='#0f1117')
plt.close()

# Extract the optimal features
optimal_features = X_df.columns[rfecv.support_].tolist()
print(f"\nTop Features Selected:\n{optimal_features[:10]} ...")

print(f"\nSaving optimal dataset to {RFECV_FEATURES_OUT}...")
new_df = df[['severity'] + optimal_features]
new_df.to_csv(RFECV_FEATURES_OUT)

print("Done!")
