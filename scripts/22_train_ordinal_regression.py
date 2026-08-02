import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
# We will use the ultra-clean SHAP features since they performed the best!
FEATURES_IN = os.path.join(DATA_DIR, "selected_features_shap.csv")
PLOT_OUT = os.path.join(DATA_DIR, "ordinal_regression_confusion_matrices.png")

print("="*80)
print("ORDINAL REGRESSION & CLINICAL UTILITY (WITHIN-1-GRADE ACCURACY)")
print("="*80)

print(f"\n📂 Loading SHAP optimized data from {FEATURES_IN}...")
df = pd.read_csv(FEATURES_IN, index_col=0)

y = df['severity'].values
X = df.drop(columns=['severity']).fillna(0).values

X = X.astype(np.float32)
y = y.astype(np.int64)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
}

# 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}

for name, model in models.items():
    print(f"\n🚀 Evaluating {name} using 5-Fold CV...")
    
    exact_accs = []
    within_1_accs = []
    f1_macros = []
    
    all_y_true = []
    all_y_pred = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        
        # XGBoost handles weights slightly differently
        if name == 'XGBoost':
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            sample_weights = np.zeros(len(y_train), dtype=np.float32)
            for c in classes:
                sample_weights[y_train == c] = weights[int(c)]
            model.fit(X_train_sc, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_sc, y_train)
            
        y_pred = model.predict(X_val_sc)
        
        # Metrics
        exact_accs.append(accuracy_score(y_val, y_pred))
        f1_macros.append(f1_score(y_val, y_pred, average='macro'))
        
        # THE CLINICAL METRIC: Within-1-Grade Accuracy
        within_1 = np.mean(np.abs(y_val - y_pred) <= 1)
        within_1_accs.append(within_1)
        
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        
    print(f"   [Mathematical] Exact Accuracy:   {np.mean(exact_accs)*100:.2f}%")
    print(f"   [Mathematical] Macro F1-Score:   {np.mean(f1_macros):.4f}")
    print(f"   [CLINICAL]     Within-1-Grade:   {np.mean(within_1_accs)*100:.2f}%  <-- THE MEDICAL METRIC")
    
    results[name] = confusion_matrix(all_y_true, all_y_pred)

# Plot Special Confusion Matrices showing the "Safe Clinical Zone"
plt.figure(figsize=(18, 5), facecolor='#0f1117')
plt.style.use('dark_background')

labels = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3']

for idx, (name, cm) in enumerate(results.items()):
    ax = plt.subplot(1, 3, idx + 1)
    
    # Create a custom mask to highlight the "Within-1-Grade" safe zone
    mask = np.zeros_like(cm, dtype=bool)
    for i in range(4):
        for j in range(4):
            if abs(i - j) <= 1:
                mask[i, j] = True
                
    # Plot the full matrix in grey
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax, alpha=0.3)
    
    # Overlay the Safe Zone in Green
    sns.heatmap(cm, mask=~mask, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_title(f"{name}\nGreen = Clinically Acceptable (Within 1 Grade)", fontsize=12, color='white', pad=15)
    ax.set_xlabel('AI Predicted Grade', color='white')
    ax.set_ylabel('True Patient Grade', color='white')

plt.tight_layout()
plt.savefig(PLOT_OUT, dpi=200, facecolor='#0f1117')
print(f"\n🎨 Saved clinical confusion matrices to {PLOT_OUT}")
print("\n✅ Script Complete!")
