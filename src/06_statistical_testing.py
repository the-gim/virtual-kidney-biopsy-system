import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import xgboost as xgb
from scipy.stats import chi2
import itertools
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

# Fix for Apple Silicon (M1/M2/M3) PyTorch OpenMP deadlock/hang
torch.set_num_threads(1)

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 6: STATISTICAL TESTING (McNemar's Test)")
print("=" * 80)

BASE_WORKSPACE = '/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research'
PROCESSED_DATA_PATH = os.path.join(BASE_WORKSPACE, 'data/processed/')
RESULTS_PATH = os.path.join(BASE_WORKSPACE, 'outputs/results/')
FIGURES_PATH = os.path.join(BASE_WORKSPACE, 'outputs/figures/')
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# 1. Load Data
print("📂 Loading data...")
X_train_ad = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_train_selected.csv'))
y_train_ad = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv')).squeeze()
X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_test_selected.csv'))
y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv')).squeeze()

# 2. PyTorch CNN Definition
class CNN1D(nn.Module):
    def __init__(self, input_dim=20, num_classes=2):
        super(CNN1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        conv_out_size = ((input_dim - 3 + 1) // 2)
        self.fc1 = nn.Linear(32 * conv_out_size, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

X_train_pt = X_train_ad.values.astype(np.float32)
y_train_pt = y_train_ad.values.astype(np.int64)
X_test_pt  = X_test.values.astype(np.float32)
y_test_pt  = y_test.values.astype(np.int64)

# 3. Model Definitions
n0 = (y_train_ad == 0).sum()
n1 = (y_train_ad == 1).sum()
scale_w = n0 / n1 if n1 > 0 else 1.0

models = {
    'SVM': SVC(kernel='rbf', C=1.0, probability=False, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, 
                                 scale_pos_weight=scale_w, random_state=42, 
                                 use_label_encoder=False, eval_metric='logloss'),
    'CNN': NeuralNetClassifier(module=CNN1D, module__input_dim=X_train_ad.shape[1], 
                               module__num_classes=2, max_epochs=20, lr=0.001, 
                               batch_size=256, optimizer=torch.optim.Adam, 
                               criterion=nn.CrossEntropyLoss, iterator_train__shuffle=True, 
                               device='cpu', verbose=0)
}

# 4. Train & Predict
print("\n🧠 Rapidly training Top 3 Models to generate test predictions...")
preds = {}
for name, model in models.items():
    print(f"   -> Training {name}...")
    if name == 'CNN':
        model.fit(X_train_pt, y_train_pt)
        preds[name] = model.predict(X_test_pt)
    else:
        model.fit(X_train_ad, y_train_ad)
        preds[name] = model.predict(X_test)

# 5. McNemar's Test
print("\n" + "=" * 80)
print("TEST 1: McNEMAR'S TEST")
print("=" * 80)

print("""
   PURPOSE: Compare two models directly
   
   H0 (Null):       Both models make same errors
   H1 (Alternative): Models make significantly different errors
   
   If p < 0.05 → Reject H0 → Models are significantly different
   If p > 0.05 → Accept H0 → No significant difference
""")

def mcnemar_test(y_true, y_pred1, y_pred2, name1, name2):
    """
    McNemar's test to compare two classifiers
    """
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    b = np.sum(correct1 & ~correct2)  # Model1 right, Model2 wrong
    c = np.sum(~correct1 & correct2)  # Model1 wrong, Model2 right

    if b + c == 0:
        return 1.0, 0.0, b, c

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - chi2.cdf(chi2_stat, df=1)

    return p_value, chi2_stat, b, c

model_names = list(preds.keys())
comparisons = list(itertools.combinations(model_names, 2))

mcnemar_results = []

print(f"\n   {'Comparison':<25} {'Chi2':>8} {'p-value':>10} {'b':>6} {'c':>6} {'Significant':>12}")
print(f"   {'-'*70}")

p_values = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
for name in model_names:
    p_values.loc[name, name] = 1.0

for name1, name2 in comparisons:
    pred1 = preds[name1]
    pred2 = preds[name2]
    
    p_val, chi2_stat, b, c = mcnemar_test(
        y_test, pred1, pred2, name1, name2
    )
    
    # Fill symmetric matrix for heatmap
    p_values.loc[name1, name2] = p_val
    p_values.loc[name2, name1] = p_val
    
    significant = '✅ YES' if p_val < 0.05 else '❌ NO'
    print(f"   {name1} vs {name2:<15} {chi2_stat:>8.4f} {p_val:>10.4f} {b:>6} {c:>6} {significant:>12}")

    mcnemar_results.append({
        'Comparison': f'{name1} vs {name2}',
        'Chi2':       round(chi2_stat, 4),
        'p-value':    round(p_val, 4),
        'b (M1>M2)':  b,
        'c (M2>M1)':  c,
        'Significant': 'Yes' if p_val < 0.05 else 'No',
    })

mcnemar_df = pd.DataFrame(mcnemar_results)
mcnemar_df.to_csv(os.path.join(RESULTS_PATH, 'mcnemar_test.csv'), index=False)
print(f"\n   ✅ Saved mcnemar_test.csv")

# 6. Generate Heatmap
print("\n🎨 Generating P-Value Heatmap...")
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')

# Create a mask for the upper triangle so we only show bottom triangle
mask = np.triu(np.ones_like(p_values, dtype=bool))

sns.heatmap(p_values, mask=mask, annot=True, fmt=".4f", cmap='coolwarm', 
            vmax=0.1, vmin=0, center=0.05, 
            cbar_kws={'label': 'P-Value'}, ax=ax,
            annot_kws={"size": 12, "weight": "bold"})

ax.set_title("McNemar's Test P-Values", fontsize=15, fontweight='bold', color='white', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_07_statistical_significance.png'), dpi=200, facecolor='#0f1117')
plt.close()

print(f"✅ Statistical testing complete. Results and graphs saved.")
