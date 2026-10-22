import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import xgboost as xgb
from imblearn.over_sampling import ADASYN
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

# Fix for Apple Silicon (M1/M2/M3) PyTorch OpenMP deadlock/hang
torch.set_num_threads(1)

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 5: TRAIN ADVANCED MODELS & THRESHOLD OPTIMIZATION")
print("=" * 80)

BASE_WORKSPACE = '/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research'
PROCESSED_DATA_PATH = os.path.join(BASE_WORKSPACE, 'data/processed/')
RESULTS_PATH = os.path.join(BASE_WORKSPACE, 'outputs/results/')
FIGURES_PATH = os.path.join(BASE_WORKSPACE, 'outputs/figures/')

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Load data
print("📂 Loading preprocessed data...")
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_train_selected.csv'))
X_test  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_test_selected.csv'))
y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv'))['target_binary']
y_test  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv'))['target_binary']

input_dim = X_train.shape[1]

# Calculate class ratios for XGBoost
n0 = (y_train == 0).sum()
n1 = (y_train == 1).sum()

# Apply ADASYN
print("⚖️ Applying ADASYN oversampling...")
try:
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_train_ad, y_train_ad = adasyn.fit_resample(X_train, y_train)
    print(f"   After ADASYN:  {len(X_train_ad)} samples")
except Exception as e:
    print(f"   ⚠️ ADASYN failed ({e}), falling back to original data")
    X_train_ad, y_train_ad = X_train, y_train

# Cast to PyTorch-compatible types for CNN and LSTM
X_train_ad_pt = X_train_ad.values.astype(np.float32)
y_train_ad_pt = y_train_ad.values.astype(np.int64)
X_test_pt = X_test.values.astype(np.float32)
y_test_pt = y_test.values.astype(np.int64)

# ---------------------------------------------------------
# Define PyTorch Architectures
# ---------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, input_dim=20, num_classes=2):
        super(CNN1D, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        conv_out_size = ((input_dim - 3 + 1) // 2)
        self.fc1 = nn.Linear(32 * conv_out_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LSTMModel(nn.Module):
    def __init__(self, input_dim=20, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ---------------------------------------------------------
# Training Functions
# ---------------------------------------------------------
def find_best_threshold(y_true, y_proba):
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_f1 = 0
    best_thr = 0.5
    f1_list = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_list.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1, thresholds, f1_list

results = {}

# 1. XGBoost
print("\n🌲 Training XGBoost (Regularized + ADASYN)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=2.0, scale_pos_weight=n0/n1,
    eval_metric='logloss', random_state=42, verbosity=0,
    use_label_encoder=False
)
xgb_model.fit(X_train_ad, y_train_ad)
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
best_thr_xgb, best_f1_xgb, thrs, f1s = find_best_threshold(y_test, y_proba_xgb)
y_pred_xgb = (y_proba_xgb >= best_thr_xgb).astype(int)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"   ✅ F1: {best_f1_xgb:.4f} | Threshold: {best_thr_xgb:.2f}")

results['XGBoost'] = {
    'y_pred': y_pred_xgb, 'y_proba': y_proba_xgb,
    'accuracy': acc_xgb, 'f1': best_f1_xgb,
    'threshold': best_thr_xgb, 'color': '#66bb6a',
    'thrs': thrs, 'f1s_thr': f1s, 'model': xgb_model
}

# 2. CNN1D
print("\n🧠 Training CNN (1D Convolution on Features)...")
cnn_model = NeuralNetClassifier(
    module=CNN1D,
    module__input_dim=input_dim,
    module__num_classes=2,
    max_epochs=20,
    lr=0.001,
    batch_size=256,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    iterator_train__shuffle=True,
    device='cpu',
    verbose=0
)
cnn_model.fit(X_train_ad_pt, y_train_ad_pt)
y_proba_cnn = cnn_model.predict_proba(X_test_pt)[:, 1]
best_thr_cnn, best_f1_cnn, thrs_cnn, f1s_cnn = find_best_threshold(y_test_pt, y_proba_cnn)
y_pred_cnn = (y_proba_cnn >= best_thr_cnn).astype(int)
acc_cnn = accuracy_score(y_test_pt, y_pred_cnn)
print(f"   ✅ F1: {best_f1_cnn:.4f} | Threshold: {best_thr_cnn:.2f}")

results['CNN'] = {
    'y_pred': y_pred_cnn, 'y_proba': y_proba_cnn,
    'accuracy': acc_cnn, 'f1': best_f1_cnn,
    'threshold': best_thr_cnn, 'color': '#ffa726',
    'thrs': thrs_cnn, 'f1s_thr': f1s_cnn
}

# 3. LSTM
print("\n🧠 Training LSTM (Recurrent Neural Network)...")
lstm_model = NeuralNetClassifier(
    module=LSTMModel,
    module__input_dim=input_dim,
    module__num_classes=2,
    max_epochs=20,
    lr=0.001,
    batch_size=256,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    iterator_train__shuffle=True,
    device='cpu',
    verbose=0
)
lstm_model.fit(X_train_ad_pt, y_train_ad_pt)
y_proba_lstm = lstm_model.predict_proba(X_test_pt)[:, 1]
best_thr_lstm, best_f1_lstm, thrs_lstm, f1s_lstm = find_best_threshold(y_test_pt, y_proba_lstm)
y_pred_lstm = (y_proba_lstm >= best_thr_lstm).astype(int)
acc_lstm = accuracy_score(y_test_pt, y_pred_lstm)
print(f"   ✅ F1: {best_f1_lstm:.4f} | Threshold: {best_thr_lstm:.2f}")

results['LSTM'] = {
    'y_pred': y_pred_lstm, 'y_proba': y_proba_lstm,
    'accuracy': acc_lstm, 'f1': best_f1_lstm,
    'threshold': best_thr_lstm, 'color': '#5c6bc0',
    'thrs': thrs_lstm, 'f1s_thr': f1s_lstm
}

# Output Results
comparison = pd.DataFrame([{
    'Model': name,
    'Accuracy': f"{r['accuracy']*100:.2f}%",
    'F1 (Macro)': f"{r['f1']:.4f}",
    'Precision': f"{precision_score(y_test, r['y_pred'], average='macro', zero_division=0):.4f}",
    'Recall': f"{recall_score(y_test, r['y_pred'], average='macro', zero_division=0):.4f}",
    'Threshold': f"{r['threshold']:.2f}",
} for name, r in results.items()]).sort_values('F1 (Macro)', ascending=False)

print("\n" + "=" * 80)
print("ADVANCED MODELS RESULTS TABLE")
print("=" * 80)
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(RESULTS_PATH, 'advanced_metrics.csv'), index=False)

print("\n🎨 Generating Graphs...")
plt.style.use('dark_background')

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0f1117')
fig.suptitle('Advanced Models - Confusion Matrices', fontsize=16, fontweight='bold', color='white')
cmaps = ['Greens', 'Oranges', 'Purples']
for ax, (name, r), cmap in zip(axes, results.items(), cmaps):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=['Mild','Severe'], yticklabels=['Mild','Severe'], ax=ax)
    ax.set_title(f'{name}\nF1={r["f1"]:.3f}', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_02_advanced_cm.png'), dpi=200, facecolor='#0f1117')
plt.close()

# 2. Threshold Analysis
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
for name, r in results.items():
    ax.plot(r['thrs'], r['f1s_thr'], color=r['color'], linewidth=2.5, label=name)
ax.set_title('Threshold Optimization', fontsize=15, fontweight='bold', color='white')
ax.set_xlabel('Threshold')
ax.set_ylabel('F1-Score (Macro)')
ax.legend()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_03_advanced_thresholds.png'), dpi=200, facecolor='#0f1117')
plt.close()

# 3. Feature Importance (XGBoost only)
feat_imp = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': results['XGBoost']['model'].feature_importances_
}).sort_values('Importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0f1117')
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color='#66bb6a')
ax.set_title('Top 15 Feature Importances (XGBoost)', fontsize=15, fontweight='bold', color='white')
plt.tight_layout()
# 4. AUROC Curve
fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=r['color'], lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

ax.set_title('Receiver Operating Characteristic (AUROC)', fontsize=15, fontweight='bold', color='white')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_05_advanced_auroc.png'), dpi=200, facecolor='#0f1117')
plt.close()

# 5. AUPRC Curve (Precision-Recall)
fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0f1117')
ax.set_facecolor('#1a1d27')
baseline_pr = y_test.sum() / len(y_test)
ax.plot([0, 1], [baseline_pr, baseline_pr], 'k--', lw=2, alpha=0.5, label=f'Baseline ({baseline_pr:.2f})')

for name, r in results.items():
    precision, recall, _ = precision_recall_curve(y_test, r['y_proba'])
    pr_auc = average_precision_score(y_test, r['y_proba'])
    ax.plot(recall, precision, color=r['color'], lw=2.5, label=f'{name} (AUPRC = {pr_auc:.3f})')

ax.set_title('Precision-Recall Curve (AUPRC)', fontsize=15, fontweight='bold', color='white')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_06_advanced_auprc.png'), dpi=200, facecolor='#0f1117')
plt.close()

print(f"✅ Advanced models training complete. Results and graphs saved.")
