import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 3: FEATURE ANALYSIS & BASELINE EVALUATION")
print("=" * 80)

BASE_WORKSPACE = '/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research'
PROCESSED_DATA_PATH = os.path.join(BASE_WORKSPACE, 'data/processed/')
FIGURES_PATH = os.path.join(BASE_WORKSPACE, 'outputs/figures/')
os.makedirs(FIGURES_PATH, exist_ok=True)

# Load selected features
print("📂 Loading selected train and test datasets...")
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_train_selected.csv'))
X_test  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'X_test_selected.csv'))
y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv'))['target_binary']
y_test  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv'))['target_binary']

plt.style.use('dark_background')

# 1. Correlation Matrix of Top Features
print("📊 Generating Correlation Matrix...")
corr = X_train.corr()
plt.figure(figsize=(14, 12), facecolor='#0f1117')
ax = plt.gca()
ax.set_facecolor('#1a1d27')
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Correlation Matrix of Selected Features', fontsize=16, fontweight='bold', color='white')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_00_correlation_matrix.png'), dpi=200, facecolor='#0f1117')
plt.close()

# 2. Baseline Model for initial Confusion Matrix
print("🧠 Training Baseline Logistic Regression for initial analysis...")
baseline_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
baseline_model.fit(X_train, y_train)

y_pred_base = baseline_model.predict(X_test)

print("\n--- Baseline Logistic Regression Report ---")
print(classification_report(y_test, y_pred_base, target_names=['Mild', 'Severe']))

print("📊 Generating Baseline Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_base)
plt.figure(figsize=(6, 5), facecolor='#0f1117')
ax = plt.gca()
ax.set_facecolor('#1a1d27')
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Mild', 'Severe'], 
            yticklabels=['Mild', 'Severe'],
            linewidths=1, linecolor='#0f1117',
            annot_kws={'size':14, 'weight':'bold'})
plt.title('Baseline Model\nConfusion Matrix', fontweight='bold', fontsize=14, color='white')
plt.xlabel('Predicted', color='white')
plt.ylabel('Actual', color='white')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_00_baseline_confusion_matrix.png'), dpi=200, facecolor='#0f1117')
plt.close()

print(f"✅ Feature analysis complete. Graphs saved to {FIGURES_PATH}")
