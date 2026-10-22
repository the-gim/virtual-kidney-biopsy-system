import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import ADASYN
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 4: TRAIN BASIC MODELS")
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

# Apply ADASYN
print("⚖️ Applying ADASYN oversampling...")
try:
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_train_ad, y_train_ad = adasyn.fit_resample(X_train, y_train)
    print(f"   Before ADASYN: {len(X_train)} samples")
    print(f"   After ADASYN:  {len(X_train_ad)} samples")
except Exception as e:
    print(f"   ⚠️ ADASYN failed ({e}), falling back to original data")
    X_train_ad, y_train_ad = X_train, y_train

# Define basic models
models = {
    'ANN': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'SVM': SVC(probability=True, random_state=42)
}

results = []
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0f1117')
fig.suptitle('Basic Models - Confusion Matrices', fontsize=16, fontweight='bold', color='white')

for ax, (name, model) in zip(axes, models.items()):
    print(f"\n🧠 Training {name}...")
    model.fit(X_train_ad, y_train_ad)
    y_pred = model.predict(X_test)
    
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"   ✅ Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
    
    results.append({
        'Model': name,
        'Accuracy': f"{acc*100:.2f}%",
        'F1 (Macro)': f"{f1:.4f}",
        'Precision': f"{prec:.4f}",
        'Recall': f"{rec:.4f}"
    })
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Mild', 'Severe'], yticklabels=['Mild', 'Severe'],
                linewidths=1, linecolor='#0f1117', annot_kws={'size':12, 'weight':'bold'}, ax=ax)
    ax.set_title(f'{name}\nF1={f1:.3f}', fontweight='bold', fontsize=12, color='white')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# Save Confusion Matrices
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_PATH, 'graph_01_basic_models_cm.png'), dpi=200, facecolor='#0f1117')
plt.close()

# Save Results
results_df = pd.DataFrame(results).sort_values('F1 (Macro)', ascending=False)
print("\n" + "=" * 80)
print("BASIC MODELS RESULTS TABLE")
print("=" * 80)
print(results_df.to_string(index=False))

results_path = os.path.join(RESULTS_PATH, 'basic_metrics.csv')
results_df.to_csv(results_path, index=False)
print(f"\n✅ Basic models training complete. Results saved to {results_path}")
