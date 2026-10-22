import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib
module_11 = importlib.import_module("11_train_timeseries_models")
CNN1D = module_11.CNN1D
LSTMModel = module_11.LSTMModel

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
IN_X = os.path.join(DATA_DIR, "timeseries_X.npy")
IN_y = os.path.join(DATA_DIR, "timeseries_y.npy")

ROC_OUT = os.path.join(DATA_DIR, "final_roc_curves.png")
PRC_OUT = os.path.join(DATA_DIR, "final_prc_curves.png")
CM_OUT = os.path.join(DATA_DIR, "final_all_confusion_matrices.png")

def plot_macro_roc(y_bin, y_pred_proba, ax, name, color):
    n_classes = y_bin.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    macro_roc_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color=color, lw=2, label=f'{name} (Macro AUROC = {macro_roc_auc:.3f})')

def plot_macro_prc(y_bin, y_pred_proba, ax, name, color):
    n_classes = y_bin.shape[1]
    
    # Calculate Macro AUPRC
    macro_auprc = average_precision_score(y_bin, y_pred_proba, average='macro')
    
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_pred_proba[:, i])
        # PR curves need to be interpolated backwards for recall
        mean_precision += np.interp(mean_recall, recall[::-1], precision[::-1])
    mean_precision /= n_classes
    
    ax.plot(mean_recall, mean_precision, color=color, lw=2, label=f'{name} (Macro AUPRC = {macro_auprc:.3f})')


def main():
    print("Loading Tabular Data for XGBoost...")
    df = pd.read_csv(FEATURES_IN, index_col=0)
    y_tab = df['severity'].astype(np.int64).values
    X_tab = df.drop(columns=['severity']).fillna(0).astype(np.float32).values
    
    print("Loading 3D Data for CNN/LSTM...")
    X_3d = np.load(IN_X)
    y_3d = np.load(IN_y)
    
    classes = np.unique(y_3d)
    y_bin = label_binarize(y_3d, classes=classes)
    
    weights = compute_class_weight('balanced', classes=classes, y=y_3d)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    batch, seq_len, features = X_3d.shape
    X_flat = X_3d.reshape(-1, features)
    scaler_3d = StandardScaler()
    X_flat = scaler_3d.fit_transform(X_flat)
    X_3d_scaled = X_flat.reshape(batch, seq_len, features).astype(np.float32)
    
    # Define XGBoost Pipeline
    xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=1))
    ])
    
    # Define CNN Pipeline
    cnn = NeuralNetClassifier(
        module=CNN1D,
        module__num_features=features,
        module__seq_len=seq_len,
        module__num_classes=len(classes),
        max_epochs=20,
        lr=0.001,
        batch_size=256,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        criterion__weight=class_weights,
        iterator_train__shuffle=True,
        train_split=None,
        verbose=0
    )
    
    # Define LSTM Pipeline
    lstm = NeuralNetClassifier(
        module=LSTMModel,
        module__num_features=features,
        module__num_classes=len(classes),
        max_epochs=20,
        lr=0.001,
        batch_size=256,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        criterion__weight=class_weights,
        iterator_train__shuffle=True,
        train_split=None,
        verbose=0
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = [
        ('XGBoost (Tabular)', xgb, X_tab, y_tab, 'orange'),
        ('1D CNN (Timeseries)', cnn, X_3d_scaled, y_3d, 'blue'),
        ('LSTM (Timeseries)', lstm, X_3d_scaled, y_3d, 'green')
    ]
    
    print("Beginning 5-Fold Cross Validation for all models...")
    
    # Prepare plots
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_prc, ax_prc = plt.subplots(figsize=(8, 6))
    
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    
    # Baseline for ROC
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Chance')
    
    for idx, (name, model, X_data, y_data, color) in enumerate(models):
        print(f"[{name}] Generating cross-validated predictions...", flush=True)
        y_pred_proba = cross_val_predict(model, X_data, y_data, cv=cv, method='predict_proba', n_jobs=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Plot ROC
        plot_macro_roc(y_bin, y_pred_proba, ax_roc, name, color)
        
        # Plot PRC
        plot_macro_prc(y_bin, y_pred_proba, ax_prc, name, color)
        
        # Plot CM
        ConfusionMatrixDisplay.from_predictions(
            y_data, y_pred, 
            ax=axes_cm[idx], 
            cmap='Blues',
            colorbar=False
        )
        axes_cm[idx].set_title(f'{name}\nConfusion Matrix')
        
    # Format ROC plot
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (Macro-Average)')
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(ROC_OUT, dpi=300)
    print(f"Saved ROC Curves to {ROC_OUT}")
    
    # Format PRC plot
    ax_prc.set_xlim([0.0, 1.0])
    ax_prc.set_ylim([0.0, 1.05])
    ax_prc.set_xlabel('Recall')
    ax_prc.set_ylabel('Precision')
    ax_prc.set_title('Precision-Recall Curve (Macro-Average)')
    ax_prc.legend(loc="upper right")
    fig_prc.tight_layout()
    fig_prc.savefig(PRC_OUT, dpi=300)
    print(f"Saved PRC Curves to {PRC_OUT}")
    
    # Format CM plot
    fig_cm.tight_layout()
    fig_cm.savefig(CM_OUT, dpi=300)
    print(f"Saved Confusion Matrices to {CM_OUT}")

if __name__ == "__main__":
    main()
