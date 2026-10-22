import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Use importlib to import from scripts starting with numbers if needed, but python 3.11 allows it if we rename it. Wait, 11_train_timeseries_models is not a valid python identifier for direct import.
# I will use importlib.
import importlib
module_11 = importlib.import_module("11_train_timeseries_models")
CNN1D = module_11.CNN1D
LSTMModel = module_11.LSTMModel

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
IN_X = os.path.join(DATA_DIR, "timeseries_X.npy")
IN_y = os.path.join(DATA_DIR, "timeseries_y.npy")
RESULTS_OUT = os.path.join(DATA_DIR, "deep_learning_email_report.txt")
CM_OUT = os.path.join(DATA_DIR, "final_dl_confusion_matrices.png")

def main():
    print(f"Loading {IN_X} and {IN_y}...")
    X = np.load(IN_X)
    y = np.load(IN_y)
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    
    # Scale data
    batch, seq_len, features = X.shape
    X_flat = X.reshape(-1, features)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(batch, seq_len, features)
    X = X.astype(np.float32)
    
    # Binarize labels for AUPRC
    y_bin = label_binarize(y, classes=classes)
    
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
    
    models = {
        '1D CNN (Timeseries)': cnn,
        'LSTM (Timeseries)': lstm
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    report = "Deep Learning Methods Results (5-Fold Cross Validation)\n"
    report += "Dataset: 45,435 eICU patients (Class Imbalance: 0: ~98%, 1-3: ~2%)\n"
    report += "Data Format: Raw 3D Timeseries (12 time steps of 4 hours each, 16 features)\n"
    report += "=================================================================================\n\n"
    
    fig_cm, axes_cm = plt.subplots(1, len(models), figsize=(12, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"Evaluating {name} using 5-Fold CV (Generating probabilities)...", flush=True)
        
        # Get out-of-fold probability predictions
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba', n_jobs=1)
        
        # Get hard predictions from probabilities
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        
        # AUROC (macro, one-vs-rest)
        auroc = roc_auc_score(y, y_pred_proba, multi_class='ovr', average='macro')
        
        # AUPRC (macro)
        auprc = average_precision_score(y_bin, y_pred_proba, average='macro')
        
        # Format for email
        report += f"Model: {name}\n"
        report += f"----------------------------------------\n"
        report += f"Accuracy (Overall): {acc:.4f}\n"
        report += f"Precision (Macro):  {prec:.4f}\n"
        report += f"Recall (Macro):     {rec:.4f}\n"
        report += f"F1-Score (Macro):   {f1:.4f}\n"
        report += f"AUROC (Macro OVR):  {auroc:.4f}\n"
        report += f"AUPRC (Macro):      {auprc:.4f}\n"
        report += "\n"
        
        print(f"Finished {name}.")
        
        # Plot Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred, 
            ax=axes_cm[idx], 
            cmap='Blues',
            colorbar=False
        )
        axes_cm[idx].set_title(f'{name}\nConfusion Matrix')
    
    # Save Report
    with open(RESULTS_OUT, "w") as f:
        f.write(report)
    print(f"\nSaved final email report to {RESULTS_OUT}")
    
    # Save CM image
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=300)
    print(f"Saved confusion matrices image to {CM_OUT}")

if __name__ == "__main__":
    main()
