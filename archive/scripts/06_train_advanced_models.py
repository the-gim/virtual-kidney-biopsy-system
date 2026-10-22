import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import joblib

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "advanced_model_results_cv.txt")
PLOT_OUT = os.path.join(DATA_DIR, "advanced_model_training_graphs.png")
CM_OUT = os.path.join(DATA_DIR, "advanced_model_confusion_matrices.png")

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from xgboost import XGBClassifier

class CNN1D(nn.Module):
    def __init__(self, input_dim=115, num_classes=4):
        super(CNN1D, self).__init__()
        # Input shape for CNN 1D: (batch, channels, seq_len)
        # Tabular data initially (batch, features). We reshape to (batch, 1, features)
        self.conv = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size
        conv_out_size = ((input_dim - 3 + 1) // 2)
        self.fc1 = nn.Linear(32 * conv_out_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is (batch, features). Convert to (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=115, num_classes=4):
        super(LSTMModel, self).__init__()
        # Input shape for LSTM: (batch, seq_len, features)
        # We treat tabular features as sequence of length 'input_dim' with 1 feature each
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is (batch, features). Convert to (batch, features, 1)
        x = x.unsqueeze(-1)
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return self.softmax(out)

def main():
    if not os.path.exists(FEATURES_IN):
        print(f"Error: {FEATURES_IN} not found.")
        return
        
    print(f"Loading {FEATURES_IN}...")
    df = pd.read_csv(FEATURES_IN, index_col=0)
    
    y = df['severity']
    X = df.drop(columns=['severity']).fillna(0)
    
    # Cast to float32 for PyTorch
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    
    print(f"Dataset shape: {X.shape}")
    
    input_dim = X.shape[1]
    
    cnn = Pipeline([
        ('scaler', StandardScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('cnn', NeuralNetClassifier(
            module=CNN1D,
            module__input_dim=input_dim,
            module__num_classes=4,
            max_epochs=20,
            lr=0.001,
            batch_size=256,
            optimizer=torch.optim.Adam,
            criterion=nn.CrossEntropyLoss,
            iterator_train__shuffle=True,
            train_split=None, # Disable internal validation so we can train fully, cross_validate handles validation
            verbose=0
        ))
    ])
    
    lstm = Pipeline([
        ('scaler', StandardScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('lstm', NeuralNetClassifier(
            module=LSTMModel,
            module__input_dim=input_dim,
            module__num_classes=4,
            max_epochs=20,
            lr=0.001,
            batch_size=256,
            optimizer=torch.optim.Adam,
            criterion=nn.CrossEntropyLoss,
            iterator_train__shuffle=True,
            train_split=None,
            verbose=0
        ))
    ])
    
    xgb = Pipeline([
        ('scaler', StandardScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('xgb', XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42, 
            n_jobs=1
        ))
    ])
    
    models = {
        '1D CNN': cnn,
        'LSTM': lstm,
        'XGBoost': xgb
    }
    
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results_str = "Advanced Model Evaluation Results (5-Fold CV)\n"
    results_str += "===============================================\n\n"
    
    train_f1s = []
    val_f1s = []
    model_names = []
    model_cv_preds = {}
    
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=True)
        
        print(f"Generating out-of-fold predictions for {name} confusion matrix...", flush=True)
        model_cv_preds[name] = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        
        val_acc = np.mean(scores['test_accuracy'])
        val_prec = np.mean(scores['test_precision_macro'])
        val_rec = np.mean(scores['test_recall_macro'])
        val_f1 = np.mean(scores['test_f1_macro'])
        
        train_acc = np.mean(scores['train_accuracy'])
        train_f1_score = np.mean(scores['train_f1_macro'])
        
        train_f1s.append(train_f1_score)
        val_f1s.append(val_f1)
        model_names.append(name)
        
        res = f"{name}:\n"
        res += f"  [Train] Accuracy: {train_acc:.4f} | F1-Score: {train_f1_score:.4f} (Macro)\n"
        res += f"  [Valid] Accuracy: {val_acc:.4f} | F1-Score: {val_f1:.4f} (Macro)\n"
        res += f"  [Valid] Precision: {val_prec:.4f} | Recall: {val_rec:.4f} (Macro)\n\n"
        
        print(res, flush=True)
        results_str += res
        
        print(f"Saving final {name} model...", flush=True)
        model.fit(X, y)
        model_filename = os.path.join(DATA_DIR, f"{name.replace(' ', '_').lower()}.joblib")
        joblib.dump(model, model_filename)
        
    with open(RESULTS_OUT, "w") as f:
        f.write(results_str)
    print(f"Saved numerical results to {RESULTS_OUT}")
    
    # 1. Generate Bar Chart
    print("Generating train vs validation comparison graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width/2, train_f1s, width, label='Training F1-Score')
    ax.bar(x + width/2, val_f1s, width, label='Validation (CV) F1-Score')
    ax.set_ylabel('F1-Score (Macro Average)')
    ax.set_title('Advanced Models Generalization: Train vs Validation')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in model_names])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300)
    print(f"Saved graphs to {PLOT_OUT}")
    
    # 2. Confusion Matrices
    print("Generating confusion matrices...")
    fig_cm, axes_cm = plt.subplots(1, len(models), figsize=(18, 5))
    
    for idx, name in enumerate(models.keys()):
        ConfusionMatrixDisplay.from_predictions(
            y, model_cv_preds[name], 
            ax=axes_cm[idx], 
            cmap='Blues',
            colorbar=False
        )
        axes_cm[idx].set_title(f'{name}\nConfusion Matrix')
        
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=300)
    print(f"Saved confusion matrices to {CM_OUT}")

if __name__ == "__main__":
    main()
