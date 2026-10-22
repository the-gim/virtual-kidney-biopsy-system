import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import joblib
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features_shap.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "advanced_model_results_shap_cv.txt")
PLOT_OUT = os.path.join(DATA_DIR, "advanced_model_training_graphs_shap.png")
CM_OUT = os.path.join(DATA_DIR, "advanced_model_confusion_matrices_shap.png")

class CNN1D(nn.Module):
    def __init__(self, input_dim=115, num_classes=4):
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
        return self.softmax(x)

class LSTMModel(nn.Module):
    def __init__(self, input_dim=115, num_classes=4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, (hn, cn) = self.lstm(x)
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
    
    y = df['severity'].values
    X = df.drop(columns=['severity']).fillna(0).values
    
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    
    print(f"Dataset shape: {X.shape}")
    input_dim = X.shape[1]
    
    # Compute Class Weights
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Computed Class Weights: {weights}")
    
    # Pre-compute sample weights for XGBoost
    sample_weights = np.zeros(len(y), dtype=np.float32)
    for c in classes:
        sample_weights[y == c] = weights[int(c)]
    
    cnn = NeuralNetClassifier(
        module=CNN1D,
        module__input_dim=input_dim,
        module__num_classes=4,
        max_epochs=20,
        lr=0.001,
        batch_size=256,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        criterion__weight=class_weights_tensor,
        iterator_train__shuffle=True,
        train_split=None,
        verbose=0
    )
    
    lstm = NeuralNetClassifier(
        module=LSTMModel,
        module__input_dim=input_dim,
        module__num_classes=4,
        max_epochs=20,
        lr=0.001,
        batch_size=256,
        optimizer=torch.optim.Adam,
        criterion=nn.CrossEntropyLoss,
        criterion__weight=class_weights_tensor,
        iterator_train__shuffle=True,
        train_split=None,
        verbose=0
    )
    
    xgb = XGBClassifier(
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        random_state=42, 
        n_jobs=1
    )
    
    models = {
        '1D CNN (CW)': cnn,
        'LSTM (CW)': lstm,
        'XGBoost (CW)': xgb
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results_str = "Advanced Model Evaluation Results (Class Weights, 5-Fold CV)\n"
    results_str += "============================================================\n\n"
    
    train_f1s = []
    val_f1s = []
    model_names = []
    
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, model) in enumerate(models.items()):
        print(f"Evaluating {name} using custom 5-Fold CV...", flush=True)
        
        train_accs_fold = []
        train_f1s_fold = []
        val_accs_fold = []
        val_f1s_fold = []
        val_prec_fold = []
        val_rec_fold = []
        
        y_pred_all = np.zeros_like(y)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            if "XGBoost" in name:
                w_train = sample_weights[train_idx]
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)
                
            y_train_pred = model.predict(X_train)
            train_accs_fold.append(accuracy_score(y_train, y_train_pred))
            train_f1s_fold.append(f1_score(y_train, y_train_pred, average='macro', zero_division=0))
            
            y_val_pred = model.predict(X_val)
            y_pred_all[val_idx] = y_val_pred
            
            val_accs_fold.append(accuracy_score(y_val, y_val_pred))
            val_f1s_fold.append(f1_score(y_val, y_val_pred, average='macro', zero_division=0))
            val_prec_fold.append(precision_score(y_val, y_val_pred, average='macro', zero_division=0))
            val_rec_fold.append(recall_score(y_val, y_val_pred, average='macro', zero_division=0))
            
        val_acc = np.mean(val_accs_fold)
        val_prec = np.mean(val_prec_fold)
        val_rec = np.mean(val_rec_fold)
        val_f1 = np.mean(val_f1s_fold)
        
        train_acc = np.mean(train_accs_fold)
        train_f1_score = np.mean(train_f1s_fold)
        
        train_f1s.append(train_f1_score)
        val_f1s.append(val_f1)
        model_names.append(name)
        
        res = f"{name}:\n"
        res += f"  [Train] Accuracy: {train_acc:.4f} | F1-Score: {train_f1_score:.4f} (Macro)\n"
        res += f"  [Valid] Accuracy: {val_acc:.4f} | F1-Score: {val_f1:.4f} (Macro)\n"
        res += f"  [Valid] Precision: {val_prec:.4f} | Recall: {val_rec:.4f} (Macro)\n\n"
        
        print(res, flush=True)
        results_str += res
        
        # Plot Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred_all, 
            ax=axes_cm[idx], 
            cmap='Blues',
            colorbar=False
        )
        axes_cm[idx].set_title(f'{name}\nConfusion Matrix')
        
        print(f"Saving final {name} model on all data...", flush=True)
        # Scale whole dataset for final fit
        scaler_full = StandardScaler()
        X_full = scaler_full.fit_transform(X)
        if "XGBoost" in name:
            model.fit(X_full, y, sample_weight=sample_weights)
        else:
            model.fit(X_full, y)
        model_filename = os.path.join(DATA_DIR, f"{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.joblib")
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
    ax.set_title('Models with Class Weights: Train vs Validation')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in model_names])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300)
    print(f"Saved graphs to {PLOT_OUT}")
    
    # 2. Confusion Matrices
    plt.tight_layout()
    fig_cm.savefig(CM_OUT, dpi=300)
    print(f"Saved confusion matrices to {CM_OUT}")

if __name__ == "__main__":
    main()
