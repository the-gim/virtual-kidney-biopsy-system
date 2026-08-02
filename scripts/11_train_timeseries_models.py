import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import joblib
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
IN_X = os.path.join(DATA_DIR, "timeseries_X.npy")
IN_y = os.path.join(DATA_DIR, "timeseries_y.npy")
RESULTS_OUT = os.path.join(DATA_DIR, "timeseries_model_results_cv.txt")
PLOT_OUT = os.path.join(DATA_DIR, "timeseries_model_training_graphs.png")
CM_OUT = os.path.join(DATA_DIR, "timeseries_model_confusion_matrices.png")

class CNN1D(nn.Module):
    def __init__(self, num_features=16, seq_len=12, num_classes=4):
        super(CNN1D, self).__init__()
        # Input shape expected by PyTorch Conv1d: (batch, channels, seq_len)
        # channels = num_features (16)
        self.conv = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size
        # L_out = floor((L_in - kernel_size) / stride) + 1
        # L_in = 12, kernel=3, stride=1 -> L_out = 10
        # pool size 2 -> 10 / 2 = 5
        conv_out_size = 5
        self.fc1 = nn.Linear(32 * conv_out_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is (batch, seq_len, features) = (batch, 12, 16)
        # We need to permute it for CNN to (batch, features, seq_len) -> (batch, 16, 12)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class LSTMModel(nn.Module):
    def __init__(self, num_features=16, num_classes=4):
        super(LSTMModel, self).__init__()
        # Input shape for LSTM: (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=64, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is (batch, 12, 16)
        out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :] 
        out = self.dropout(out)
        out = self.fc(out)
        return self.softmax(out)

def main():
    print(f"Loading {IN_X} and {IN_y}...")
    X = np.load(IN_X)
    y = np.load(IN_y)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Calculate Class Weights to handle the huge imbalance (instead of SMOTE)
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    print(f"Computed Class Weights: {weights}")
    
    # Scale data: standard scaler but applied to the 3D tensor
    # Flatten -> Scale -> Reshape
    batch, seq_len, features = X.shape
    X_flat = X.reshape(-1, features)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(batch, seq_len, features)
    X = X.astype(np.float32)
    
    # Define Models
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
    
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results_str = "Timeseries Deep Learning Models Evaluation Results (5-Fold CV)\n"
    results_str += "===============================================================\n\n"
    
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
    ax.set_title('Timeseries Deep Learning Models: Train vs Validation')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in model_names])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300)
    print(f"Saved graphs to {PLOT_OUT}")
    
    # 2. Confusion Matrices
    print("Generating confusion matrices...")
    fig_cm, axes_cm = plt.subplots(1, len(models), figsize=(12, 5))
    
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
