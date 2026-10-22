import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
import numpy as np

df = pd.read_csv("/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data/selected_features.csv", index_col=0)
X = df.drop(columns=['severity']).fillna(0)
y = df['severity']

X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_sub_scaled = scaler.fit_transform(X_train_sub)
X_val_sub_scaled = scaler.transform(X_val_sub)

smoteenn = SMOTEENN(random_state=42)
X_train_sub_resampled, y_train_sub_resampled = smoteenn.fit_resample(X_train_sub_scaled, y_train_sub)

mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=1, warm_start=True, random_state=42)
classes = np.unique(y)

train_losses = []
val_losses = []

for epoch in range(150):
    mlp.partial_fit(X_train_sub_resampled, y_train_sub_resampled, classes=classes)
    train_probs = mlp.predict_proba(X_train_sub_scaled)
    val_probs = mlp.predict_proba(X_val_sub_scaled)
    train_losses.append(log_loss(y_train_sub, train_probs))
    val_losses.append(log_loss(y_val_sub, val_probs))

print("Train (imbalanced):", train_losses[:5], "...", train_losses[-5:])
print("Val (imbalanced):", val_losses[:5], "...", val_losses[-5:])
