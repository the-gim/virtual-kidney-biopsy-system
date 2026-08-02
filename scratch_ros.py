import joblib
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import log_loss
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
import numpy as np

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
df = pd.read_csv(FEATURES_IN, index_col=0)
X = df.drop(columns=['severity']).fillna(0)
y = df['severity']

X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_sub_scaled = scaler.fit_transform(X_train_sub)
X_val_sub_scaled = scaler.transform(X_val_sub)

smoteenn = SMOTEENN(random_state=42)
X_train_sub_resampled, y_train_sub_resampled = smoteenn.fit_resample(X_train_sub_scaled, y_train_sub)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=1, warm_start=True, random_state=42)
classes = np.unique(y)

ros = RandomOverSampler(random_state=42)
X_val_sub_resampled, y_val_sub_resampled = ros.fit_resample(X_val_sub_scaled, y_val_sub)

train_losses = []
val_losses = []

for epoch in range(150):
    mlp.partial_fit(X_train_sub_resampled, y_train_sub_resampled, classes=classes)
    
    train_probs = mlp.predict_proba(X_train_sub_resampled)
    val_probs = mlp.predict_proba(X_val_sub_resampled)
    
    train_loss = log_loss(y_train_sub_resampled, train_probs)
    val_loss = log_loss(y_val_sub_resampled, val_probs)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

print("Train:", train_losses[:5], "...", train_losses[-5:])
print("Val:", val_losses[:5], "...", val_losses[-5:])
