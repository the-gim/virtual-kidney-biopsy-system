import pandas as pd
import os
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
df = pd.read_csv(FEATURES_IN, index_col=0)
X = df.drop(columns=['severity']).fillna(0)
y = df['severity']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Original classes:", np.bincount(y))
try:
    smoteenn = SMOTEENN(random_state=42)
    X_res, y_res = smoteenn.fit_resample(X_scaled, y)
    print("Resampled classes:", np.bincount(y_res))
except Exception as e:
    print("Error:", e)
