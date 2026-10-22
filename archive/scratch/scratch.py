import pandas as pd
import os
import numpy as np

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
df = pd.read_csv(FEATURES_IN, index_col=0)
print(df['severity'].value_counts())
