import pandas as pd
import os
from tsfresh.feature_selection import select_features
from sklearn.feature_selection import mutual_info_classif
import numpy as np

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "extracted_features.csv")
FEATURES_OUT = os.path.join(DATA_DIR, "selected_features.csv")
REPORT_OUT = os.path.join(DATA_DIR, "feature_analysis_report.txt")

def main():
    if not os.path.exists(FEATURES_IN):
        print(f"Error: {FEATURES_IN} not found. Run script 03 first.")
        return
        
    print(f"Loading {FEATURES_IN}...")
    df = pd.read_csv(FEATURES_IN, index_col=0)
    
    # Separate features and target
    y = df['severity']
    X = df.drop(columns=['severity'])
    
    print(f"Original shape: {X.shape}")
    
    # 1. Remove features with zero variance or all NaNs
    X = X.dropna(axis=1, how='all')
    
    # 2. Use tsfresh's built-in feature selection (hypothesis testing)
    print("Selecting relevant features using tsfresh...")
    X_selected = select_features(X, y)
    print(f"Shape after tsfresh selection: {X_selected.shape}")
    
    # 3. Calculate Mutual Information to find the top 20 most important features
    print("Calculating Mutual Information scores...")
    mi_scores = mutual_info_classif(X_selected.fillna(0), y)
    mi_series = pd.Series(mi_scores, index=X_selected.columns)
    mi_series = mi_series.sort_values(ascending=False)
    
    print("\nTop 20 Most Predictive Features:")
    print(mi_series.head(20))
    
    # Dictionary to map itemids to clinical names
    itemid_map = {
        '50912': 'Creatinine', '51006': 'BUN', '50862': 'Albumin', '50882': 'Bicarbonate', 
        '50983': 'Sodium', '50971': 'Potassium', '51081': 'Urine Color', '51100': 'Urine Specific Gravity',
        '220045': 'Heart Rate', '220179': 'NBP Systolic', '220180': 'NBP Diastolic', '220181': 'NBP Mean', 
        '220615': 'Creatinine (Serum)', '225624': 'BUN (Chart)', '227442': 'Potassium (Serum)', '227443': 'Bicarbonate (Serum)'
    }
    
    # Save the report
    with open(REPORT_OUT, "w") as f:
        f.write("Feature Analysis Report\n")
        f.write("========================\n")
        f.write(f"Original Features: {X.shape[1]}\n")
        f.write(f"Selected Features: {X_selected.shape[1]}\n\n")
        f.write("Top 20 Features by Mutual Information:\n")
        for idx, val in mi_series.head(20).items():
            # Replace itemid with clinical name
            feature_name = idx
            for item_id, name in itemid_map.items():
                if idx.startswith(item_id):
                    feature_name = idx.replace(item_id, f"{name} ({item_id})")
                    break
            f.write(f"{feature_name}: {val:.4f}\n")
            
    # Save the selected features
    final_df = X_selected.copy()
    final_df['severity'] = y
    final_df.to_csv(FEATURES_OUT)
    
    print(f"\nSaved selected features to {FEATURES_OUT}")
    print(f"Saved feature analysis report to {REPORT_OUT}")

if __name__ == "__main__":
    main()
