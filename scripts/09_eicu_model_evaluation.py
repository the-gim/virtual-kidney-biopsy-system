import joblib
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
TEST_DATA_DIR = os.path.join(DATA_DIR, "test data")
FEATURES_IN = os.path.join(TEST_DATA_DIR, "eicu_extracted_features.csv")
RESULTS_OUT = os.path.join(TEST_DATA_DIR, "eicu_evaluation_results.txt")

def main():
    if not os.path.exists(FEATURES_IN):
        print(f"Error: {FEATURES_IN} not found. Run script 08 first.")
        return
        
    print(f"Loading {FEATURES_IN}...")
    df = pd.read_csv(FEATURES_IN, index_col=0)
    
    if 'severity' not in df.columns:
        print("Error: 'severity' column not found in test data.")
        return
        
    y = df['severity']
    X = df.drop(columns=['severity'])
    X = X.fillna(0)
    
    print(f"Test dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    model_names = [
        'artificial_neural_network',
        'support_vector_machine',
        'decision_tree'
    ]
    
    results_str = "eICU External Validation Results\n"
    results_str += "================================\n\n"
    
    for name in model_names:
        model_path = os.path.join(DATA_DIR, f"{name}.joblib")
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping.")
            continue
            
        print(f"\nEvaluating {name.replace('_', ' ').title()}...")
        model = joblib.load(model_path)
        
        y_pred = model.predict(X)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='macro', zero_division=0)
        rec = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        
        res = f"{name.replace('_', ' ').title()}:\n"
        res += f"  Accuracy:  {acc:.4f}\n"
        res += f"  Precision: {prec:.4f} (Macro Avg)\n"
        res += f"  Recall:    {rec:.4f} (Macro Avg)\n"
        res += f"  F1-Score:  {f1:.4f} (Macro Avg)\n\n"
        
        res += "Classification Report:\n"
        res += classification_report(y, y_pred, zero_division=0) + "\n\n"
        
        print(res)
        results_str += res
        
    with open(RESULTS_OUT, "w") as f:
        f.write(results_str)
        
    print(f"Saved results to {RESULTS_OUT}")

if __name__ == "__main__":
    main()
