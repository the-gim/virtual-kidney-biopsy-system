import pandas as pd
import os
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data/test data"
MIMIC_FEATURES_CSV = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data/selected_features.csv"
COMBINED_PQ = os.path.join(DATA_DIR, "eicu_combined.parquet")
LABELS_CSV = os.path.join(DATA_DIR, "eicu_labels.csv")
FEATURES_OUT = os.path.join(DATA_DIR, "eicu_extracted_features.csv")

def main():
    print(f"Loading {COMBINED_PQ}...")
    if not os.path.exists(COMBINED_PQ):
        print(f"Error: {COMBINED_PQ} not found.")
        return
        
    df_combined = pd.read_parquet(COMBINED_PQ)
    
    # Extract features using tsfresh (must use exact same settings as MIMIC)
    print(f"Extracting features for {df_combined['id'].nunique()} eICU patients...")
    settings = MinimalFCParameters()
    extracted_features = extract_features(df_combined, 
                                          column_id="id", 
                                          column_sort="time", 
                                          column_kind="kind", 
                                          column_value="value",
                                          default_fc_parameters=settings,
                                          impute_function=impute,
                                          n_jobs=0)
    
    # Load expected MIMIC features schema
    mimic_df = pd.read_csv(MIMIC_FEATURES_CSV, index_col=0)
    expected_cols = [c for c in mimic_df.columns if c != 'severity']
    
    # Enforce schema
    print("Aligning eICU features to MIMIC-IV trained feature schema...")
    for col in expected_cols:
        if col not in extracted_features.columns:
            extracted_features[col] = 0.0  # Fill missing features with 0
            
    # Drop any extra features eICU might have generated
    extracted_features = extracted_features[expected_cols]
    
    # Merge with target labels
    labels = pd.read_csv(LABELS_CSV)
    labels = labels.set_index('patientunitstayid')
    
    final_dataset = extracted_features.join(labels[['severity']], how='inner')
    
    # Final imputation for any NaNs introduced during joining
    final_dataset = final_dataset.fillna(0)
    
    final_dataset.to_csv(FEATURES_OUT)
    print(f"Saved extracted eICU features to {FEATURES_OUT}")
    print(f"Final shape: {final_dataset.shape}")

if __name__ == "__main__":
    main()
