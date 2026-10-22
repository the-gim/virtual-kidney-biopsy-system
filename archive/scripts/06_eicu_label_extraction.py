import pandas as pd
import os
import re

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data/test data"
DIAGNOSIS_CSV = os.path.join(DATA_DIR, "diagnosis.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "eicu_labels.csv")

def get_severity(diagnosis_str):
    if pd.isna(diagnosis_str):
        return 0
    diag = str(diagnosis_str).lower()
    
    # 3: Severe (End stage, very advanced)
    if 'end stage' in diag or 'esrd' in diag:
        return 3
    # 2: Moderate (Chronic Kidney Disease)
    elif 'chronic kidney disease' in diag or 'ckd' in diag:
        return 2
    # 1: Mild (Acute, unspecified renal disorder)
    elif 'acute kidney' in diag or 'aki' in diag or 'renal failure' in diag or 'disorder of kidney' in diag:
        return 1
    # 0: None
    else:
        return 0

def main():
    print(f"Reading {DIAGNOSIS_CSV}...")
    try:
        df = pd.read_csv(DIAGNOSIS_CSV)
    except FileNotFoundError:
        print(f"Error: Could not find {DIAGNOSIS_CSV}")
        return
        
    df['severity'] = df['diagnosisstring'].apply(get_severity)
    
    # Map back to patientunitstayid and take max severity if multiple
    labels = df.groupby('patientunitstayid')['severity'].max().reset_index()
    
    labels.to_csv(OUTPUT_CSV, index=False)
    print(f"Found {len(labels)} patients in eICU with diagnostic records.")
    print("Severity distribution:")
    print(labels['severity'].value_counts())
    print(f"Saved eICU labels to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
