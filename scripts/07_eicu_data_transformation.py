import pandas as pd
import os

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data/test data"
LAB_CSV = os.path.join(DATA_DIR, "lab.csv")
VITAL_CSV = os.path.join(DATA_DIR, "vitalPeriodic.csv")
LABELS_CSV = os.path.join(DATA_DIR, "eicu_labels.csv")
OUT_PQ = os.path.join(DATA_DIR, "eicu_combined.parquet")

def main():
    print("Loading labels...")
    if not os.path.exists(LABELS_CSV):
        print(f"Error: {LABELS_CSV} not found. Run script 06 first.")
        return
    labels = pd.read_csv(LABELS_CSV)
    valid_ids = labels['patientunitstayid'].unique()
    
    print(f"Loading {LAB_CSV}...")
    df_lab = pd.read_csv(LAB_CSV)
    df_lab = df_lab[df_lab['patientunitstayid'].isin(valid_ids)]
    
    # Map eICU labnames to MIMIC itemids
    lab_map = {
        'creatinine': '50912',
        'BUN': '51006',
        'albumin': '50862',
        'bicarbonate': '50882',
        'sodium': '50983',
        'potassium': '50971',
        'urinalysis color': '51081',
        'urinalysis specific gravity': '51100'
    }
    
    # Filter and rename labs
    df_lab['labname'] = df_lab['labname'].str.lower()
    df_lab_filtered = df_lab[df_lab['labname'].isin([k.lower() for k in lab_map.keys()])].copy()
    
    # Standardize names to map
    df_lab_filtered['kind'] = df_lab_filtered['labname'].apply(
        lambda x: lab_map.get(next((k for k in lab_map if k.lower() == x), None))
    )
    df_lab_final = df_lab_filtered[['patientunitstayid', 'labresultoffset', 'kind', 'labresult']].rename(
        columns={'patientunitstayid': 'id', 'labresultoffset': 'time', 'labresult': 'value'}
    )
    
    print(f"Loading {VITAL_CSV}...")
    df_vital = pd.read_csv(VITAL_CSV)
    df_vital = df_vital[df_vital['patientunitstayid'].isin(valid_ids)]
    
    # Map eICU vital columns to MIMIC itemids
    vital_map = {
        'heartrate': '220045',
        'systemicsystolic': '220179',
        'systemicdiastolic': '220180',
        'systemicmean': '220181'
    }
    
    # Melt vitals from wide to long format
    vital_cols = ['patientunitstayid', 'observationoffset'] + list(vital_map.keys())
    # Only keep columns that actually exist in the file
    existing_cols = [c for c in vital_cols if c in df_vital.columns]
    df_vital_sub = df_vital[existing_cols]
    
    df_vital_melt = df_vital_sub.melt(id_vars=['patientunitstayid', 'observationoffset'], 
                                      var_name='vital_name', 
                                      value_name='value')
                                      
    df_vital_melt = df_vital_melt.dropna(subset=['value'])
    df_vital_melt['kind'] = df_vital_melt['vital_name'].map(vital_map)
    
    df_vital_final = df_vital_melt[['patientunitstayid', 'observationoffset', 'kind', 'value']].rename(
        columns={'patientunitstayid': 'id', 'observationoffset': 'time'}
    )
    
    print("Combining datasets...")
    df_combined = pd.concat([df_lab_final, df_vital_final], ignore_index=True)
    
    # Drop rows without a mapped kind or value
    df_combined = df_combined.dropna(subset=['kind', 'value'])
    
    # Ensure time is sortable (eICU uses minute offsets instead of timestamps, which is fine for tsfresh if numeric)
    df_combined['time'] = pd.to_numeric(df_combined['time'])
    
    df_combined.to_parquet(OUT_PQ)
    print(f"Saved combined long-format data to {OUT_PQ}")
    print(f"Total rows: {len(df_combined)}")

if __name__ == "__main__":
    main()
