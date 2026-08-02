import pandas as pd
import numpy as np
import os
from tqdm import tqdm

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
CHARTEVENTS_PQ = os.path.join(DATA_DIR, "filtered_chartevents.parquet")
LABEVENTS_PQ = os.path.join(DATA_DIR, "filtered_labevents.parquet")
LABELS_CSV = os.path.join(DATA_DIR, "biopsy_labels.csv")

OUT_X = os.path.join(DATA_DIR, "timeseries_X.npy")
OUT_y = os.path.join(DATA_DIR, "timeseries_y.npy")

def main():
    print("Loading parquet files...")
    df_chart = pd.read_parquet(CHARTEVENTS_PQ)[['subject_id', 'charttime', 'itemid', 'valuenum']]
    df_lab = pd.read_parquet(LABEVENTS_PQ)[['subject_id', 'charttime', 'itemid', 'valuenum']]
    
    df_combined = pd.concat([df_chart, df_lab], ignore_index=True)
    df_combined['charttime'] = pd.to_datetime(df_combined['charttime'])
    
    print("Loading labels...")
    labels_df = pd.read_csv(LABELS_CSV)
    # Only keep patients that have labels
    valid_subjects = set(labels_df['subject_id'])
    df_combined = df_combined[df_combined['subject_id'].isin(valid_subjects)]
    
    # Calculate time relative to first measurement per patient
    print("Calculating relative time...")
    min_time = df_combined.groupby('subject_id')['charttime'].transform('min')
    df_combined['time_hours'] = (df_combined['charttime'] - min_time).dt.total_seconds() / 3600.0
    
    # Bin into 4-hour intervals
    # e.g., 0-4 hours -> bin 0, 4-8 hours -> bin 1
    df_combined['time_bin'] = (df_combined['time_hours'] // 4).astype(int)
    
    # We only care about the first 48 hours (12 bins: 0 to 11)
    df_combined = df_combined[df_combined['time_bin'] < 12]
    
    # Aggregate (mean) if multiple measurements of same itemid exist in the same bin
    print("Aggregating into 4-hour bins...")
    df_agg = df_combined.groupby(['subject_id', 'time_bin', 'itemid'])['valuenum'].mean().reset_index()
    
    # Pivot so itemids are columns
    print("Pivoting data...")
    df_pivot = df_agg.pivot(index=['subject_id', 'time_bin'], columns='itemid', values='valuenum')
    
    # Get all unique itemids to ensure consistent feature dimension
    features = sorted(df_combined['itemid'].unique())
    num_features = len(features)
    
    # Build final 3D tensor
    print(f"Building 3D tensor of shape (num_patients, 12, {num_features})...")
    
    subject_ids = sorted(df_agg['subject_id'].unique())
    num_patients = len(subject_ids)
    
    X = np.zeros((num_patients, 12, num_features), dtype=np.float32)
    y = np.zeros(num_patients, dtype=np.int64)
    
    label_dict = dict(zip(labels_df['subject_id'], labels_df['severity']))
    
    for i, subject_id in enumerate(tqdm(subject_ids, desc="Processing patients")):
        # Assign label
        y[i] = label_dict[subject_id]
        
        # Get patient data
        try:
            pat_data = df_pivot.loc[subject_id]
            
            # Place values into corresponding time bin
            for bin_idx in pat_data.index:
                for f_idx, feature_name in enumerate(features):
                    if feature_name in pat_data.columns and not pd.isna(pat_data.at[bin_idx, feature_name]):
                        X[i, bin_idx, f_idx] = pat_data.at[bin_idx, feature_name]
                        
            # Forward fill the features within the 12 steps if needed (optional, we can just leave as 0 or forward fill)
            # A simple forward fill across time steps for this patient
            for f_idx in range(num_features):
                last_val = 0.0 # Default if no value yet
                for step in range(12):
                    if X[i, step, f_idx] != 0.0:
                        last_val = X[i, step, f_idx]
                    else:
                        X[i, step, f_idx] = last_val
                        
        except KeyError:
            # Patient has no data in the first 48 hours
            pass

    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    np.save(OUT_X, X)
    np.save(OUT_y, y)
    print(f"Saved to {OUT_X} and {OUT_y}")

if __name__ == "__main__":
    main()
