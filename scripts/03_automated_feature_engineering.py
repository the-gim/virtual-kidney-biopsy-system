import pandas as pd
import os
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
LABELS_CSV = os.path.join(DATA_DIR, "biopsy_labels.csv")
CHARTEVENTS_PQ = os.path.join(DATA_DIR, "filtered_chartevents.parquet")
LABEVENTS_PQ = os.path.join(DATA_DIR, "filtered_labevents.parquet")
FEATURES_OUT = os.path.join(DATA_DIR, "extracted_features.csv")

def main():
    print("Loading filtered parquet files...")
    df_chart = pd.read_parquet(CHARTEVENTS_PQ)
    df_lab = pd.read_parquet(LABEVENTS_PQ)
    
    # We only need: subject_id, charttime, itemid, valuenum
    df_chart = df_chart[['subject_id', 'charttime', 'itemid', 'valuenum']].dropna(subset=['valuenum'])
    df_lab = df_lab[['subject_id', 'charttime', 'itemid', 'valuenum']].dropna(subset=['valuenum'])
    
    # Combine them
    df_combined = pd.concat([df_chart, df_lab], ignore_index=True)
    df_combined['charttime'] = pd.to_datetime(df_combined['charttime'])
    
    # Format for tsfresh: we need a 'kind' column (itemid) and 'value' column (valuenum)
    # tsfresh long format: id, time, kind, value
    df_tsfresh = df_combined.rename(columns={'subject_id': 'id', 'charttime': 'time', 'itemid': 'kind', 'valuenum': 'value'})
    
    # Due to size, we will use minimal efficient feature extraction to avoid Out of Memory (OOM) errors
    from tsfresh.feature_extraction import MinimalFCParameters
    settings = MinimalFCParameters()
    
    print(f"Extracting features for {df_tsfresh['id'].nunique()} patients using tsfresh. This may take a while...")
    # Extract features (n_jobs=0 disables multiprocessing to save memory)
    extracted_features = extract_features(df_tsfresh, 
                                          column_id="id", 
                                          column_sort="time", 
                                          column_kind="kind", 
                                          column_value="value",
                                          default_fc_parameters=settings,
                                          impute_function=impute,
                                          n_jobs=0)
    
    # Merge with target labels
    labels = pd.read_csv(LABELS_CSV)
    labels = labels.set_index('subject_id')
    
    final_dataset = extracted_features.join(labels[['severity']], how='inner')
    
    # Save the dataset
    final_dataset.to_csv(FEATURES_OUT)
    print(f"Saved extracted features to {FEATURES_OUT}")
    print(f"Shape of final dataset: {final_dataset.shape}")

if __name__ == "__main__":
    main()
