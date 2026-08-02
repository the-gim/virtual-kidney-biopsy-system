import csv
import re
import os
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
DISCHARGE_CSV = os.path.join(DATA_DIR, "discharge.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "biopsy_labels.csv")

# Regular expressions to find severity of scarring/fibrosis near the word biopsy
# We use a window approach: find 'biopsy' and look for severity keywords in the text
biopsy_pattern = re.compile(r'biopsy', re.IGNORECASE)
severe_pattern = re.compile(r'(severe|advanced|extensive|significant)\s+(scarring|fibrosis|sclerosis|damage)', re.IGNORECASE)
moderate_pattern = re.compile(r'(moderate)\s+(scarring|fibrosis|sclerosis|damage)', re.IGNORECASE)
mild_pattern = re.compile(r'(mild|minimal|slight|focal)\s+(scarring|fibrosis|sclerosis|damage)', re.IGNORECASE)
none_pattern = re.compile(r'(no|without)\s+(scarring|fibrosis|sclerosis|damage|evidence of)', re.IGNORECASE)

def extract_severity(text):
    if not isinstance(text, str):
        return None
        
    # Only process if 'biopsy' or 'renal' or 'kidney' is mentioned
    if not (biopsy_pattern.search(text) or 'renal' in text.lower() or 'kidney' in text.lower()):
        return None

    # Check for patterns
    if severe_pattern.search(text):
        return 3 # Severe
    elif moderate_pattern.search(text):
        return 2 # Moderate
    elif mild_pattern.search(text):
        return 1 # Mild
    elif none_pattern.search(text):
        return 0 # None
    return None

def main():
    print(f"Reading from {DISCHARGE_CSV}...")
    results = []
    
    # We use csv.reader to stream the file line by line to avoid memory issues
    with open(DISCHARGE_CSV, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Processing notes"):
            text = row.get('text', '')
            subject_id = row.get('subject_id')
            hadm_id = row.get('hadm_id')
            
            severity = extract_severity(text)
            if severity is not None:
                results.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'severity': severity
                })
                
    df_results = pd.DataFrame(results)
    # If a patient has multiple notes, take the max severity (worst case)
    if not df_results.empty:
        df_results = df_results.groupby('subject_id').agg({'severity': 'max', 'hadm_id': 'last'}).reset_index()
        df_results.to_csv(OUTPUT_CSV, index=False)
        print(f"Found {len(df_results)} patients with biopsy scarring records.")
        print(df_results['severity'].value_counts())
    else:
        print("No biopsy records with scarring severity found. We might need to adjust our regex.")
        df_results.to_csv(OUTPUT_CSV, index=False)
        
    print(f"Saved extracted labels to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
