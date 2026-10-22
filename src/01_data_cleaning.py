import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 1: DATA CLEANING & FEATURE EXTRACTION")
print("=" * 80)

# Paths
BASE_WORKSPACE = '/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research'
DATA_PATH = '/Users/gimhaniwijesuriya/Documents/Virtual Biopsy/new_files/physionet.org/files/mimiciv/3.1/hosp/'
PROCESSED_DATA_PATH = os.path.join(BASE_WORKSPACE, 'data/processed/')

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Load base data
print("📂 Loading core tables (patients, admissions, diagnoses)...")
patients   = pd.read_csv(os.path.join(DATA_PATH, 'patients.csv'))
admissions = pd.read_csv(os.path.join(DATA_PATH, 'admissions.csv'))
diagnoses  = pd.read_csv(os.path.join(DATA_PATH, 'diagnoses_icd.csv'))

patients.columns   = patients.columns.str.lower()
admissions.columns = admissions.columns.str.lower()
diagnoses.columns  = diagnoses.columns.str.lower()

# Admissions processing
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['los_days']  = (admissions['dischtime'] - admissions['admittime']).dt.days

# Identify CKD Patients (ICD codes starting with N18)
print("🔍 Identifying CKD patients...")
ckd_diag = diagnoses[diagnoses['icd_code'].str.startswith('N18', na=False)]
ckd_ids  = ckd_diag['subject_id'].unique()
print(f"   ✅ Total CKD patients found: {len(ckd_ids)}")

# Filter datasets
ckd_patients   = patients[patients['subject_id'].isin(ckd_ids)].copy()
ckd_admissions = admissions[admissions['subject_id'].isin(ckd_ids)].copy()
ckd_diagnoses  = diagnoses[diagnoses['subject_id'].isin(ckd_ids)].copy()

# Key Lab Items
key_labs = {
    50912: 'creatinine', 51006: 'bun',       50862: 'albumin',
    50971: 'potassium',  51222: 'hemoglobin', 50983: 'sodium',
    50893: 'calcium',    50970: 'phosphate',  50931: 'glucose',
    50882: 'bicarbonate',
}

# Process Lab Events in chunks (huge file)
print("🧪 Processing lab events (this may take a few minutes)...")
chunks = []
for chunk in pd.read_csv(os.path.join(DATA_PATH, 'labevents.csv'), chunksize=500000):
    filtered = chunk[
        (chunk['subject_id'].isin(ckd_ids)) &
        (chunk['itemid'].isin(key_labs.keys()))
    ]
    if len(filtered) > 0:
        chunks.append(filtered)
labevents = pd.concat(chunks, ignore_index=True)
print(f"   ✅ Lab measurements extracted: {len(labevents):,}")

# Feature Extraction
print("📊 Engineering clinical features...")
ckd_patients.rename(columns={'anchor_age': 'age'}, inplace=True)
ckd_patients['gender_encoded'] = (ckd_patients['gender'] == 'M').astype(int)
demo = ckd_patients[['subject_id', 'age', 'gender_encoded']].copy()

# Comorbidities
comorbidities = []
for patient_id in ckd_ids:
    pt   = ckd_diagnoses[ckd_diagnoses['subject_id'] == patient_id]
    icds = pt['icd_code'].str.upper().tolist()
    comorbidities.append({
        'subject_id':        patient_id,
        'num_diagnoses':     len(pt),
        'has_diabetes':      int(any(c.startswith(('E10','E11')) for c in icds)),
        'has_hypertension':  int(any(c.startswith('I10') for c in icds)),
        'has_heart_disease': int(any(c.startswith(('I2','I5')) for c in icds)),
        'has_anemia':        int(any(c.startswith(('D50','D64')) for c in icds)),
        'has_copd':          int(any(c.startswith('J4') for c in icds)),
        'has_cancer':        int(any(c.startswith('C') for c in icds)),
    })
comorb_df = pd.DataFrame(comorbidities)
comorb_df['comorbidity_score'] = comorb_df[[
    'has_diabetes','has_hypertension','has_heart_disease',
    'has_anemia','has_copd','has_cancer'
]].sum(axis=1)

# Admissions aggregation
adm_agg = ckd_admissions.groupby('subject_id').agg(
    num_admissions=('hadm_id','count'),
    avg_los=('los_days','mean'),
).reset_index()

# Lab aggregation
lab_records = []
for patient_id in ckd_ids:
    pt_labs = labevents[labevents['subject_id'] == patient_id]
    row     = {'subject_id': patient_id}
    for itemid, lab_name in key_labs.items():
        vals = pt_labs[pt_labs['itemid'] == itemid]['valuenum'].dropna()
        if len(vals) > 0:
            row[f'{lab_name}_mean']  = vals.mean()
            row[f'{lab_name}_min']   = vals.min()
            row[f'{lab_name}_max']   = vals.max()
            row[f'{lab_name}_std']   = vals.std()
            row[f'{lab_name}_count'] = len(vals)
        else:
            for s in ['mean','min','max','std','count']:
                row[f'{lab_name}_{s}'] = np.nan
    lab_records.append(row)

lab_df = pd.DataFrame(lab_records)

# Derived lab features
lab_df['bun_creatinine_ratio'] = lab_df['bun_mean'] / (lab_df['creatinine_mean'] + 0.01)
lab_df['creatinine_range']     = lab_df['creatinine_max'] - lab_df['creatinine_min']
lab_df['bun_range']            = lab_df['bun_max']   - lab_df['bun_min']
lab_df['hemoglobin_range']     = lab_df['hemoglobin_max'] - lab_df['hemoglobin_min']
lab_df['anemia_flag']          = (lab_df['hemoglobin_mean'] < 12).astype(int)
lab_df['severe_anemia']        = (lab_df['hemoglobin_mean'] < 8).astype(int)
lab_df['low_albumin']          = (lab_df['albumin_mean'] < 3.5).astype(int)
lab_df['hyperkalemia']         = (lab_df['potassium_mean'] > 5.0).astype(int)
lab_df['high_creatinine']      = (lab_df['creatinine_mean'] > 1.5).astype(int)
lab_df['very_high_creatinine'] = (lab_df['creatinine_mean'] > 3.0).astype(int)
lab_df['high_bun']             = (lab_df['bun_mean'] > 30).astype(int)
lab_df['very_high_bun']        = (lab_df['bun_mean'] > 60).astype(int)
lab_df['egfr_approx']          = 175 * (lab_df['creatinine_mean']**-1.154) * (demo['age'].values**-0.203)

# Merge all
df = demo \
    .merge(comorb_df, on='subject_id', how='left') \
    .merge(lab_df,    on='subject_id', how='left') \
    .merge(adm_agg,   on='subject_id', how='left')

# Target creation (Mild vs Severe) using KDIGO 2012 eGFR Standard
print("🎯 Generating Target Variable (target_binary) using KDIGO eGFR...")

def calculate_egfr_ckd_epi(row):
    try:
        cr = row['creatinine_mean']
        age = row['age']
        gender_encoded = row['gender_encoded'] # 1 is M, 0 is F

        if pd.isna(cr) or pd.isna(age):
            return np.nan

        if gender_encoded == 0: # Female
            kappa = 0.7
            alpha = -0.241
            gender_factor = 1.012
        else: # Male
            kappa = 0.9
            alpha = -0.302
            gender_factor = 1.0

        ratio = cr / kappa
        if ratio < 1:
            egfr = 142 * (ratio ** alpha) * (0.9938 ** age) * gender_factor
        else:
            egfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * gender_factor
        return egfr
    except:
        return np.nan

df['egfr_kdigo'] = df.apply(calculate_egfr_ckd_epi, axis=1)

# KDIGO Binary Target: Mild (G1-G3a, eGFR >= 45) = 0, Severe (G3b-G5, eGFR < 45) = 1
df['target_binary'] = df['egfr_kdigo'].apply(lambda e: 1 if e < 45 else 0)


n0 = (df['target_binary'] == 0).sum()
n1 = (df['target_binary'] == 1).sum()
print(f"   Mild (0):   {n0:,} ({n0/len(df)*100:.1f}%)")
print(f"   Severe (1): {n1:,} ({n1/len(df)*100:.1f}%)")

# Impute missing values
print("🧹 Imputing missing values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Save
output_path = os.path.join(PROCESSED_DATA_PATH, '01_cleaned_data.csv')
df.to_csv(output_path, index=False)
print(f"✅ Data cleaning complete. Saved to: {output_path}")
