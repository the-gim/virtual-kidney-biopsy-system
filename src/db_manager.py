import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'virtual_biopsy_system.db')
DB_PATH = os.path.abspath(DB_PATH)

def init_db():
    """Initializes the SQLite database schema if not already present."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patient_biopsy_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            patient_name TEXT NOT NULL,
            age REAL,
            gender TEXT,
            damage_level INTEGER NOT NULL,
            damage_title TEXT NOT NULL,
            risk_probability REAL NOT NULL,
            has_hypertension INTEGER,
            glucose_min REAL,
            phosphate_mean REAL,
            potassium_mean REAL,
            bicarbonate_min REAL,
            albumin_min REAL,
            primary_intervention TEXT,
            causal_warning TEXT,
            counterfactual_reduction REAL,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient_record(record: dict) -> int:
    """Inserts a new virtual biopsy record into SQLite database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO patient_biopsy_records (
            timestamp, patient_id, patient_name, age, gender,
            damage_level, damage_title, risk_probability,
            has_hypertension, glucose_min, phosphate_mean,
            potassium_mean, bicarbonate_min, albumin_min,
            primary_intervention, causal_warning, counterfactual_reduction, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        record.get('patient_id', 'PAT-0000'),
        record.get('patient_name', 'Anonymous'),
        float(record.get('age', 55.0)),
        record.get('gender', 'Male'),
        int(record.get('damage_level', 0)),
        record.get('damage_title', 'Damage Level 0 (Mild / Stable)'),
        float(record.get('risk_probability', 0.0)),
        int(record.get('has_hypertension', 0)),
        float(record.get('glucose_min', 80.0)),
        float(record.get('phosphate_mean', 3.5)),
        float(record.get('potassium_mean', 4.2)),
        float(record.get('bicarbonate_min', 22.0)),
        float(record.get('albumin_min', 3.8)),
        record.get('primary_intervention', 'Routine Monitoring'),
        record.get('causal_warning', 'None'),
        float(record.get('counterfactual_reduction', 0.0)),
        record.get('notes', '')
    ))
    record_id = c.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_all_records() -> pd.DataFrame:
    """Fetches all biopsy records as a pandas DataFrame."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM patient_biopsy_records ORDER BY id DESC", conn)
    conn.close()
    return df

def delete_record(record_id: int):
    """Deletes a record by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM patient_biopsy_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def clear_all_records():
    """Wipes the database table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM patient_biopsy_records")
    conn.commit()
    conn.close()

def seed_sample_patients():
    """Seeds realistic clinical sample patient profiles for testing and demonstration."""
    init_db()
    existing = get_all_records()
    if len(existing) > 0:
        return len(existing)
    
    samples = [
        {
            'patient_id': 'MIMIC-41029',
            'patient_name': 'Eleanor Vance',
            'age': 67.0,
            'gender': 'Female',
            'damage_level': 3,
            'damage_title': 'Damage Level 3: Critical Failure (End-Stage Fibrosis)',
            'risk_probability': 0.942,
            'has_hypertension': 1,
            'glucose_min': 185.0,
            'phosphate_mean': 6.4,
            'potassium_mean': 5.8,
            'bicarbonate_min': 14.0,
            'albumin_min': 2.1,
            'primary_intervention': 'Aggressive Blood Pressure Control (ACEi/ARB) & Glycemic Optimization',
            'causal_warning': 'DO NOT target Phosphate/Potassium directly to reverse fibrosis (ATE = 0.0)',
            'counterfactual_reduction': 13.1,
            'notes': 'Diabetic nephropathy with severe diffuse glomerulosclerosis and tubular atrophy.'
        },
        {
            'patient_id': 'MIMIC-58211',
            'patient_name': 'Robert Sterling',
            'age': 59.0,
            'gender': 'Male',
            'damage_level': 2,
            'damage_title': 'Damage Level 2: Severe Tissue Damage',
            'risk_probability': 0.684,
            'has_hypertension': 1,
            'glucose_min': 110.0,
            'phosphate_mean': 4.8,
            'potassium_mean': 5.1,
            'bicarbonate_min': 18.0,
            'albumin_min': 2.8,
            'primary_intervention': 'Initiate Antihypertensives (ACEi/ARB) & Monitor Renal Hemodynamics',
            'causal_warning': 'Elevated Phosphate is a symptom of reduced eGFR, not the primary etiologic driver.',
            'counterfactual_reduction': 8.3,
            'notes': 'Hypertensive nephrosclerosis with progressive tubular injury.'
        },
        {
            'patient_id': 'MIMIC-89104',
            'patient_name': 'Sarah Jenkins',
            'age': 44.0,
            'gender': 'Female',
            'damage_level': 1,
            'damage_title': 'Damage Level 1: Moderate Risk',
            'risk_probability': 0.385,
            'has_hypertension': 0,
            'glucose_min': 140.0,
            'phosphate_mean': 3.9,
            'potassium_mean': 4.4,
            'bicarbonate_min': 21.0,
            'albumin_min': 3.4,
            'primary_intervention': 'Intensive Glycemic Control (Metformin / SGLT2i)',
            'causal_warning': 'Normal phosphate and potassium; focus on preventing microvascular injury.',
            'counterfactual_reduction': 4.8,
            'notes': 'Early stage diabetic stress; glomeruli mostly intact.'
        },
        {
            'patient_id': 'MIMIC-10294',
            'patient_name': 'Arthur Pendelton',
            'age': 38.0,
            'gender': 'Male',
            'damage_level': 0,
            'damage_title': 'Damage Level 0: Mild / Preserved Architecture',
            'risk_probability': 0.128,
            'has_hypertension': 0,
            'glucose_min': 88.0,
            'phosphate_mean': 3.2,
            'potassium_mean': 4.1,
            'bicarbonate_min': 24.5,
            'albumin_min': 4.1,
            'primary_intervention': 'Routine Preventive Surveillance (Annual Renal Profile)',
            'causal_warning': 'No causal root causes identified. Normal renal parenchyma.',
            'counterfactual_reduction': 0.0,
            'notes': 'Normal healthy baseline; preserved glomerular filtration and intact tubules.'
        }
    ]
    
    for s in samples:
        save_patient_record(s)
    return len(samples)

if __name__ == '__main__':
    init_db()
    seeded = seed_sample_patients()
    df = get_all_records()
    print(f"Database initialized. Total records in DB: {len(df)}")
