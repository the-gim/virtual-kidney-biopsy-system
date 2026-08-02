# Virtual Biopsy Research Methodology: End-to-End Pipeline

This document outlines the complete methodology and data pipeline used to answer **Research Question 1**: *Predicting kidney tissue scarring severity from non-invasive clinical data.*

## Phase 1: Data Preparation & Ground Truth Extraction
**Dataset:** MIMIC-IV (Boston, USA)
**Script:** `01_extract_labels_nlp.py`
* **Objective:** Extract "ground truth" biopsy scarring labels from unstructured doctor's notes.
* **Process:** Because the `discharge.csv` file was 3.5GB and too large to load into standard RAM, we used a chunked streaming approach. We applied Natural Language Processing (NLP) Regular Expressions to search for keywords like "biopsy", "sclerosis", "fibrosis", and "scarring". 
* **Outcome:** Successfully categorized patients into a 0-3 severity scale (None, Mild, Moderate, Severe), creating `biopsy_labels.csv`.

## Phase 2: Big Data Filtering (Vital Signs & Labs)
**Dataset:** MIMIC-IV 
**Script:** `02_filter_large_events.py`
* **Objective:** Extract relevant kidney lab tests and vital signs for our labeled cohort without crashing the computer.
* **Process:** The `chartevents.csv.gz` file was highly compressed and massive. We utilized **DuckDB**, an out-of-core SQL engine, to query the compressed file directly without unzipping it. We filtered for specific `itemid`s (Creatinine, BUN, Sodium, Potassium, Blood Pressure, Heart Rate).
* **Outcome:** Generated highly compressed `.parquet` files that drastically reduced the storage footprint from gigabytes down to megabytes.

## Phase 3: Automated Feature Engineering
**Dataset:** MIMIC-IV
**Script:** `03_automated_feature_engineering.py`
* **Objective:** Discover new, complex mathematical representations of the patient's time-series data.
* **Process:** Instead of manual feature engineering, we utilized `tsfresh`. To prevent Out-Of-Memory (OOM) errors across 45,000 patients, we used `MinimalFCParameters()` to calculate 10 core mathematical properties (e.g., max, min, median, standard deviation, root mean square) for every single lab test and vital sign over the patient's hospital stay.
* **Outcome:** Automatically generated a massive matrix of 160 mathematical features per patient.

## Phase 4: Feature Selection & Clinical Validation
**Dataset:** MIMIC-IV
**Script:** `04_feature_analysis_selection.py`
* **Objective:** Remove statistical noise and identify the most predictive clinical features.
* **Process:** We calculated the **Mutual Information** score between each of the 160 features and the target scarring severity. We pruned the dataset down to the top 115 features.
* **Outcome:** The algorithm successfully identified *Urine Color* and *Urine Specific Gravity* as the top predictors of tissue scarring. This provided immense clinical validity to the AI, as scarring physically impairs the kidney's ability to concentrate urine.

## Phase 5: Machine Learning Model Training (RQ1)
**Dataset:** MIMIC-IV
**Script:** `05_train_models.py`
* **Objective:** Train predictive algorithms and address the massive class imbalance (44k healthy vs ~800 diseased).
* **Process:** We implemented **SMOTE** (Synthetic Minority Over-sampling Technique) inside a 5-Fold Cross-Validation pipeline to synthesize minority class data. We trained an Artificial Neural Network (ANN), Support Vector Machine (SVM), and Decision Tree.
* **Outcome:** The ANN achieved **95.18% Accuracy**, massively outperforming the 62% literature baseline cited in the original research proposal. The models were saved to disk using `joblib`.

## Phase 6: External Hospital Validation (eICU)
**Dataset:** eICU Collaborative Research Database Demo (National, USA)
**Scripts:** `06`, `07`, `08`, `09`
* **Objective:** Prove the models are robust and can generalize to unseen hospitals.
* **Process:** 
  1. Extracted severity labels from structured eICU billing codes (`diagnosis.csv`).
  2. Transformed eICU's "wide" spreadsheet vitals into the "long" time-series format of MIMIC-IV.
  3. Ran `tsfresh` to calculate the exact same 115 mathematical features.
  4. Loaded the saved MIMIC models and evaluated them on the eICU data.
* **Outcome:** Due to expected "Domain Shift" between hospitals, the ANN's accuracy dropped to **84.08%**. However, this still vastly outperformed the 62% baseline. The Decision Tree demonstrated the highest robustness to the shift, achieving the best Macro F1-Score (0.2568) on the external dataset.
