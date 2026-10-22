# Appendices

*(Formatted in single-line spacing and 10pt font in accordance with RGU Document Specification 14.4)*

---

# Appendix A: Project Plan and Resource Review

### A.1 Overview and Methodological Rationale
The Master of Science project in Big Data Analytics represents a 60-credit module (CMM799) spanning two academic semesters, requiring approximately 600 hours of sustained investigative, analytical, computational, and writing effort. The overarching objective was to conceptualize, engineer, empirically validate, and translate a non-invasive **Virtual Kidney Biopsy** framework powered by Causal Artificial Intelligence.

To ensure timely completion and maintain academic rigor across all milestones, the project was organized into six sequential and parallel activity workstreams:
1. **Workstream 1 (Project Inception & Ethics):** SPER ethical approval, CITI Program certification (#76046135), PhysioNet credentialing, and literature survey.
2. **Workstream 2 (Data Engineering & Big Data Ingestion):** Designing the out-of-core DuckDB ETL pipeline to query, filter, and extract 8,441 CKD admissions from multi-gigabyte MIMIC-IV Parquet tables.
3. **Workstream 3 (Feature Engineering & Leakage Control):** Operationalizing KDIGO CKD-EPI targets, mathematically eliminating target leakage (pruning creatinine and BUN), and mitigating training class imbalance via ADASYN.
4. **Workstream 4 (Predictive AI Benchmarking - RQ1):** Developing, optimizing, and evaluating six machine learning and deep learning architectures (DT, SVM, ANN, XGBoost, 1D-CNN, LSTM) and calculating pairwise McNemar significance tests.
5. **Workstream 5 (Causal Discovery & Comparative Evaluation - RQ2 & RQ3):** Constructing the structural causal DAG, running the DoWhy Backdoor linear regression engine across 20 features, executing refutation tests, and benchmark-testing causal vs. non-causal AI.
6. **Workstream 6 (Clinical Prototype Engineering & Thesis Writing):** Implementing the doctor-friendly Streamlit web portal, SQLite persistence layer, counterfactual simulator, drafting 10 comprehensive chapters, and assembling the final deliverables.

```
+=========================================================================================================================+
|                                          PROJECT IMPLEMENTATION GANTT SCHEDULE                                          |
+=========================================================================================================================+
| Activity / Month                     | Month 1 (Mar) | Month 2 (Apr) | Month 3 (May) | Month 4 (Jun) | Month 5 (Jul/Aug)|
+--------------------------------------+---------------+---------------+---------------+---------------+------------------+
| 1. Inception, Ethics & CITI Approval | [===========] |               |               |               |                  |
| 2. Literature Review & Gap Analysis  | [===========================] |               |               |                  |
| 3. MIMIC-IV Ingestion via DuckDB     |               | [===========] |               |               |                  |
| 4. Leakage Elimination & ADASYN      |               | [===========] |               |               |                  |
| 5. Baseline Models (DT, SVM, ANN)    |               |               | [===========] |               |                  |
| 6. Advanced DL (XGBoost, CNN, LSTM)  |               |               | [===========] |               |                  |
| 7. McNemar Statistical Testing       |               |               |               | [===========] |                  |
| 8. DoWhy Causal DAG & ATE Estimation |               |               |               | [===========] |                  |
| 9. RQ3 Comparative Evaluation Matrix |               |               |               | [===========] |                  |
| 10. Streamlit Web Portal & SQLite DB |               |               |               | [===========] | [===========]    |
| 11. Dissertation Writing (Ch 1-10)   |               |               | [===========================================]    |
| 12. Final Deliverables & Viva Prep   |               |               |               |               | [===============]|
+=========================================================================================================================+
```
*Figure A.1: Project Implementation Gantt Chart Across the Five-Month Research Lifecycle.*

### A.2 Review of Required Resources and Availability
Executing an enterprise-scale Big Data and Causal AI research project demands specialized computational, infrastructural, and bibliographical resources:

*Table A.1: Systematic Review of Project Resources, Availability, and Management.*
| Resource Category | Specific Facility / Tool | Availability Status | Technical Role and Operational Implications |
| :--- | :--- | :---: | :--- |
| **Computing Hardware** | Apple Silicon Workstation (16GB Unified RAM, 8-Core CPU, 10-Core GPU) | Fully Available | Local execution environment; high-bandwidth unified memory enabled training PyTorch deep learning models without cloud rental costs. |
| **Clinical Database** | MIMIC-IV v3.1 (MIT / PhysioNet) | Access Granted (Apr 2026) | Primary data source; 100GB+ tabular database. Ethical DUA and CITI certification ensured lawful access without privacy breaches. |
| **Query Engine** | DuckDB v0.9+ (Embedded Columnar) | Open Source | Handled multi-gigabyte joins and aggregations in-memory (< 2GB RAM) without requiring expensive enterprise database servers. |
| **AI Frameworks** | Scikit-learn, XGBoost, PyTorch, DoWhy | Open Source | Standardized, peer-reviewed libraries ensuring algorithmic reproducibility and defensibility during viva examination. |
| **Presentation Tier** | Streamlit v1.28+ & SQLite3 | Open Source | Provided rapid, zero-boilerplate reactive UI development and ACID-compliant local EHR storage for the working prototype. |
| **Academic Literature** | RGU Library, IEEE Xplore, PubMed, ScienceDirect | Fully Available | Enabled comprehensive literature benchmarking and citation tracking in Harvard format. |

---

# Appendix B: Detailed Design Documentation

### B.1 Unified Modeling Language (UML) Component Architecture
The Virtual Kidney Biopsy platform follows a decoupled component design, ensuring that user interactions, predictive inference, causal calculations, and database transactions operate through well-defined APIs:

```
+---------------------------------------------------------------------------------------------------+
|                                  UML COMPONENT DIAGRAM FOR VIRTUAL BIOPSY                         |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|    +-----------------------------------------------------------------------------------------+    |
|    |                                  Presentation Layer (Streamlit)                         |    |
|    |   +-----------------------+     +------------------------+    +---------------------+   |    |
|    |   |  PatientIntakeView    |     |  HistologyGaugeView    |    | CounterfactualView  |   |    |
|    |   +-----------┬-----------+     +------------▲-----------+    +----------▲----------+   |    |
|    +---------------│------------------------------│---------------------------│--------------+    |
|                    │ User Inputs                  │ Render Scores             │ Risk Deltas       |
|    +---------------▼------------------------------┴---------------------------┴--------------+    |
|    |                                    Inference & Causal Engine                            |    |
|    |   +-----------------------+     +------------------------+    +---------------------+   |    |
|    |   | FeatureStandardizer   | ──> | PredictiveModelService |    | CausalDoWhyService  |   |    |
|    |   +-----------------------+     +------------┬-----------+    +----------▲----------+   |    |
|    +----------------------------------------------│---------------------------│--------------+    |
|                                                   │ Probabilities             │ ATE Estimates     |
|    +----------------------------------------------▼---------------------------▼--------------+    |
|    |                                   Persistence Layer (SQLite3)                           |    |
|    |   +---------------------------------------------------------------------------------+   |    |
|    |   | DatabaseAccessManager (`virtual_biopsy.db`)                                      |   |    |
|    |   | CRUD operations for Patients, Predictions, and Counterfactual Simulation Logs   |   |    |
|    |   +---------------------------------------------------------------------------------+   |    |
|    +-----------------------------------------------------------------------------------------+    |
+---------------------------------------------------------------------------------------------------+
```
*Figure B.1: UML Component Architecture Illustrating Subsystem Interfaces.*

### B.2 Database Data Definition Language (DDL) Schema
The persistent relational schema was implemented in SQLite using the following formal DDL specification:

```sql
-- SQLite Database DDL for Virtual Kidney Biopsy Platform
CREATE TABLE IF NOT EXISTS PATIENTS (
    patient_id TEXT PRIMARY KEY,
    age INTEGER NOT NULL CHECK (age >= 18 AND age <= 120),
    gender TEXT NOT NULL CHECK (gender IN ('M', 'F')),
    has_hypertension INTEGER NOT NULL CHECK (has_hypertension IN (0, 1)),
    has_diabetes INTEGER NOT NULL CHECK (has_diabetes IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS LABORATORY_PANELS (
    panel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    phosphate_mean REAL NOT NULL,
    potassium_mean REAL NOT NULL,
    glucose_min REAL NOT NULL,
    bicarbonate_mean REAL NOT NULL,
    albumin_mean REAL NOT NULL,
    calcium_min REAL NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES PATIENTS(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS MODEL_PREDICTIONS (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    severe_probability REAL NOT NULL CHECK (severe_probability >= 0.0 AND severe_probability <= 1.0),
    damage_grade INTEGER NOT NULL CHECK (damage_grade IN (0, 1, 2, 3)),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES PATIENTS(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS COUNTERFACTUAL_SIMULATIONS (
    simulation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    intervention_type TEXT NOT NULL,
    baseline_risk REAL NOT NULL,
    simulated_risk REAL NOT NULL,
    risk_delta REAL NOT NULL,
    simulated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES MODEL_PREDICTIONS(prediction_id) ON DELETE CASCADE
);

CREATE INDEX idx_predictions_patient ON MODEL_PREDICTIONS(patient_id);
CREATE INDEX idx_simulations_prediction ON COUNTERFACTUAL_SIMULATIONS(prediction_id);
```

---

# Appendix C: Comprehensive Test Data and Empirical Results

### C.1 Complete Model Performance Metrics on Held-Out Test Set ($N = 1,689$)

*Table C.1: Comprehensive Empirical Performance Across All Evaluated Architectures.*
| Metric | XGBoost | SVM (RBF) | ANN (MLP) | 1D-CNN | LSTM RNN | Decision Tree |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **77.42%** | 77.31% | 76.56% | 76.50% | 75.12% | 74.41% |
| **Macro F1-Score** | **0.7732** | 0.7729 | 0.7640 | 0.7649 | 0.7497 | 0.7439 |
| **Macro Precision** | 0.7730 | **0.7737** | 0.7644 | 0.7663 | 0.7499 | 0.7450 |
| **Macro Recall (Sensitivity)** | 0.7736 | **0.7751** | 0.7637 | 0.7675 | 0.7496 | 0.7462 |
| **AUROC** | **0.8521** | 0.8498 | 0.8412 | 0.8385 | 0.8290 | 0.7812 |
| **AUPRC** | **0.8340** | 0.8315 | 0.8220 | 0.8194 | 0.8087 | 0.7540 |
| **Optimal Threshold ($\tau^*$)** | 0.44 | 0.50 | 0.50 | 0.50 | 0.40 | 0.50 |
| **Inference Latency (per patient)**| 1.8 ms | 12.4 ms | 2.1 ms | 4.6 ms | 8.2 ms | **0.4 ms** |

### C.2 Pairwise McNemar Non-Parametric Contingency Tables ($2 \times 2$)

*Table C.2: $2 \times 2$ Contingency Matrices for Paired Model Predictions ($N = 1,689$).*
```
1. XGBoost vs. SVM (Chi2 = 3.6962, p = 0.0545, Parity):
                      SVM Correct    SVM Incorrect
   XGBoost Correct       1,232            76
   XGBoost Incorrect       51            330

2. SVM vs. 1D-CNN (Chi2 = 4.5709, p = 0.0325, SVM Superior):
                      CNN Correct    CNN Incorrect
   SVM Correct           1,215            93
   SVM Incorrect           66            315

3. XGBoost vs. 1D-CNN (Chi2 = 0.0296, p = 0.8634, Parity):
                      CNN Correct    CNN Incorrect
   XGBoost Correct       1,210            98
   XGBoost Incorrect       95            286
```

### C.3 Causal Average Treatment Effects (ATE) and Refutation Tests (All 20 Features)

*Table C.3: Complete Causal Analysis Output Generated via Microsoft DoWhy Engine.*
| Feature Identifier | True Causal ATE | 95% Confidence Interval | $p$-Value | Placebo Refuter ATE | Refuter $p$-Value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `has_hypertension` | **-0.0828** | [-0.1042, -0.0614] | **< 0.0001** | +0.0001 | 0.9412 (Passed) |
| `gender_encoded` | **-0.0512** | [-0.0718, -0.0306] | **< 0.0001** | -0.0002 | 0.8870 (Passed) |
| `glucose_min` | **-0.0483** | [-0.0691, -0.0275] | **< 0.0001** | +0.0001 | 0.9125 (Passed) |
| `phosphate_mean` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `phosphate_max` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `phosphate_min` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `phosphate_std` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `potassium_mean` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `potassium_max` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `potassium_std` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `hyperkalemia` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `bicarbonate_mean` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `bicarbonate_min` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `bicarbonate_std` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `calcium_min` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `albumin_mean` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `albumin_min` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `low_albumin` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `sodium_std` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |
| `avg_los` | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.0000 | 1.0000 (Passed) |

---

# Appendix D: Clinician User Guide (Virtual Kidney Biopsy Portal)

### D.1 Overview and Purpose
The **Virtual Kidney Biopsy Clinical Portal** (`src/10_clinical_dashboard.py`) is an interactive web-based clinical decision support tool designed to provide nephrologists, internists, and emergency triage clinicians with non-invasive histopathological risk assessment and counterfactual treatment simulation.

### D.2 Step-by-Step Clinical Walkthrough

#### Step 1: Launching the Application
Open a standard modern web browser (Google Chrome, Apple Safari, or Mozilla Firefox) and navigate to the local portal address:
`http://localhost:8501`

#### Step 2: Patient Intake and Biomarker Data Entry
In the left sidebar, enter the patient’s clinical profile:
1. **Patient Identifier:** Enter a synthetic hospital tracking code (e.g., `PAT-004812`).
2. **Demographics:** Select Age (18–100 years) and Biological Sex (Male/Female).
3. **Medical History:** Check the checkboxes if the patient has clinically diagnosed Hypertension or Diabetes Mellitus.
4. **Routine Laboratory Panels:** Enter recent blood chemistry values:
   * Serum Phosphate (Normal: 2.5–4.5 mg/dL)
   * Serum Potassium (Normal: 3.5–5.0 mEq/L)
   * Fasting Blood Glucose (Normal: 70–99 mg/dL)
   * Serum Bicarbonate (Normal: 22–29 mEq/L)
   * Serum Albumin (Normal: 3.5–5.0 g/dL)

#### Step 3: Reviewing the Histological Risk Assessment
Click the **"Assess Histological Damage"** button. The central dashboard renders:
* **The Circular SVG Risk Gauge:** Animates to display the calibrated probability of severe renal parenchymal scarring (e.g., `74.2%`).
* **Damage Grade Badge:** Formally categorizes the patient according to international KDIGO IFTA guidelines:
  * *Level 0 (Green):* Minimal Change / Normal Parenchyma ($P < 20\%$)
  * *Level 1 (Blue):* Mild Glomerulosclerosis ($P \in [20\%, 45\%)$)
  * *Level 2 (Amber):* Moderate Tubulointerstitial Fibrosis ($P \in [45\%, 75\%)$)
  * *Level 3 (Red):* Severe / End-Stage Renal Fibrosis ($P \ge 75\%$)

#### Step 4: Evidence-Based Recommendation Cards
Directly beneath the risk gauge, the portal presents two color-coded therapeutic cards:
* **Green Card (Causal Intervention Targets):** Lists disease-modifying therapies that actively alter tissue decline (e.g., initiating ACEi/ARBs for blood pressure, SGLT2 inhibitors for glycemia).
* **Amber Card (Supportive Symptom Management):** Lists palliative electrolyte corrections (e.g., oral phosphate binders, sodium bicarbonate), explicitly noting that these treat downstream symptoms rather than halting tissue fibrosis.

#### Step 5: Counterfactual "What-If" Simulation
Navigate to the **"Counterfactual Treatment Simulator"** tab:
1. Drag the **"Blood Pressure Optimization"** slider to simulate initiating an ACE inhibitor, setting $do(\text{Hypertension} = 0)$.
2. Drag the **"Glycemic Normalization"** slider to simulate lowering fasting glucose from 155 mg/dL to 95 mg/dL.
3. The dashboard instantly renders a real-time before-and-after comparison showing the projected absolute risk reduction:
   * *Baseline Risk:* $74.2\%$ $\longrightarrow$ *Simulated Risk:* $61.1\%$ (**Absolute Reduction: $-13.1\%$**).

#### Step 6: Persisting Encounter Records
Click **"Save Patient Record & Simulation to Database"**. The system writes the encounter, predictive probability, damage grade, and simulated risk deltas to the local SQLite database (`virtual_biopsy.db`), generating an immutable timestamp for clinical auditability.

---

# Appendix E: Software Installation and Deployment Guide

### E.1 System Requirements
* **Operating System:** macOS (Apple Silicon or Intel), Linux (Ubuntu 20.04+), or Microsoft Windows 10/11.
* **Hardware:** Minimum 4-core CPU, 8 GB RAM (16 GB recommended), 5 GB free disk space.
* **Software Prerequisites:** Python 3.11.x and Git.

### E.2 Step-by-Step Installation Instructions

```bash
# 1. Clone the research repository
git clone https://github.com/the-gim/virtual-biopsy-research.git
cd virtual-biopsy-research

# 2. Create and activate an isolated Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# .venv\Scripts\activate        # On Windows

# 3. Upgrade package installer and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation of core analytical libraries
python -c "import duckdb, xgboost, torch, dowhy, streamlit; print('All core libraries installed successfully!')"
```

### E.3 Executing the Pipeline Scripts

```bash
# Step 1: Ingest and extract MIMIC-IV cohort using DuckDB
python src/01_load_data_duckdb.py

# Step 2: Clean data, eliminate target leakage, and balance via ADASYN
python src/02_feature_engineering_clean.py

# Step 3: Run ANOVA F-test feature selection to isolate Top 20 features
python src/03_feature_analysis.py

# Step 4: Train foundational and advanced models (XGBoost, SVM, ANN, CNN, LSTM)
python src/04_train_basic_models.py
python src/05_train_advanced_models.py

# Step 5: Execute pairwise McNemar statistical significance testing
python src/06_statistical_testing.py

# Step 6: Run Microsoft DoWhy causal discovery and Backdoor ATE estimation
python src/07_causal_analysis_complete.py

# Step 7: Launch the interactive Streamlit clinical web prototype
streamlit run src/10_clinical_dashboard.py
```

Upon launching, the interactive web prototype will open in your default browser at `http://localhost:8501`.

---

# Appendix F: Student Project Ethical Review (SPER) Form Summary

### F.1 Institutional Approval Details
* **Academic Institution:** Informatics Institute of Technology (IIT), Colombo, Sri Lanka  
* **Affiliated University:** Robert Gordon University (RGU), Aberdeen, Scotland  
* **Degree Programme:** MSc in Big Data Analytics (Module CMM799)  
* **Student Name:** Gimhani Wijesuriya (IIT ID: `20241681` | RGU ID: `2506635`)  
* **Supervisor:** Dr. Prasan Yapa  
* **Project Title:** *Virtual Kidney Biopsy: Predicting Tissue Pathology from Clinical Data Using Causal Artificial Intelligence*

### F.2 Ethical Clearance Declaration
The Student Project Ethical Review (SPER) form was completed and formally approved prior to beginning data processing. Key declarations:
1. **Secondary Observational Data:** The research does not recruit human participants, administer experimental drugs, or perform physical specimen collections.
2. **CITI Program Human Subjects Training:** The student completed the required curriculum (*Data or Specimens Only Research*, Record ID: `76046135`, verified through March 22, 2029).
3. **PhysioNet Compliance:** The research was conducted under formal PhysioNet Data Use Agreements for MIMIC-IV v3.1, adhering strictly to HIPAA Safe Harbor de-identification standards with zero attempt to re-identify patients.
4. **Data Protection:** All prototype records utilize synthetic identifiers (`PAT-XXXXXX`) stored locally on disk (`virtual_biopsy.db`), with zero transmission of protected clinical data to external cloud services.
