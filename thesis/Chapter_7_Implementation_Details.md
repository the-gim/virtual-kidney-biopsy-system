# Chapter 7: System Implementation

---

## 7.1 Computing Environment, Infrastructure, and Facilities Used

In accordance with Robert Gordon University (RGU) Guideline 9 (*Implementation details of the developed solution including resources and facilities used*), this chapter details the software engineering practices, computational infrastructure, and codebase implementation of the **Virtual Kidney Biopsy** platform.

### 7.1.1 Hardware and Development Infrastructure
The development, training, and empirical evaluation of the platform were executed on local workstation hardware, demonstrating that advanced clinical AI and causal discovery can be performed without requiring expensive enterprise cloud clusters:
* **Workstation Architecture:** Apple Silicon M-series hardware platform equipped with an 8-core CPU, 10-core GPU, and 16-core Neural Engine.
* **Memory Architecture:** 16 Gigabytes of Unified Memory Architecture (UMA), providing high-bandwidth, zero-copy memory sharing between CPU and GPU cores.
* **Operating System:** macOS Darwin (Unix-based environment) with native POSIX-compliant shell scripting and thread management.
* **Primary Runtime:** Python 3.11.x 64-bit interpreter deployed within an isolated virtual environment (`.venv`).

### 7.1.2 Software Repository and Component Organization
The software repository is organized into a modular, production-ready directory structure that isolates raw data, intermediate artifacts, source code modules, and empirical outputs:

```
virtual_biopsy_research/
│
├── data/                               # Data Directory
│   ├── raw/                            # Multi-gigabyte MIMIC-IV Parquet tables
│   └── processed/                      # Cleaned, leakage-free feature matrices
│
├── src/                                # Source Code Modules
│   ├── 01_load_data_duckdb.py          # Big Data out-of-core ingestion engine
│   ├── 02_feature_engineering_clean.py # Leakage pruning & ADASYN balancing
│   ├── 03_feature_analysis.py          # ANOVA F-statistic feature ranking
│   ├── 04_train_basic_models.py        # Decision Tree, SVM, and MLP ANN training
│   ├── 05_train_advanced_models.py     # XGBoost, 1D-CNN, and LSTM deep learning
│   ├── 06_statistical_testing.py       # McNemar hypothesis testing pipeline
│   ├── 07_causal_analysis_complete.py  # DoWhy SCM & Backdoor ATE engine
│   ├── 08_shap_counterfactuals.py      # Feature attribution & what-if analysis
│   ├── 09_causal_vs_noncausal_rq3.py   # Comparative RQ3 experimental pipeline
│   └── 10_clinical_dashboard.py        # Streamlit doctor-centric clinical portal
│
├── outputs/                            # Persistent Model & Empirical Artifacts
│   ├── models/                         # Serialized models (`.pkl`) & scalers
│   ├── results/                        # Confusion matrices, ROC curves, CSVs
│   └── causal_results/                 # DAG plots, ATE tables, refutation logs
│
├── database/                           # Persistence Layer
│   └── virtual_biopsy.db               # Local SQLite clinical records database
│
└── thesis/                             # Master's Dissertation Chapters
```
*Figure 7.1: Software Architecture and Repository Directory Hierarchy.*

---

## 7.2 Big Data Extraction & Ingestion Engine Implementation (`src/01_load_data_duckdb.py`)

Processing the MIMIC-IV clinical database requires ingesting multiple relational tables totaling tens of millions of records. Loading these uncompressed files using traditional libraries like Pandas triggers severe memory exhaustion. 

To overcome this, an embedded out-of-core ingestion pipeline was implemented using **DuckDB**. The script streams Parquet files directly from disk, executing complex SQL queries, relational joins, and group-by aggregations within a lightweight memory footprint (< 2 GB RAM):

```python
# Excerpt from src/01_load_data_duckdb.py
import duckdb

def extract_ckd_cohort(data_dir: str) -> duckdb.DuckDBPyRelation:
    """Extracts Chronic Kidney Disease cohort using ICD-10 N18 hierarchy out-of-core."""
    con = duckdb.connect(database=':memory:')
    
    query = f"""
    SELECT 
        adm.subject_id,
        adm.hadm_id,
        pat.gender,
        pat.anchor_age,
        adm.admittime,
        adm.dischtime,
        DATE_DIFF('day', adm.admittime, adm.dischtime) AS los,
        CASE WHEN diag.icd_code LIKE 'N18%' THEN 1 ELSE 0 END AS ckd_flag
    FROM '{data_dir}/admissions.parquet' adm
    JOIN '{data_dir}/patients.parquet' pat ON adm.subject_id = pat.subject_id
    JOIN '{data_dir}/diagnoses_icd.parquet' diag ON adm.hadm_id = diag.hadm_id
    WHERE diag.icd_code LIKE 'N18%' AND pat.anchor_age >= 18
    """
    return con.execute(query).df()
```

### Key Technical Achievements
* **Zero Server Overhead:** Executes in-process without requiring a dedicated PostgreSQL daemon or network socket configuration.
* **Vectorized Execution:** Processes over 12 million laboratory event rows in under 18 seconds on Apple Silicon.
* **Clean Cohort Isolation:** Successfully isolated 8,441 verified Chronic Kidney Disease hospital admissions meeting strict inclusion criteria.

---

## 7.3 Data Preprocessing & Feature Engineering Implementation (`src/02_feature_engineering_clean.py`)

The preprocessing module operationalizes clinical ground truth, implements target leakage elimination, and resolves training class imbalance:

### 7.3.1 Ground Truth Calculation and Target Formulation
The 2021 CKD-EPI refit formula was implemented in vectorized NumPy to calculate estimated Glomerular Filtration Rate ($eGFR$) without racial adjustment coefficients:

```python
# Ground truth target formulation using CKD-EPI 2021
kappa = np.where(df['gender'] == 'F', 0.7, 0.9)
alpha = np.where(df['gender'] == 'F', -0.241, -0.302)
gender_factor = np.where(df['gender'] == 'F', 1.012, 1.0)

scr_ratio = df['creatinine_mean'] / kappa
min_scr = np.minimum(scr_ratio, 1.0)
max_scr = np.maximum(scr_ratio, 1.0)

df['egfr'] = 142 * (min_scr ** alpha) * (max_scr ** -1.200) * (0.9938 ** df['anchor_age']) * gender_factor
df['severe_kidney_damage'] = (df['egfr'] < 45.0).astype(int)  # KDIGO G3b-G5 threshold
```

### 7.3.2 Mathematical Leakage Elimination
To eliminate target leakage, 28 direct mathematical proxies of $eGFR$—including raw serum creatinine and blood urea nitrogen aggregates—were explicitly purged:

```python
# Systematic leakage elimination
leaky_features = [
    'creatinine_mean', 'creatinine_min', 'creatinine_max', 'creatinine_std',
    'bun_mean', 'bun_min', 'bun_max', 'bun_std', 'egfr', 'ckd_stage'
]
df_clean = df.drop(columns=[col for col in leaky_features if col in df.columns])
```

### 7.3.3 Adaptive Synthetic (ADASYN) Oversampling
Class imbalance was resolved on the training set using `imblearn.over_sampling.ADASYN`, generating synthetic minority instances along non-linear decision boundaries:

```python
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

adasyn = ADASYN(random_state=42, n_neighbors=5)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
```

---

## 7.4 Predictive Model Training Implementation (`src/04_train_basic_models.py` & `05_train_advanced_models.py`)

Six distinct machine learning and deep learning architectures were implemented, trained, and serialized:

### 7.4.1 Foundational Models (Decision Tree, SVM, MLP ANN)
Implemented using Scikit-Learn:
* **Decision Tree:** `DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)`.
* **Support Vector Machine:** `SVC(kernel='rbf', C=1.0, probability=True, random_state=42)`.
* **Multilayer Perceptron (ANN):** `MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', early_stopping=True, max_iter=500, random_state=42)`.

### 7.4.2 Extreme Gradient Boosting (XGBoost)
Implemented using the `xgboost` library with regularized tree complexity:

```python
import xgboost as xgb

scale_pos = (len(y_train_res) - sum(y_train_res)) / sum(y_train_res)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_res, y_train_res)
```

### 7.4.3 Deep Sequential & Convolutional Architectures (PyTorch)
Constructed in PyTorch and wrapped via `skorch.NeuralNetClassifier` for seamless scikit-learn integration:
* **1D-CNN:** `Conv1d(1, 32, kernel_size=3) -> ReLU() -> MaxPool1d(2) -> Flatten() -> Linear(PostConvSize, 32) -> ReLU() -> Linear(32, 2)`.
* **LSTM:** `LSTM(input_size=1, hidden_size=32, batch_first=True) -> Dropout(0.2) -> Linear(32, 2)`.

```python
# Excerpt of PyTorch 1D-CNN implementation
import torch
import torch.nn as nn

class ConvNet1D(nn.Module):
    def __init__(self, input_dim=20):
        super(ConvNet1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Calculate post-pooling dimension dynamically
        conv_out_len = (input_dim - 3 + 1) // 2
        self.fc1 = nn.Linear(32 * conv_out_len, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.unsqueeze(1) # [Batch, 1, Features]
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

---

## 7.5 Statistical Testing Pipeline Implementation (`src/06_statistical_testing.py`)

To statistically validate predictive comparisons between models, an automated script executes non-parametric pairwise **McNemar tests** using `statsmodels.stats.contingency_tables.mcnemar`:

```python
from statsmodels.stats.contingency_tables import mcnemar

def evaluate_pairwise_mcnemar(y_true, preds_model_a, preds_model_b):
    """Computes 2x2 contingency table and Edwards-corrected McNemar test."""
    correct_a = (preds_model_a == y_true)
    correct_b = (preds_model_b == y_true)
    
    n00 = np.sum(correct_a & correct_b)
    n01 = np.sum(correct_a & ~correct_b)
    n10 = np.sum(~correct_a & correct_b)
    n11 = np.sum(~correct_a & ~correct_b)
    
    table = [[n00, n01], [n10, n11]]
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue, table
```

---

## 7.6 Causal Discovery & Effect Estimation Implementation (`src/07_causal_analysis_complete.py`)

Structural Causal Modeling was executed via Microsoft’s **DoWhy** library. The Directed Acyclic Graph was encoded in DOT format, establishing directed edges from root causes to kidney damage, and from kidney damage to downstream electrolyte markers:

```python
import dowhy
from dowhy import CausalModel

# Encode domain causal DAG
causal_graph = """
digraph {
    gender_encoded -> severe_kidney_damage;
    glucose_min -> has_hypertension;
    glucose_min -> severe_kidney_damage;
    has_hypertension -> severe_kidney_damage;
    severe_kidney_damage -> phosphate_mean;
    severe_kidney_damage -> potassium_mean;
    severe_kidney_damage -> bicarbonate_mean;
    severe_kidney_damage -> albumin_mean;
}
"""

# Execute 4-step causal workflow for each feature
model = CausalModel(
    data=df_features,
    treatment='has_hypertension',
    outcome='severe_kidney_damage',
    graph=causal_graph
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression"
)

# Robustness refutation test
refutation = model.refute_estimate(
    identified_estimand,
    estimate,
    method_name="placebo_treatment_refuter"
)
```

---

## 7.7 Clinical Dashboard & Counterfactual Simulator Implementation (`src/10_clinical_dashboard.py`)

The user-facing software prototype was developed using **Streamlit**. It integrates custom CSS for a modern, clinical look, an SVG circular risk gauge, and persistent SQLite database operations:

```python
# Excerpt from src/10_clinical_dashboard.py: Counterfactual Simulation Logic
import streamlit as st
import sqlite3

def simulate_counterfactual_intervention(baseline_vector, delta_bp, delta_glucose, model, scaler):
    """Simulates physiological intervention by modulating treatment features."""
    simulated_vector = baseline_vector.copy()
    
    # Apply clinical interventions
    if delta_bp:
        simulated_vector['has_hypertension'] = 0 # Controlled BP
    if delta_glucose:
        simulated_vector['glucose_min'] = max(70, simulated_vector['glucose_min'] - delta_glucose)
        
    # Scale and predict
    scaled_sim = scaler.transform([simulated_vector])
    simulated_prob = model.predict_proba(scaled_sim)[0][1]
    
    return simulated_prob
```

### Integrated SQLite Database Access Layer
When a clinician evaluates a patient or performs a simulation, the encounter is automatically persisted to `database/virtual_biopsy.db`:

```python
def log_patient_encounter(patient_id, age, gender, damage_grade, severe_prob):
    conn = sqlite3.connect('database/virtual_biopsy.db')
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO MODEL_PREDICTIONS (patient_id, severe_probability, damage_grade, predicted_at)
    VALUES (?, ?, ?, datetime('now'))
    """, (patient_id, severe_prob, damage_grade))
    conn.commit()
    conn.close()
```

This comprehensive, modular implementation directly satisfies RGU’s mandate for a functional software prototype, delivering an auditable, reproducible, and clinically actionable decision support platform.
