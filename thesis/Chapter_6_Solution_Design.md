# Chapter 6: Architectural and Solution Design

---

## 6.1 High-Level Multi-Tier System Architecture

Translating complex machine learning and causal inference algorithms into a functional, clinician-accessible software solution requires a decoupled, modular, and maintainable architectural design. In accordance with Robert Gordon University (RGU) Guideline 8 (*Design of the Solution*), this chapter details the structural, conceptual, and logical architecture of the **Virtual Kidney Biopsy** platform.

The system is structured as an enterprise-grade **Four-Tier Architecture**, cleanly separating data ingestion, machine learning inference, causal reasoning, and user presentation:

```
+===================================================================================================+
|                                    TIER 1: PRESENTATION TIER                                      |
|                                (Streamlit Clinical Web Portal)                                    |
|                                                                                                   |
|   [Patient Intake Form]  <--->  [Dynamic SVG Risk Gauge]  <--->  [Counterfactual Simulator UI]     |
|   • Demographic inputs         • Real-time probability          • Interactive BP/Glucose sliders  |
|   • 20 Clinical labs           • KDIGO Damage Grades 0–3        • Before-and-after risk delta     |
+===================================================================================================+
                                                 │
                                                 │ REST / In-Memory Python API
                                                 ▼
+===================================================================================================+
|                              TIER 2: ANALYTICAL & INFERENCE TIER                                  |
|                            (Model Execution & Threshold Optimization)                             |
|                                                                                                   |
|   [Scikit-learn Transformers]  ──>  [Pre-Trained Classifiers]   ──>  [Threshold Calibrator]       |
|   • Z-Score Standard Scaler         • XGBoost (Depth 4, Reg)         • Dynamic F1-Threshold (tau) |
|   • Categorical Preservers          • SVM (RBF Kernel, C=1.0)        • Calibrated Probabilities   |
|                                     • 1D-CNN & LSTM (PyTorch)        • Multi-Class Pathology Maps |
+===================================================================================================+
                                                 │
                                                 │ Causal Vectors / DAG Queries
                                                 ▼
+===================================================================================================+
|                                  TIER 3: CAUSAL REASONING TIER                                    |
|                           (Microsoft DoWhy SCM & Backdoor Engine)                                 |
|                                                                                                   |
|   [Directed Acyclic Graph (DAG)]  ──>  [Backdoor Identifier]  ──>  [Counterfactual Calculator]    |
|   • Root Causes vs Symptoms            • d-Separation Logic         • do(X = x) Interventions     |
|   • Structural Equations               • Confounder Conditioning    • True ATE Calculations       |
+===================================================================================================+
                                                 │
                                                 │ Local SQL & Chunked I/O
                                                 ▼
+===================================================================================================+
|                                   TIER 4: DATA & PERSISTENCE TIER                                 |
|                                  (Embedded DuckDB & Local SQLite)                                 |
|                                                                                                   |
|   [DuckDB Analytical Engine]                    [SQLite Transactional Database]                   |
|   • Out-of-core Parquet query engine            • `virtual_biopsy.db`                             |
|   • MIMIC-IV cohort extraction (8,441 rows)     • Patients, Predictions & Simulation Logs         |
+===================================================================================================+
```
*Figure 6.1: High-Level Four-Tier Architecture of the Virtual Kidney Biopsy Platform.*

### Architectural Separation of Concerns
1. **Tier 1 (Presentation):** Implemented in Python using Streamlit with custom CSS. Renders a responsive, doctor-friendly clinical dashboard that abstracts all algorithmic complexity behind intuitive medical inputs, interactive SVG gauges, and before-and-after intervention delta cards.
2. **Tier 2 (Analytical & Inference):** Loads serialized machine learning pipelines, executes real-time feature normalization, evaluates probability vectors, and applies calibrated decision thresholds ($\tau^*$) to stratify histological damage severity.
3. **Tier 3 (Causal Reasoning):** Houses the Structural Causal Model (SCM) constructed via DoWhy. It maintains the Directed Acyclic Graph, applies the Backdoor adjustment criterion to block confounding paths, and computes counterfactual treatment outcomes when the clinician alters physiological targets.
4. **Tier 4 (Data & Persistence):** Employs **DuckDB** for high-throughput, out-of-core analytical aggregation of MIMIC-IV Big Data tables, and **SQLite** for serverless, ACID-compliant local storage of patient profiles and simulation logs.

---

## 6.2 Conceptual and Logical Data Model

Data persistence is partitioned into two distinct paradigms: an analytical warehouse for retrospective research extraction, and an operational transactional database for clinical intake.

### 6.2.1 Analytical Data Model (DuckDB)
To manage the tens of millions of records in MIMIC-IV v3.1, an out-of-core columnar data model was constructed within DuckDB. Analytical queries join four relational tables without loading them entirely into system RAM:
* `patients`: Provides biological sex, anchor age, and mortality timestamps.
* `admissions`: Captures admission type, discharge disposition, and length of stay (LOS).
* `diagnoses_icd`: Contains ICD-10 diagnostic billing codes used to filter the Chronic Kidney Disease cohort (`N18` hierarchy).
* `labevents`: Contains time-series biochemical measurements filtered by `itemid` (phosphate, potassium, sodium, bicarbonate, calcium, albumin, glucose).

### 6.2.2 Transactional Data Model (SQLite)
For the clinical web prototype, an ACID-compliant SQLite schema (`virtual_biopsy.db`) was engineered to track patient encounters, automated predictions, and simulated interventions over time:

```
+---------------------------------------------------------------------------------------------------+
|                                 SQLITE CLINICAL DATABASE SCHEMA (ERD)                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|    +-----------------------+                   +----------------------------------+               |
|    |       PATIENTS        | 1               * |         LABORATORY_PANELS        |               |
|    +-----------------------+-------------------+----------------------------------+               |
|    | PK  patient_id (TEXT) |                   | PK  panel_id (INTEGER AUTO)      |               |
|    |     age (INTEGER)     |                   | FK  patient_id (TEXT)            |               |
|    |     gender (TEXT)     |                   |     recorded_at (DATETIME)       |               |
|    |     has_hypertension  |                   |     phosphate_mean (REAL)        |               |
|    |     has_diabetes      |                   |     potassium_mean (REAL)        |               |
|    |     created_at        |                   |     glucose_min (REAL)           |               |
|    +-----------------------+                   |     bicarbonate_mean (REAL)      |               |
|                │                               |     albumin_mean (REAL)          |               |
|                │ 1                             +----------------------------------+               |
|                │                                                 │                                |
|                │ *                                               │ 1                              |
|    +-----------▼----------------------+                          │                                |
|    |        MODEL_PREDICTIONS         |                          │ *                              |
|    +----------------------------------+                          │                                |
|    | PK  prediction_id (INTEGER AUTO) |                          │                                |
|    | FK  patient_id (TEXT)            |                          │                                |
|    |     model_name (TEXT)            |                          │                                |
|    |     severe_probability (REAL)    |                          │                                |
|    |     damage_grade (INTEGER)       |                          │                                |
|    |     predicted_at (DATETIME)      |                          │                                |
|    +----------------------------------+                          │                                |
|                │ 1                                               │                                |
|                │                                                 │                                |
|                │ *                                               ▼                                |
|    +-----------▼--------------------------------------------------------------------+             |
|    |                             COUNTERFACTUAL_SIMULATIONS                         |             |
|    +--------------------------------------------------------------------------------+             |
|    | PK  simulation_id (INTEGER AUTO)                                               |             |
|    | FK  prediction_id (INTEGER)                                                    |             |
|    |     intervention_type (TEXT: 'Hypertension', 'Glucose')                        |             |
|    |     baseline_risk (REAL)                                                       |             |
|    |     simulated_risk (REAL)                                                      |             |
|    |     risk_delta (REAL)                                                          |             |
|    |     simulated_at (DATETIME)                                                    |             |
|    +--------------------------------------------------------------------------------+             |
+---------------------------------------------------------------------------------------------------+
```
*Figure 6.2: Entity-Relationship Diagram for Clinical Decision Support and Simulation Persistence.*

---

## 6.3 Structural Causal Model & Directed Acyclic Graph (DAG) Design

The defining innovation of the Virtual Kidney Biopsy framework is replacing associative correlations with formal structural causal mechanisms.

### 6.3.1 Structural Equation Specifications
In accordance with Judea Pearl's Structural Causal Models (Pearl, 2009), the clinical domain was formalized as a system of non-linear structural functions governed by background noise distributions:

1. **Exogenous Demographic & Comorbidity Foundations:**
   $$\text{Gender} = U_{\text{Gender}}, \quad U_{\text{Gender}} \sim \text{Bernoulli}(p_{\text{sex}})$$
   $$\text{Glucose}_{\min} = f_{\text{gluc}}(\text{Age}, U_{\text{gluc}})$$
   $$\text{Hypertension} = f_{\text{hyp}}(\text{Glucose}_{\min}, \text{Age}, U_{\text{hyp}})$$

2. **Central Pathological Tissue State ($Y$):**
   Renal parenchymal damage ($Y$, binary severe scarring or multi-grade damage) is determined exclusively by true etiologic root causes:
   $$Y = f_{\text{path}}(\text{Hypertension}, \text{Glucose}_{\min}, \text{Gender}, U_Y)$$

3. **Downstream Symptom Manifestations ($S_i$):**
   Electrolyte and mineral concentrations are determined *downstream* as physiological consequences of failing filtration capacity ($Y$):
   $$\text{Phosphate}_{\text{mean}} = f_{\text{phos}}(Y, U_{\text{phos}})$$
   $$\text{Potassium}_{\text{mean}} = f_{\text{pot}}(Y, U_{\text{pot}})$$
   $$\text{Bicarbonate}_{\text{mean}} = f_{\text{bicarb}}(Y, U_{\text{bicarb}})$$
   $$\text{Albumin}_{\text{mean}} = f_{\text{alb}}(Y, U_{\text{alb}})$$

```
                               STRUCTURAL CAUSAL MODEL (DAG)
  
      [Gender] ───────────────┐
                              ▼
      [Glucose_min] ───> [Kidney Tissue Damage (Y)] ───┬───> [Phosphate (mean/max/min)]
           │                  ▲                        ├───> [Potassium (mean/max/std)]
           ▼                  │                        ├───> [Bicarbonate (mean/min)]
      [Hypertension] ─────────┘                        ├───> [Albumin (mean/min)]
                                                       └───> [Hospital LOS]

      ================================================================================
      ROOT CAUSES (Intervention Targets):            DOWNSTREAM SYMPTOMS (Non-Causal):
      • ATE(Hypertension) = -0.0828 (p < 0.001)      • ATE(Phosphate) = 0.0000
      • ATE(Glucose_min)  = -0.0483 (p < 0.001)      • ATE(Potassium) = 0.0000
      • ATE(Gender)       = -0.0512 (p < 0.001)      • ATE(Bicarbonate) = 0.0000
```
*Figure 6.3: Causal Graph Encoding Directional Mechanisms and Average Treatment Effects.*

### 6.3.2 Backdoor Identification Guarantee
Under this DAG architecture, every path connecting downstream symptom markers (e.g., `phosphate_mean`) to kidney damage ($Y$) that does not follow a directed causal edge is an outbound manifestation path ($Y \to \text{Phosphate}$). Because there are no arrows directed into $Y$ originating from electrolyte nodes, applying Pearl's Backdoor Criterion confirms that the causal effect of altering downstream symptoms on kidney damage is identically zero ($ATE = 0.0000$).

Conversely, for the root causes (e.g., `has_hypertension`), the backdoor path through `glucose_min` and `age` is blocked by conditioning on the sufficient adjustment set:

$$Z = \{\text{glucose\_min}, \text{gender\_encoded}\}$$

This guarantees that the Average Treatment Effect of hypertension on renal scarring is non-parametrically identifiable from observational clinical records.

---

## 6.4 Machine Learning Inference Pipeline Design

The inference engine was engineered to ensure strict separation between model training artifacts and real-time execution:

```
+---------------------------------------------------------------------------------------------------+
|                              MODEL PIPELINE AND INFERENCE WORKFLOW                                |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|    [Raw Patient Input Vector] (20 Clinical & Demographic Indicators)                              |
|                │                                                                                  |
|                ▼                                                                                  |
|    [Pre-Fitted StandardScaler] (Loaded from `outputs/models/scaler.pkl`)                          |
|    • Normalizes 16 continuous laboratory aggregates                                               |
|    • Preserves 4 binary comorbidity flags (`has_hypertension`, `gender_encoded`, etc.)            |
|                │                                                                                  |
|                ▼                                                                                  |
|    [Optimized XGBoost Classifier] (Loaded from `outputs/models/xgboost_model.pkl`)                |
|    • Computes continuous severe pathology probability: P(Damage = 1 | X)                          |
|                │                                                                                  |
|                ├─────────────────────────────────────────────────┐                                |
|                ▼                                                 ▼                                |
|    [Threshold Evaluator (tau* = 0.42)]          [KDIGO Damage Grade Mapper]                       |
|    • IF P >= 0.42 -> Binary High Risk (Class 1) • P < 0.20  -> Level 0 (Minimal Change)          |
|    • IF P < 0.42  -> Binary Low Risk (Class 0)  • P 0.20-0.45 -> Level 1 (Mild Glomerulosclerosis) |
|                                                 • P 0.45-0.75 -> Level 2 (Moderate Fibrosis)      |
|                                                 • P >= 0.75  -> Level 3 (Severe / End-Stage)      |
+---------------------------------------------------------------------------------------------------+
```
*Figure 6.4: Two-Stage Inference Pipeline Mapping Continuous Probabilities to Discrete Histopathology.*

### Pipeline Features
* **Zero Leakage Execution:** The scaler object applied during inference was fit strictly on the training partition of the MIMIC-IV dataset, preventing data distribution shifts.
* **Dual Output Modality:** Provides both binary triage categorization (satisfying emergency clinical screening) and 4-tier histological damage staging (aligning with international KDIGO IFTA pathology guidelines).

---

## 6.5 Clinical User Interface Architecture

To ensure high usability in fast-paced hospital settings, the presentation layer was designed using human factors engineering principles:

```
+===================================================================================================+
|                                  VIRTUAL KIDNEY BIOPSY CLINICAL PORTAL                            |
+===================================================================================================+
|  PATIENT PROFILE               HISTOPATHOLOGICAL RISK ASSESSMENT                                  |
|  • ID: PAT-849201              +---------------------------------------------------------------+  |
|  • Age: 64  | Sex: Male        |                   CIRCULAR RISK GAUGE                         |  |
|  • History: Hypertension [X]   |                         [ 74.2% ]                             |  |
|             Diabetes     [X]   |                 PREDICTED SEVERE SCARRING                     |  |
|                                |                                                               |  |
|  LABORATORY VALUES             |  [BADGE: GRADE 2 - MODERATE TUBULOINTERSTITIAL FIBROSIS]      |  |
|  • Phosphate: 4.8 mg/dL        +---------------------------------------------------------------+  |
|  • Potassium: 5.2 mEq/L                                                                           |
|  • Glucose:   145 mg/dL        EVIDENCE-BASED CLINICAL RECOMMENDATIONS                            |
|  • Bicarb:    21  mEq/L        +-------------------------------+-------------------------------+  |
|  • Albumin:   3.2 g/dL         | CAUSAL TARGETS (Disease-Mod.) | SYMPTOM TARGETS (Supportive)  |  |
|                                | • Initiate ACEi/ARB for BP    | • Oral Phosphate Binder       |  |
|                                | • Target Fasting Glucose <120 | • Oral Sodium Bicarbonate     |  |
|                                +-------------------------------+-------------------------------+  |
+---------------------------------------------------------------------------------------------------+
|  COUNTERFACTUAL TREATMENT SIMULATOR                                                               |
|  Adjust Interventions:                                                                            |
|  [--o-------] Systolic BP Reduction (-20 mmHg)  ──>  Projected Risk: 65.9% (Absolute Delta: -8.3%) |
|  [----o-----] Normalize Fasting Glucose (<110)  ──>  Projected Risk: 69.4% (Absolute Delta: -4.8%) |
|  [Combined Simultaneous Optimization]           ──>  PROJECTED RISK: 61.1% (TOTAL REDUCTION: -13.1%)|
+===================================================================================================+
```
*Figure 6.5: Conceptual Wireframe of the Clinical Web Dashboard and Counterfactual Simulator.*

### Key UI Subsystems
1. **Patient Intake Sidebar:** Streamlined numerical inputs with biological validation bounds to prevent erroneous data entry.
2. **Circular SVG Risk Gauge:** A custom SVG element that animates smoothly from green (low risk) to amber (moderate) and red (severe), displaying calibrated probability.
3. **Dual Recommendation Cards:** Explicitly separates disease-modifying interventions (green card, backed by significant ATEs) from supportive symptom management (amber card, non-causal), preventing therapeutic misdirection.
4. **Interactive Treatment Simulator:** Real-time sliders allowing clinicians to simulate medical interventions and observe projected risk reductions before writing prescriptions.

---

## 6.6 Security-by-Design and Privacy Architecture

Operating on sensitive biomedical data requires adherence to strict healthcare privacy regulations:

1. **Local-First Architecture:** The entire Virtual Biopsy pipeline runs locally on the clinician’s workstation or internal hospital intranet server. No patient telemetry, laboratory readings, or predictive outputs are transmitted across public internet connections.
2. **Pseudonymization and HIPAA Safe Harbor:** Patient records within the SQLite database utilize synthetic alphanumeric primary keys (`PAT-XXXXXX`). Direct identifiers (patient names, social security numbers, precise birth dates) are never stored.
3. **Deterministic Audit Logging:** Every prediction and counterfactual simulation generates a timestamped, immutable record in the `COUNTERFACTUAL_SIMULATIONS` table, ensuring complete traceability for medical-legal review.

By uniting an enterprise-grade multi-tier architecture, an ACID-compliant data model, mathematically rigorous SCM causal equations, and a doctor-friendly presentation layer, this solution design establishes a defensible, reproducible, and clinically viable platform for non-invasive renal biopsy.
