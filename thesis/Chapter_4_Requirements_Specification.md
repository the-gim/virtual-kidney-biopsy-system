# Chapter 4: Requirements Specification

---

## 4.1 Stakeholder Identification and Clinical Use Case Scenarios

The transition of artificial intelligence from theoretical algorithmic design to translational clinical software demands a rigorous, user-centered requirements engineering methodology. In accordance with Robert Gordon University (RGU) guidelines, this chapter details the complete specification of functional and non-functional requirements governing the **Virtual Kidney Biopsy** system.

### 4.1.1 Clinical Stakeholder Analysis
The operational success of a non-invasive diagnostic and causal decision support platform depends on understanding the distinct clinical needs, cognitive workflows, and risk tolerances of its stakeholders:

1. **Consultant Nephrologists (Primary End-Users):** Specialist physicians responsible for diagnosing kidney disease etiology, interpreting renal histopathology, and establishing long-term immunosuppressive, antifibrotic, or cardiovascular therapeutic strategies. Nephrologists require high-confidence pathology risk stratification, algorithmic explainability grounded in pathophysiology, and clear separation between disease-modifying intervention targets and passive symptoms.
2. **Acute Care and Triage Clinicians (Secondary End-Users):** Hospital physicians evaluating patients admitted with acute-on-chronic renal deterioration. These clinicians require rapid, low-latency risk grading from routine emergency blood chemistries without waiting days for specialized histological staining or risking invasive bedside procedures.
3. **Biomedical Researchers and Biostatisticians:** Clinical investigators studying renal disease progression mechanisms and evaluating pharmacological interventions. They require transparent Structural Causal Model (SCM) specifications, verifiable Average Treatment Effect (ATE) calculations, and statistical hypothesis testing outputs.
4. **Chronic Kidney Disease Patients (Ultimate Beneficiaries):** Individuals living with CKD who directly benefit from avoiding invasive surgical needle punctures, eliminating the risk of internal hemorrhage, and receiving causally optimized, personalized therapies.

```
+---------------------------------------------------------------------------------------------------+
|                         CLINICAL STAKEHOLDER WORKFLOW AND INTERACTION                             |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|    [Patient Admission] ───> [Routine Blood Tests & EHR] ───> [Virtual Biopsy Pipeline]            |
|                                                                     │                             |
|                                      ┌──────────────────────────────┴──────────────────────────┐  |
|                                      ▼                                                         ▼  |
|                         [Predictive Track (RQ1)]                                  [Causal Track (RQ2)] |
|                         • 77.4% Pathology Accuracy                                • Structural DAG Engine |
|                         • Damage Grades 0 to 3                                    • 3 Root Causes Isolated|
|                                      │                                                         │  |
|                                      └──────────────────────────────┬──────────────────────────┘  |
|                                                                     ▼                             |
|                                                        [Doctor-Centric Portal]                    |
|                                                        • Patient Intake Form                      |
|                                                        • Dynamic Risk Gauge                       |
|                                                        • Counterfactual Simulator                 |
|                                                                     │                             |
|                                      ┌──────────────────────────────┴──────────────────────────┐  |
|                                      ▼                                                         ▼  |
|                           [Triage Clinician]                                        [Nephrologist]        |
|                           Immediate risk grade;                                     Simulate BP/Glucose   |
|                           Non-invasive triage                                       interventions safely  |
+---------------------------------------------------------------------------------------------------+
```

### 4.1.2 Clinical Use Case Scenarios

#### Use Case Scenario 1: Non-Invasive Kidney Scarring Assessment at Initial Presentation
* **Actor:** Consultant Nephrologist.
* **Pre-condition:** A 58-year-old male with a history of hypertension presents with declining renal function. Coagulation studies indicate elevated bleeding risk, rendering surgical needle biopsy clinically hazardous.
* **Flow of Events:**
  1. The clinician accesses the Virtual Biopsy portal and enters the patient's routine laboratory values (blood glucose, electrolytes, serum albumin, complete blood count) and clinical history.
  2. The system checks input validity against biological plausibility ranges and executes the optimized XGBoost inference pipeline.
  3. The system returns the predicted **Pathological Damage Grade (Grade 2: Moderate Scarring, 26%–50% cortical fibrosis)** along with an estimated severe scarring probability of $74.2\%$.
* **Post-condition:** The clinician establishes a definitive clinical staging without subjecting the patient to needle biopsy risks.

#### Use Case Scenario 2: Counterfactual Treatment Planning (What-If Simulation)
* **Actor:** Consultant Nephrologist.
* **Pre-condition:** A patient is classified with Grade 3 Severe Scarring risk, driven by severe hypertension and fluctuating glycemic control. The clinician wishes to evaluate whether aggressive blood pressure management or oral phosphate binders will yield greater therapeutic benefit.
* **Flow of Events:**
  1. The clinician opens the **Counterfactual Treatment Simulator** tab.
  2. The clinician observes that serum phosphate has an $ATE$ of $0.0000$, signaling that medically lowering phosphate will not alter underlying tissue progression.
  3. The clinician adjusts the interactive Blood Pressure / Hypertension slider, simulating successful pharmacological control ($do(\text{Hypertension} = 0)$).
  4. The causal inference engine re-calculates the counterfactual tissue damage distribution, projecting an absolute risk reduction of $8.28\%$ in severe tissue scarring.
* **Post-condition:** The clinician prescribes targeted antihypertensive therapy rather than ineffective surrogate-targeting medications.

#### Use Case Scenario 3: Longitudinal Disease Monitoring Without Re-Biopsy
* **Actor:** Nephrology Nurse Specialist / Triage Clinician.
* **Pre-condition:** A patient previously staged with Grade 1 Mild Scarring returns for a 6-month outpatient follow-up.
* **Flow of Events:**
  1. The clinician enters the patient's updated 6-month laboratory panels into the system.
  2. The system queries historical records from the local SQLite database and overlays the new risk trajectory.
  3. The system confirms that tissue scarring probability remains stable at $18.4\%$, indicating therapeutic stability without requiring repeat hospitalization.
* **Post-condition:** Longitudinal surveillance is achieved safely, non-invasively, and cost-effectively.

---

## 4.2 Clinical User Personas

To ensure user-centered interface design and appropriate cognitive ergonomics, three archetypal user personas were developed:

### Persona 1: Dr. Eleanor Vance (Consultant Nephrologist)
* **Demographics:** Age 46, Senior Nephrologist at a tertiary academic teaching hospital.
* **Clinical Focus:** Managing complex glomerulonephritis and advanced diabetic kidney disease.
* **Pain Points:** 
  * Hesitates to order surgical biopsies in frail, elderly, or anticoagulated patients due to bleeding hazards.
  * Frustrated by conventional "black-box" AI tools that highlight unhelpful correlations (e.g., advising her that phosphate is high when she already knows the patient is in renal failure).
* **System Goals:** Requires mathematically grounded causal insights; wants to simulate the prognostic impact of blood pressure and glucose normalization before prescribing aggressive medications.

### Persona 2: Dr. Marcus Chen (Acute Medical Registrar)
* **Demographics:** Age 31, Hospital Medical Registrar covering emergency admissions and intensive care units.
* **Clinical Focus:** Rapid triage, stabilization, and bed allocation for acute medical admissions.
* **Pain Points:**
  * Needs instant clinical risk estimates during night shifts when specialized renal biopsy teams and ultrasound operators are unavailable.
  * Overloaded by complex software interfaces with multi-step navigation.
* **System Goals:** Demands a streamlined, zero-latency dashboard where routine blood panel inputs immediately yield an intuitive, color-coded histological risk score.

### Persona 3: Prof. Sarah Jenkins (Clinical Data Scientist & Researcher)
* **Demographics:** Age 39, Principal Investigator in Computational Medicine.
* **Clinical Focus:** Conducting observational studies on MIMIC-IV to identify novel CKD biomarkers.
* **Pain Points:**
  * Wary of data leakage in published clinical AI studies where outcome variables are inadvertently leaked through laboratory ratios.
  * Requires rigorous statistical validation (e.g., McNemar's tests) and transparent causal DAG definitions.
* **System Goals:** Wants to inspect model confusion matrices, review backdoor adjustment criteria, and verify that empirical results match clinical literature.

---

## 4.3 Functional Requirements (FR1 to FR12)

The functional requirements define the core operational behaviors, data transformations, analytical calculations, and user interactions implemented within the Virtual Kidney Biopsy software ecosystem:

*Table 4.1: Comprehensive Functional Requirements Specification.*
| Requirement ID | Requirement Name | Description and Operational Specification | Priority | Code / Module Implementation |
| :--- | :--- | :--- | :---: | :--- |
| **FR-01** | Big Data Cohort Ingestion | The system shall query, filter, and extract Chronic Kidney Disease admissions (ICD-10 `N18` hierarchy) from multi-gigabyte MIMIC-IV Parquet tables using out-of-core SQL execution. | High | `src/01_load_data_duckdb.py` |
| **FR-02** | Data Leakage Prevention | The system shall systematically eliminate all mathematical components of the target KDIGO formula (raw serum creatinine, BUN, and recorded eGFR) prior to feature ranking and model training. | Critical | `src/02_feature_engineering_clean.py` |
| **FR-03** | Adaptive Synthetic Balancing | The system shall mitigate severe training class imbalance by generating synthetic minority examples along difficult decision boundaries using the ADASYN algorithm. | High | `src/02_feature_engineering_clean.py` |
| **FR-04** | Feature Selection & Ranking | The system shall evaluate candidate clinical features using ANOVA F-statistic ranking and isolate the Top 20 most predictive clinical variables. | High | `src/03_feature_analysis.py` |
| **FR-05** | Multi-Model Predictive Inference | The system shall support training, serialization, and real-time execution across six distinct architectures: Decision Trees, SVM, MLP ANN, XGBoost, CNN1D, and LSTM. | Critical | `src/04_train_basic_models.py`<br>`src/05_train_advanced_models.py` |
| **FR-06** | Optimal Decision Thresholding | The system shall calculate optimal classification thresholds by maximizing the F1-score across validation curves, outputting calibrated probability estimates. | High | `src/04_train_basic_models.py` |
| **FR-07** | Non-Parametric Hypothesis Testing | The system shall compute pairwise McNemar's statistical tests across all model prediction vectors, outputting $2 \times 2$ contingency tables, test statistics, and exact $p$-values. | High | `src/06_statistical_testing.py` |
| **FR-08** | Causal DAG Encoding | The system shall formalize a Directed Acyclic Graph (DAG) encoding clinical domain relationships, separating root cause nodes from intermediate states and downstream symptoms. | Critical | `src/07_causal_analysis_complete.py` |
| **FR-09** | Average Treatment Effect Estimation | The system shall execute the Microsoft DoWhy causal pipeline using the Backdoor Adjustment criterion and linear regression estimators to calculate numerical ATEs and $p$-values. | Critical | `src/07_causal_analysis_complete.py` |
| **FR-10** | Causal vs. Symptom Segregation | The system shall mathematically categorize features into True Root Causes ($ATE \neq 0, p < 0.05$) versus Downstream Symptoms ($ATE = 0.0000$). | High | `src/09_causal_vs_noncausal_rq3.py` |
| **FR-11** | Real-Time Counterfactual Simulation | The clinical web portal shall provide interactive sliders allowing clinicians to simulate physiological interventions ($do(X = x)$) and observe real-time projected risk changes. | Critical | `src/10_clinical_dashboard.py` |
| **FR-12** | Local Patient Record Persistence | The system shall persist patient demographic data, laboratory panels, predicted risk grades, and counterfactual simulation logs in a local, ACID-compliant SQLite database. | Medium | `src/10_clinical_dashboard.py` (`virtual_biopsy.db`) |

---

## 4.4 Non-Functional Requirements (NFR1 to NFR8)

Non-functional requirements specify the operational criteria, performance constraints, usability standards, and safety safeguards required for clinical software:

*Table 4.2: Comprehensive Non-Functional Requirements Specification.*
| Requirement ID | Quality Attribute | Target Metric / Benchmark | Verification Methodology |
| :--- | :--- | :--- | :--- |
| **NFR-01** | **Diagnostic Accuracy** | Classification accuracy $> 75.0\%$ and Macro F1-score $> 0.75$ on the held-out test split, significantly exceeding the 62.0% literature baseline (Li et al., 2017). | Held-out empirical evaluation on 1,689 unseen test admissions. |
| **NFR-02** | **Inference Latency** | End-to-end predictive scoring latency $< 500\text{ ms}$; counterfactual recalculation latency $< 1.0\text{ second}$ per patient record. | Execution time benchmarking via Python `timeit` within the Streamlit runtime. |
| **NFR-03** | **Data Ingestion Throughput** | The ETL pipeline must ingest and aggregate $> 10\text{ million}$ raw MIMIC-IV laboratory event rows in $< 30\text{ seconds}$ using $< 2\text{ GB}$ peak memory. | Profiling memory and CPU utilization of DuckDB out-of-core queries. |
| **NFR-04** | **Data Privacy & Governance** | Strict compliance with HIPAA Safe Harbor de-identification standards; zero transmission of Protected Health Information (PHI) to external cloud servers; local-only SQLite storage. | Static code analysis verifying absence of external network calls during patient intake. |
| **NFR-05** | **Algorithmic Interpretability** | Transparent graphical display of the causal Directed Acyclic Graph (DAG) and numerical reporting of ATEs with corresponding 95% confidence intervals and $p$-values. | UI inspection of the Causal Reasoning tab within the clinical portal. |
| **NFR-06** | **Usability & Ergonomics** | Intuitive doctor-friendly interface requiring no command-line interactions; System Usability Scale (SUS) target score $> 80/100$; single-click counterfactual resetting. | Clinical walkthrough evaluation with nephrology clinical scenarios. |
| **NFR-07** | **Portability & Zero Config** | Standalone deployment across standard operating systems (macOS, Linux, Windows 11) using a standard Python 3.11 virtual environment without external database servers. | Multi-platform testing verifying clean execution from a fresh `requirements.txt`. |
| **NFR-08** | **Software Robustness & Safety** | Biological plausibility range checking on all input fields; graceful exception handling preventing system crashes on missing or anomalous inputs. | Automated boundary value testing across extreme physiological parameters. |

---

## 4.5 Operational, Regulatory, and Software as a Medical Device (SaMD) Constraints

Developing software intended for clinical decision support requires navigating strict biomedical regulatory boundaries and ethical governance frameworks:

```
+---------------------------------------------------------------------------------------------------+
|                        REGULATORY AND CLINICAL GOVERNANCE BOUNDARIES                              |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|    [FDA / IMDRF SaMD Framework]                                                                   |
|    • Categorization: Clinical Decision Support (CDS) Software                                     |
|    • Regulatory Status: Non-Device CDS (21 U.S.C. 360bbb-5c)                                      |
|    • Essential Requirement: "Clinician-in-the-Loop"                                               |
|      The software must NOT autonomously execute medical therapy; it must provide transparent,     |
|      auditable evidence allowing the qualified physician to independently review recommendations. |
|                                                                                                   |
|    [Data Governance & Ethics]                                                                     |
|    • Ethics Protocol: Student Project Ethical Review (SPER) Form approved                         |
|    • Training Credential: CITI Program Human Subjects Certification (Record ID: 76046135)         |
|    • Data Use Agreement: PhysioNet MIMIC-IV Research Agreement executed                           |
|    • Re-Identification Prohibition: Zero attempt to re-identify de-identified patient data        |
+---------------------------------------------------------------------------------------------------+
```

### 4.5.1 The "Clinician-in-the-Loop" Principle
Under the International Medical Device Regulators Forum (IMDRF) and United States FDA guidance for *Clinical Decision Support Software* (FDA, 2022), software functions intended to analyze medical data are exempt from high-risk medical device classification **only if the healthcare professional can independently review the basis for the recommendations**. 

The Virtual Kidney Biopsy portal explicitly satisfies this requirement:
* The system does not dictate automated prescribing.
* Instead, it displays the underlying Structural Causal Model, visualizes the causal DAG, reports exact Average Treatment Effects, and presents the counterfactual risk shifts as an advisory simulation, ensuring the clinician maintains full diagnostic autonomy and legal accountability.

### 4.5.2 Ethical Compliance and SPER Alignment
In accordance with Robert Gordon University's ethical review requirements (Section 8 of the guidelines):
* A **Student Project Ethical Review (SPER)** form was executed and approved prior to project initiation.
* Secondary data analysis of the MIMIC-IV clinical repository was conducted under strict compliance with PhysioNet credentialing requirements. The author completed formal human subjects training under the **Collaborative Institutional Training Initiative (CITI Program)**, earning the *Data or Specimens Only Research* credential (Certificate ID: `76046135`, completed March 22, 2026).
* All records stored within the local SQLite database utilize synthetic identifiers (`PAT-XXXXXX`) to ensure zero leakage of protected health information.

By fulfilling these functional, non-functional, and regulatory specifications, the Virtual Kidney Biopsy framework establishes a rigorous, defensible, and clinically safe foundation for empirical implementation.
