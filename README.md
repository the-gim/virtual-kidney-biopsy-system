# Virtual Kidney Biopsy: Non-Invasive Histopathology Assessment via Causal Machine Learning

> **MSc in Big Data Analytics Dissertation & Clinical Prototype**  
> **School of Computing, Robert Gordon University (Aberdeen, UK) & Informatics Institute of Technology (Sri Lanka)**  
> **Student:** Gimhani Wijesuriya (Student ID: 20241681 / RGU: 2427771)  
> **Module:** CMM799 MSc Project

---

## 🔬 Project Overview

Percutaneous kidney biopsy is the gold standard for diagnosing renal parenchymal scarring (Interstitial Fibrosis and Tubular Atrophy - IFTA, and Glomerulosclerosis). However, invasive biopsy entails life-threatening risks (e.g., severe perinephric hematoma in 11–13% of cases) and cannot be repeated longitudinally.

This research introduces a **Non-Invasive Virtual Kidney Biopsy System** that:
1. **Eliminates Circular Data Leakage:** Prunes direct deterministic eGFR equations (Serum Creatinine, BUN) and isolates 20 aggregate routine non-invasive biomarkers from the MIMIC-IV clinical database (8,000+ patient cohort).
2. **Predictive Feasibility (RQ1):** Trains, optimizes, and statistically benchmarks machine learning and deep learning models (Regularized XGBoost, RBF-SVM, 1D-CNN, Bidirectional LSTM) achieving superior diagnostic accuracy (**F1: 0.7732, AUROC: 0.865**) with non-parametric McNemar validation ($p < 0.001$).
3. **Causal Discovery & Attribution (RQ2):** Employs Judea Pearl's Structural Causal Models (SCM) via Microsoft DoWhy to mathematically separate **true etiologic root causes** (Hypertension $\text{ATE} = -0.0828$, Glucose $\text{ATE} = -0.0483$) from 17 misleading **downstream symptom biomarkers** ($\text{ATE} = 0.0000$).
4. **Empirical Evaluation (RQ3):** Resolves the tension between correlational AI (20 features, optimal diagnostic staging) and causal AI (3 etiology features, 96.2% interventional targeting).
5. **Translational Clinical Decision Support System (CDSS):** An interactive Streamlit portal with real-time feature derivation, a counterfactual treatment simulator, and a persistent SQLite longitudinal registry.

---

## 🗂️ Repository Structure

```
├── src/                                  # Production Source Code
│   ├── 01_data_cleaning.py               # MIMIC-IV cohort extraction & leakage control
│   ├── 02_data_transformation.py         # Standardized scaling, encoding, and train-test splits
│   ├── 03_feature_analysis.py            # Feature distribution & correlation analysis
│   ├── 04_feature_selection.py           # 20-feature non-leaky biomarker selection
│   ├── 05_model_training.py              # ML/DL training pipeline (XGBoost, SVM, CNN, LSTM)
│   ├── 06_statistical_testing.py         # Pairwise McNemar statistical hypothesis tests
│   ├── 07_causal_analysis_complete.py    # Structural Causal Model (DAG) formulation
│   ├── 07b_causal_graphs_individual.py   # Publication-grade DAG visualizer
│   ├── 08_causal_effect_estimation.py    # DoWhy Backdoor Linear Regression ATE estimation
│   ├── 09_causal_vs_noncausal_rq3.py     # RQ3 empirical comparative evaluation
│   ├── 10_clinical_dashboard.py          # Interactive Clinical Decision Support Portal
│   └── db_manager.py                     # SQLite longitudinal clinical database engine
│
├── outputs/                              # Validated Research Artifacts
│   ├── models/                           # Serialized models (XGBoost JSON, scaler parameters)
│   ├── causal_results/                   # Causal DAG graphs, ATE estimations, and treatable targets
│   ├── figures/                          # AUROC, AUPRC, confusion matrices, and architecture diagrams
│   └── results/                          # McNemar statistical tables & benchmark CSV summaries
│
├── thesis/                               # MSc Dissertation Writing (Aligned with RGU Guidelines)
│   ├── 00_Front_Matter.md                # Title, Declaration, Abstract, Table of Contents
│   ├── Chapter_1_Introduction.md         # Context, motivation, aim, and objectives
│   ├── Chapter_2_Literature_Review.md    # Nephrology, invasive biopsy, and AI in medicine
│   ├── Chapter_3_Software_Tools.md       # Comparative evaluation of frameworks & algorithms
│   ├── Chapter_4_Requirements.md         # Functional, non-functional, and data specifications
│   ├── Chapter_5_Research_Methodology.md # End-to-end experimental design & causal methodology
│   ├── Chapter_6_Solution_Design.md      # Architectural design, UML, and data flow diagrams
│   ├── Chapter_7_Implementation.md       # Code implementation, data structures, and pipeline
│   ├── Chapter_8_Results_Evaluation.md   # Empirical findings, McNemar tests, and causal findings
│   ├── Chapter_9_Ethical_Issues.md       # CITI certification, PhysioNet DUA, and ethics
│   ├── Chapter_10_Conclusions.md         # Summary of contributions, limitations, future work
│   ├── References.md                     # Comprehensive Harvard-formatted bibliography
│   ├── Appendices.md                     # Code listings, hyperparameter tables, statistical logs
│   ├── Virtual_Kidney_Biopsy_Full.docx   # Complete compiled Word dissertation
│   └── convert_to_word.sh                # Automated pandoc compilation script
│
├── docs/                                 # Research documentation & planning protocols
├── archive/                              # Legacy baseline experiments & exploratory scripts
├── requirements.txt                      # Python dependencies
└── README.md                             # Project overview and instructions
```

---

## 🚀 Quickstart & Interactive Clinical Portal

### Prerequisites
* Python 3.10 or 3.11
* Virtual environment recommended

### Installation
```bash
# Clone the repository
git clone https://github.com/the-gim/virtual-kidney-biopsy-system.git
cd virtual-kidney-biopsy-system

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Launch the Clinical Portal
```bash
streamlit run src/10_clinical_dashboard.py
```
Open your browser at `http://localhost:8501`.

---

## 🩺 Clinical Portal Features
* **Live Feature Auto-Derivation:** Clinicians enter standard routine blood tests (Phosphate, Potassium, Bicarbonate, Albumin, Calcium, Sodium, Glucose, Blood Pressure), while all 20 statistical aggregates (mean, min, max, std) are automatically computed and visually tracked in real-time.
* **Non-Invasive Histopathology Staging:** Classifies renal parenchymal injury into 4 damage tiers (Level 0: Preserved to Level 3: Critical Fibrosis) with estimated chronic scarring (% IFTA).
* **Counterfactual Treatment Simulator:** Interactive simulator powered by Microsoft DoWhy ATE calculations, allowing doctors to simulate the direct causal benefit of achieving blood pressure control ($-8.3\%$ risk reduction) and glycemic normalization ($-4.8\%$ risk reduction).
* **Longitudinal Patient Registry:** Full SQLite database integration to save, review, search, export to CSV, or manage patient biopsy records.

---

## 📜 Ethical Compliance & Data Privacy
This study utilizes de-identified electronic health records from the **MIMIC-IV v3.1** database provided by the PhysioNet consortium under human subjects research credentialing (CITI Certification ID: 76046135). In strict accordance with the PhysioNet Data Use Agreement (DUA) and HIPAA Safe Harbor privacy provisions, raw patient data files are excluded from version control.

---

## 📄 License & Attribution
Academic research project developed for the MSc in Big Data Analytics, Robert Gordon University. Confidential.
