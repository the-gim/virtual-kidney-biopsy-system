# Front Matter

---

## Title Page (RGU Specification 14.2)

<br><br><br>

# MSc Project Report

<br>

# Virtual Kidney Biopsy: Predicting Tissue Pathology from Clinical Data Using Causal Artificial Intelligence

<br><br>

### Gimhani Wijesuriya
**IIT Student ID:** 20241681  
**RGU Student ID:** 2506635  
**Course Code:** CMM799 (60 Credits)

<br>

### 2026

<br><br>

**Supervised by:**  
Dr. Prasan Yapa  
School of Computing  
Informatics Institute of Technology (IIT), Sri Lanka  
in collaboration with  
Robert Gordon University, Aberdeen, Scotland

<br><br>

*A report submitted as part of the requirements for the degree of*  
**MSc Big Data Analytics at Robert Gordon University, Aberdeen, Scotland**

<br><br><br>

---

\newpage

## Declaration (RGU Specification 14.3)

<br><br>

### Declaration

I confirm that the work contained in this MSc project report has been composed solely by myself and has not been accepted in any previous application for a degree. All sources of information have been specifically acknowledged and all verbatim extracts are distinguished by quotation marks.

<br><br><br>

**Signed:** ………………………………………………………………………………  

**Date:** ………………………………………………………………………………

<br><br>

*(Gimhani Wijesuriya)*  
*MSc in Big Data Analytics*  
*Robert Gordon University / Informatics Institute of Technology*

---

\newpage

## Acknowledgements

I would like to express my deepest and most sincere gratitude to my project supervisor, **Dr. Prasan Yapa**, for his invaluable guidance, continuous intellectual mentorship, and constructive feedback throughout the inception, analytical formulation, and writing of this MSc dissertation. His critical insights into both artificial intelligence methodologies and academic rigor were foundational to the successful realization of this research.

I extend my appreciation to the academic and administrative staff at the **Informatics Institute of Technology (IIT), Sri Lanka**, and **Robert Gordon University (RGU), Aberdeen, Scotland**, for providing a rich learning environment, high-performance computing resources, and administrative support across the MSc Big Data Analytics programme.

I gratefully acknowledge the **Massachusetts Institute of Technology (MIT) Laboratory for Computational Physiology** and **PhysioNet** for maintaining and granting open-access research permissions to the MIMIC-IV clinical database under rigorous data governance protocols.

Special thanks are also due to the clinical domain experts and nephrology advisors who generously shared their clinical perspectives on renal histopathology, biopsy procedures, and clinical decision workflows, which were instrumental in framing the practical utility of the Virtual Biopsy system.

Finally, I am profoundly indebted to my family and friends for their enduring patience, encouragement, and unwavering support throughout this demanding academic endeavor.

---

\newpage

## Abstract

*(Formatted in single-line spacing in accordance with RGU Document Specification 14.4)*

Chronic Kidney Disease (CKD) affects over 850 million individuals globally, with histopathological evaluation of renal parenchymal scarring and cellular damage representing the diagnostic gold standard for staging disease progression and tailoring therapeutic regimens. However, clinical assessment currently relies on invasive percutaneous surgical needle biopsies, which carry documented major complication risks of 1–2% (including life-threatening hemorrhage and organ loss), cost $1,000–$5,000 per procedure, and cannot be safely repeated for longitudinal monitoring. Furthermore, while conventional machine learning models can classify clinical risk, they operate purely as correlation-seeking engines. In healthcare, strong statistical correlations frequently mirror downstream symptoms or reverse causality (such as hyperphosphatemia and hyperkalemia) rather than true biological root causes, potentially misleading clinicians into treating non-causal surrogate markers.

To overcome these dual clinical and computational limitations, this dissertation presents **Virtual Kidney Biopsy**, an end-to-end clinical artificial intelligence framework that predicts renal tissue damage severity directly from routine, non-invasive electronic health records and standard blood tests, while employing **Causal Artificial Intelligence** to distinguish true disease-modifying root causes from downstream symptom biomarkers. 

Using an ethically governed cohort of 8,441 CKD admissions from the MIMIC-IV v3.1 database (CITI Program Certification #76046135), ground-truth tissue pathology was operationalized via international KDIGO clinical guidelines and the CKD-EPI formula, stratified into binary damage thresholds and multi-class pathological grades (Levels 0–3). To guarantee methodological rigor and prevent data leakage, all direct mathematical components of the target equation (serum creatinine, BUN, and recorded eGFR) were systematically pruned. The remaining features were prioritized using ANOVA F-statistic ranking, isolating the Top 20 clinical predictors. Training data class imbalance was resolved using Adaptive Synthetic (ADASYN) sampling. 

Across six benchmarked predictive architectures (RQ1), Extreme Gradient Boosting (**XGBoost**) achieved the highest diagnostic performance with **77.42% accuracy, an F1-score of 0.7732, and an AUROC of 0.8521**, significantly outperforming the 62% literature baseline established by Li et al. (2017). Non-parametric McNemar hypothesis tests demonstrated statistical parity across top ensemble and deep learning models (SVM: 77.31%, CNN1D: 76.50%, LSTM: 75.12%). 

To address causal attribution (RQ2), a clinically grounded Directed Acyclic Graph (DAG) was constructed, and Structural Causal Modeling (SCM) was executed via the Microsoft DoWhy library using the Backdoor Adjustment criterion. The causal engine quantitatively proved that **17 out of the top 20 "predictive" features are non-causal downstream symptoms** with an Average Treatment Effect ($ATE$) of $0.0000$. Conversely, the framework isolated three definitive, modifiable root causes: **Hypertension ($ATE = -0.0828, p < 0.001$)**, **Minimum Blood Glucose ($ATE = -0.0483, p < 0.001$)**, and **Biological Gender ($ATE = -0.0512, p < 0.001$)**. 

Comparative evaluation (RQ3) revealed that while correlational models deliver optimal diagnostic discrimination for disease triage, causal AI is essential for therapeutic planning, providing 96.24% recall on actionable etiology without confounding bias. Finally, to fulfill RGU module requirements for a fully functional software prototype, a production-grade, doctor-friendly **Virtual Kidney Biopsy Portal** was implemented using Python, Streamlit, DuckDB, and SQLite. The portal integrates real-time patient risk grading with an interactive **Counterfactual Treatment Simulator**, allowing nephrologists to simulate hypothetical physiological interventions (e.g., blood pressure reduction) and preview projected risk reductions prior to prescribing therapy. This research demonstrates the feasibility of safe, non-invasive, and causally grounded decision support for modern nephrology.

**Keywords:** Virtual Kidney Biopsy; Chronic Kidney Disease; Causal Artificial Intelligence; Structural Causal Models; DoWhy; Machine Learning; Deep Learning; Counterfactual Simulation; Medical Decision Support.

---

\newpage

## Table of Contents

* **Front Matter**
  * Title Page ....................................................................................................... i
  * Declaration .................................................................................................... ii
  * Acknowledgements ......................................................................................... iii
  * Abstract ........................................................................................................ iv
  * Table of Contents ........................................................................................... v
  * List of Figures ............................................................................................... vii
  * List of Tables ................................................................................................. viii
  * List of Abbreviations ....................................................................................... ix

* **Chapter 1: Introduction and Motivation** ............................................................. 1
  * 1.1 Clinical Background and Problem Domain ...................................................... 1
  * 1.2 Surgical Biopsy: Gold Standard, Hazards, and Limitations ............................... 3
  * 1.3 The Research Problem: The Correlation vs. Causation Dilemma in Medical AI ....... 5
  * 1.4 Proposed Solution: Virtual Kidney Biopsy Framework .................................... 7
  * 1.5 Research Aim and Detailed Objectives .......................................................... 9
  * 1.6 Research Questions (RQ1, RQ2, RQ3) ........................................................... 11
  * 1.7 Scope, Assumptions, and Boundary Conditions ............................................. 13
  * 1.8 Contributions to Knowledge and Clinical Practice ......................................... 15
  * 1.9 Dissertation Structure and Overview ........................................................... 17

* **Chapter 2: Literature Review and Related Work** ................................................. 19
  * 2.1 Renal Pathophysiology and Histopathology of Tissue Scarring .......................... 19
  * 2.2 Percutaneous Needle Biopsies: Clinical Evidence and Morbidity ........................ 23
  * 2.3 Non-Invasive Biomarkers and Previous Machine Learning Benchmarks .............. 26
  * 2.4 Fundamental Pitfalls of Associative Machine Learning in Healthcare ................ 31
  * 2.5 Causal Artificial Intelligence, Structural Causal Models, and *do*-Calculus ............ 35
  * 2.6 Critical Gap Analysis and Synthesis ............................................................. 40

* **Chapter 3: Software Tools and Technology Evaluation** ....................................... 43
  * 3.1 Overview of Computational Requirements .................................................... 43
  * 3.2 Evaluation and Selection of Programming Environments (Python vs. R) ............ 45
  * 3.3 Evaluation of Big Data Ingestion Technologies (DuckDB vs. PostgreSQL/Pandas) .. 48
  * 3.4 Evaluation of Machine Learning & Deep Learning Frameworks .......................... 51
  * 3.5 Evaluation of Causal Inference Frameworks (DoWhy vs. CausalNex) ................. 54
  * 3.6 Evaluation of Web Deployment & Persistence Frameworks (Streamlit & SQLite) ... 57
  * 3.7 Technology Stack Summary ........................................................................ 60

* **Chapter 4: Requirements Specification** ............................................................ 62
  * 4.1 Stakeholder Identification and Clinical Use Case Scenarios ............................. 62
  * 4.2 Clinical User Personas (Nephrologist, Triage Clinician, Researcher) .................. 65
  * 4.3 Functional Requirements (FR1 to FR12) ....................................................... 68
  * 4.4 Non-Functional Requirements (NFR1 to NFR8: Accuracy, Safety, Usability) .......... 72
  * 4.5 System and Regulatory Constraints (HIPAA, SaMD Guidelines) .......................... 76

* **Chapter 5: Research Methodology and Experimental Design** ............................... 78
  * 5.1 Methodological Workflow and Philosophical Approach .................................... 78
  * 5.2 Cohort Identification and Data Extraction (MIMIC-IV v3.1) ............................... 81
  * 5.3 Ground Truth Operationalization (KDIGO Staging & eGFR Formulations) ............. 85
  * 5.4 Data Preprocessing, Imputation, and Mathematical Leakage Elimination ............. 89
  * 5.5 Class Imbalance Resolution via Adaptive Synthetic (ADASYN) Sampling ............ 93
  * 5.6 Feature Selection Pipeline and Candidate Ranking (Top 20 Features) ................ 96
  * 5.7 Predictive Modeling Architectures and Training Setup (RQ1) ............................ 99
  * 5.8 Structural Causal Graph Formulation and Effect Estimation (RQ2) ................... 104
  * 5.9 Comparative Framework: Causal vs. Non-Causal Feature Sets (RQ3) ................ 109
  * 5.10 Statistical Hypothesis Testing Methodology (McNemar’s Test) ......................... 112

* **Chapter 6: Architectural and Solution Design** .................................................... 115
  * 6.1 High-Level Multi-Tier Architectural Design .................................................... 115
  * 6.2 Data Model and Schema Architecture (DuckDB & SQLite) ................................ 119
  * 6.3 Structural Causal Model DAG Architecture .................................................... 123
  * 6.4 Model Pipeline Design and Integration Interfaces .......................................... 127
  * 6.5 Clinical User Interface and Interaction Architecture ....................................... 130
  * 6.6 Privacy-Preserving and Security-by-Design Architecture ................................. 134

* **Chapter 7: System Implementation** .................................................................. 136
  * 7.1 Computing Environment, Infrastructure, and Facilities .................................... 136
  * 7.2 Big Data Extraction and Ingestion Engine Implementation ............................... 139
  * 7.3 Data Preprocessing and Feature Pipeline Implementation ................................ 142
  * 7.4 Predictive Model Training and Hyperparameter Tuning Implementation ............. 146
  * 7.5 Causal Inference and Backdoor Engine Implementation ................................... 151
  * 7.6 Doctor-Centric Web Prototype Implementation (`10_clinical_dashboard.py`) ........ 155
  * 7.7 Counterfactual Simulation Engine Implementation ......................................... 159

* **Chapter 8: Results and Critical Evaluation** ........................................................ 163
  * 8.1 Empirical Results for Predictive Modeling (RQ1) ............................................ 163
  * 8.2 Statistical Validation and McNemar Significance Testing ................................. 168
  * 8.3 Empirical Results for Causal Discovery and ATE Estimation (RQ2) .................... 172
  * 8.4 Empirical Results for Causal vs. Non-Causal Comparison (RQ3) ....................... 177
  * 8.5 Critical Comparison Between Original Objectives and Achieved Outcomes .......... 182
  * 8.6 Critical Evaluation of Efficacy, Performance, Efficiency, and Cost-Benefit ......... 186
  * 8.7 Critical Comparison with Documented Literature Baselines ............................. 191

* **Chapter 9: Professional, Legal, Social and Ethical Issues** ................................... 195
  * 9.1 Professional Responsibility and Clinical Accountability in SaMD ....................... 195
  * 9.2 Legal Frameworks, Data Protection, and HIPAA Compliance ............................. 198
  * 9.3 Research Ethics Governance and SPER Form Compliance ................................. 201
  * 9.4 Algorithmic Bias, Demographic Fairness, and Health Equity ............................. 204
  * 9.5 Broader Social and Healthcare Economic Implications ..................................... 207

* **Chapter 10: Conclusions and Future Work** ......................................................... 210
  * 10.1 Synthesis of Completed Work and Contributions ........................................... 210
  * 10.2 Critical Reflection on Project Limitations ................................................... 213
  * 10.3 Potential Improvements to the Current Solution .......................................... 216
  * 10.4 Strategic Directions for Future Research and Clinical Trials ........................... 218

* **References** (Harvard Referencing Style) ............................................................ 221

---

\newpage

## Appendices

* **Appendix A: Project Plan and Resource Review** (~1,000 words, ~3 pages, Gantt chart, rationale, resources) ............................................................................ 235
* **Appendix B: Detailed Design Documentation** (UML diagrams, database DDL schema, structural equations) ................................................................................. 241
* **Appendix C: Comprehensive Test Data and Empirical Results** (Confusion matrices, classification reports, McNemar tables, full ATE data) ....................................... 248
* **Appendix D: Virtual Biopsy Portal Clinician User Guide** (Step-by-step walkthrough, screenshots, interpretation guide) .................................................................... 257
* **Appendix E: Software Installation and Deployment Guide** (System requirements, virtual environment, running pipeline & Streamlit) ............................................... 264
* **Appendix F: Student Project Ethical Review (SPER) Form Summary** (Ethical clearance documentation, CITI certificate) ........................................................................ 270

---
