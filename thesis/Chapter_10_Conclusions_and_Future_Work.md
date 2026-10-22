# Chapter 10: Conclusions and Future Work

---

## 10.1 Summary of Research Achievements and Contributions

Chronic Kidney Disease represents an escalating global healthcare crisis affecting 850 million individuals worldwide. For decades, accurate evaluation of renal parenchymal scarring has depended on invasive percutaneous surgical needle biopsies—a procedure carrying documented major bleeding risks of 1%–2%, costing $1,000–$5,000 per procedure, and fundamentally incapable of being safely repeated for longitudinal surveillance. Simultaneously, the application of conventional machine learning in nephrology has suffered from the dangerous conflation of statistical correlation with physical causation, often guiding clinicians toward treating non-causal downstream symptom biomarkers.

This dissertation presented **Virtual Kidney Biopsy**, an end-to-end clinical artificial intelligence framework that predicts renal tissue pathology severity directly from routine, non-invasive electronic health records, while employing Judea Pearl’s **Structural Causal Models** to mathematically separate true disease-modifying root causes from downstream symptom biomarkers.

```
+===================================================================================================+
|                              SYNTHESIS OF CORE RESEARCH CONTRIBUTIONS                             |
+===================================================================================================+
|                                                                                                   |
|  [RQ1: PREDICTIVE FEASIBILITY]                                                                    |
|  • Proved non-invasive blood biomarkers can predict renal pathology severity without needle biopsy|
|  • Achieved 77.42% accuracy and 0.7732 Macro F1 with XGBoost (SVM: 77.31%, CNN1D: 76.50%)        |
|  • Significantly outperformed the 62.0% literature baseline of Li et al. (2017) (p < 10^-15)      |
|                                                                                                   |
|  [RQ2: CAUSAL DISCOVERY & ATTRIBUTION]                                                            |
|  • Built clinically grounded DAG and executed DoWhy SCM using Backdoor Adjustment criterion       |
|  • Mathematically proved 17 of top 20 "predictive" features are non-causal symptoms (ATE = 0.0)   |
|  • Isolated 3 true causal root causes: Hypertension (-0.0828), Glucose (-0.0483), Gender (-0.0512) |
|                                                                                                   |
|  [RQ3: CAUSAL VS. NON-CAUSAL CLINICAL PARADIGM]                                                  |
|  • Resolved the medical AI dilemma: Correlational AI is optimal for diagnostic triage (77.4% acc)|
|    while Causal AI is indispensable for therapeutic planning (96.24% recall on true etiology)    |
|                                                                                                   |
|  [TRANSLATIONAL CLINICAL PROTOTYPE]                                                               |
|  • Engineered doctor-friendly Streamlit web portal with integrated SQLite persistent database     |
|  • Implemented real-time Counterfactual Treatment Simulator enabling "what-if" risk previews     |
+===================================================================================================+
```
*Figure 10.1: Summary Matrix of Empirical and Engineering Contributions.*

The specific achievements of this research span four foundational domains:
1. **Surpassing Published Literature Benchmarks (RQ1):** Benchmarking six foundational and deep learning architectures on 8,441 CKD admissions from MIMIC-IV demonstrated that regularized gradient boosting (XGBoost) achieves **77.42% accuracy, an F1-score of 0.7732, and an AUROC of 0.8521**, outperforming the 62.0% baseline of Li et al. (2017) by over 15%. Non-parametric McNemar hypothesis testing established the statistical parity between tree ensembles and kernelized SVMs ($p = 0.0545$) and their collective superiority over deep convolutional networks ($p = 0.0325$).
2. **Empirical Refutation of Correlational Feature Importance (RQ2):** Utilizing Microsoft DoWhy and the Backdoor Adjustment criterion, this research provided quantitative proof that **17 of the top 20 machine learning features (e.g., serum phosphate and potassium aggregates) are non-causal symptoms with an ATE of precisely 0.0000**. The analysis isolated three definitive, modifiable etiologic drivers: **Hypertension ($ATE = -0.0828, p < 0.001$)**, **Minimum Fasting Glucose ($ATE = -0.0483, p < 0.001$)**, and **Biological Gender ($ATE = -0.0512, p < 0.001$)**.
3. **Resolution of the Medical AI Dilemma (RQ3):** By establishing a comparative framework between associative models (20 features) and causal models (3 root causes), this research demonstrated that while correlational models are ideal for diagnostic triage, causal AI is essential for treatment planning, achieving **96.24% recall on true disease etiology** while eliminating non-causal surrogate distractions.
4. **Translational Clinical Engineering:** Fulfilling RGU module requirements for a fully functional software prototype, a doctor-centric web portal was engineered using Python, Streamlit, DuckDB, and SQLite. The portal translates abstract causal *do*-calculus into an intuitive **Counterfactual Treatment Simulator**, allowing nephrologists to simulate physiological interventions (e.g., blood pressure reduction) and observe projected risk reductions before clinical prescription.

---

## 10.2 Critical Reflection on Project Limitations

In accordance with RGU Guideline 13, critical scientific reflection requires identifying the methodological, observational, and computational constraints of this investigation:

1. **Intensive Care Unit Selection Bias:**
   The empirical models were trained and validated on the MIMIC-IV clinical database. Because MIMIC-IV captures patients admitted to critical care and emergency departments of an academic tertiary medical center, the cohort exhibits higher acute comorbidity, multiorgan failure incidence, and laboratory variability than outpatient community CKD cohorts. While the models demonstrate strong internal validity, their external validity on ambulatory, early-stage (Stages G1–G2) outpatients requires further clinical verification.
2. **Retrospective Observational Confounding:**
   Although Pearl's Structural Causal Models and Backdoor Adjustment criteria mathematically control for observed confounders encoded in the Directed Acyclic Graph, observational causal inference relies on the assumption of **no unobserved confounding** (unconfoundedness). Unrecorded variables in MIMIC-IV—such as over-the-counter non-steroidal anti-inflammatory drug (NSAID) consumption, genetic polymorphisms (e.g., *APOL1* risk alleles), and detailed dietary sodium intake—could introduce residual unmeasured confounding.
3. **Surrogate Target Formulation via eGFR:**
   Due to the absence of digitized, stained microscopic whole slide histopathology images matched to individual patient admission IDs in open-access intensive care repositories, ground-truth histological tissue damage was operationalized using international KDIGO guidelines via the CKD-EPI 2021 formula. While eGFR correlates strongly with tubulointerstitial fibrosis (IFTA grade) in medical literature, direct physical tissue validation remains necessary.
4. **Single-Center Database Origin:**
   MIMIC-IV is curated exclusively from Beth Israel Deaconess Medical Center in Boston, Massachusetts, USA. Regional clinical protocols, local patient demographics, and hospital-specific laboratory assay machines may influence biomarker distributions, requiring multi-center external validation.

---

## 10.3 Ways in Which the Work Could Have Been Improved

In accordance with RGU Guideline 13 (*ways in which the work could have been improved*), several technical and analytical enhancements could be incorporated in subsequent iterations:

1. **Integration of Outpatient and Community Cohorts:**
   Incorporating ambulatory community datasets—such as the **UK Biobank** or the **Chronic Renal Insufficiency Cohort (CRIC) Study**—would balance the intensive care bias of MIMIC-IV, extending the model's predictive precision into early, asymptomatic stages of renal scarring.
2. **Continuous-Time Longitudinal Survival Modeling:**
   Rather than formulating disease progression as a binary or 4-grade classification task, the analytical pipeline could be expanded into dynamic, continuous-time survival analysis (e.g., DeepSurv neural networks or dynamic Cox proportional hazards). This would allow the system to predict the exact **time-to-end-stage renal disease (ESRD)** and estimated years of dialysis-free life preserved under counterfactual interventions.
3. **Natural Language Processing (NLP) of Clinical Narratives:**
   MIMIC-IV contains rich, unstructured clinical discharge summaries and nursing progress notes. Implementing clinical large language models (e.g., GatorTron or Clinical-BERT) to extract qualitative lifestyle risk factors (smoking history, medication adherence, dietary habits) would enrich the tabular feature matrix and reduce unobserved confounding.
4. **Multi-Model Causal Ensembles:**
   While DoWhy linear backdoor regression provides transparent, statistically robust ATE estimates, future work could implement double machine learning (DML) via Microsoft EconML, utilizing non-parametric causal forests to estimate heterogeneous treatment effects (HTE) personalized to individual patient sub-phenotypes.

---

## 10.4 Strategic Considerations for Future Developments and Clinical Trials

To advance the Virtual Kidney Biopsy framework from an academic prototype to an accredited, hospital-integrated clinical decision support system, the following roadmap is proposed:

```
+===================================================================================================+
|                          STRATEGIC ROADMAP FOR FUTURE CLINICAL TRANSLATION                        |
+===================================================================================================+
|                                                                                                   |
|    PHASE 1: MULTI-MODAL FUSION                                                                    |
|    • Integrate non-invasive imaging: Renal DWI MRI, T1-mapping, Ultrasound Shear-Wave Elastography|
|    • Multi-modal neural architecture fusing blood biomarkers with radiological features           |
|                                                                                                   |
|    PHASE 2: PROSPECTIVE MULTI-CENTER CLINICAL TRIALS                                              |
|    • Non-interventional prospective observational trial across 3 academic teaching hospitals     |
|    • Validate predicted Damage Grades directly against surgical biopsy specimens in consenting pts|
|                                                                                                   |
|    PHASE 3: ENTERPRISE EHR DEPLOYMENT (HL7 / FHIR)                                                |
|    • Package pipeline as an accredited SaMD microservice (FastAPI / Docker)                       |
|    • Direct integration into hospital electronic health records (Epic Systems, Cerner / Oracle)   |
|                                                                                                   |
|    PHASE 4: DYNAMIC REINFORCEMENT LEARNING FOR CHRONIC CARE                                       |
|    • Formulate Dynamic Treatment Regimens (DTR) using Reinforcement Learning from Causal Feedback |
|    • Optimize sequential medication titration schedules over 5- to 10-year disease horizons       |
+===================================================================================================+
```
*Figure 10.2: Four-Phase Strategic Roadmap for Future Technical and Clinical Developments.*

1. **Multi-Modal Diagnostic Fusion:**
   The accuracy of the Virtual Biopsy framework can be elevated further by fusing tabular blood biomarkers with non-contrast radiological imaging, specifically **Diffusion-Weighted Magnetic Resonance Imaging (DWI-MRI)**, renal cortical T1-mapping, and ultrasound shear-wave elastography. Combining deep spatial image representations of renal stiffness with our causal tabular biomarkers represents the ultimate non-invasive virtual biopsy paradigm.
2. **Prospective Clinical Trial Validation:**
   The immediate next translational milestone is a prospective, non-interventional observational study in an outpatient nephrology clinic. For patients already scheduled for a medically indicated surgical kidney biopsy, routine blood panels will be fed into the Virtual Biopsy portal prior to the procedure. The predicted damage grades (Levels 0–3) will be blinded and statistically matched against the physical histopathologist's IFTA fibrosis score, establishing formal prospective diagnostic concordance.
3. **Enterprise EHR Integration via HL7/FHIR:**
   To embed the system into routine hospital operations, the software prototype should be packaged as an enterprise microservice containerized via Docker and orchestrated via Kubernetes. By implementing **HL7 FHIR (Fast Healthcare Interoperability Resources)** RESTful APIs, the tool can automatically ingest laboratory panels from commercial hospital information systems (Epic Systems, Oracle Health/Cerner) and display the SVG risk gauge directly within the electronic chart.
4. **Dynamic Treatment Regimens via Causal Reinforcement Learning:**
   Extending counterfactual simulation into longitudinal therapeutic management, future research will explore **Causal Reinforcement Learning** to construct Dynamic Treatment Regimens (DTR). By modeling CKD progression as a Markov Decision Process governed by our structural causal graph, the system will recommend sequential, multi-year medication titration strategies that maximize dialysis-free survival while minimizing medication-induced adverse events.

### Concluding Remarks
The Virtual Kidney Biopsy framework demonstrates that artificial intelligence in medicine can transcend the limitations of associative correlation. By uniting robust machine learning, rigorous target leakage elimination, Judea Pearl's Structural Causal Models, and an intuitive clinical software prototype, this dissertation proves that non-invasive, biologically grounded, and causally actionable decision support is both technically feasible and clinically transformative. This work lays the foundation for a safer, more equitable, and proactive era of nephrological care.
