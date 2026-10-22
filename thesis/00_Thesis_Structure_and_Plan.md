# MSc Project Report: Master Thesis Structure & Roadmap
## Aligned with Robert Gordon University (RGU) / IIT MSc Project Guidelines (Version 1.0)

**Programme:** MSc in Big Data Analytics  
**Course Code:** CMM799 (60 Credits)  
**Institutions:** Informatics Institute of Technology (IIT) in collaboration with Robert Gordon University (RGU), Aberdeen, Scotland  
**Student Name:** Gimhani Wijesuriya  
**IIT Student ID:** 20241681  
**RGU Student ID:** 2506635  
**Supervisor:** Dr. Prasan Yapa  
**Project Title:** *Virtual Kidney Biopsy: Predicting Tissue Pathology from Clinical Data Using Causal Artificial Intelligence*  
**Prototype:** Doctor-Centric Virtual Kidney Biopsy Portal (`src/10_clinical_dashboard.py` running on Streamlit with SQLite & DuckDB)

---

## 1. Compliance Matrix with RGU MSc Project Guidelines (Section 14 & Deliverables)

The table below establishes the direct, 1-to-1 mapping between the **15 Document Specifications of RGU Section 14.4** and the chapters/appendices of this dissertation:

| RGU Guideline Specification (Section 14.4) | Dissertation Chapter / Section | Target Page Range | Current Status | Primary Code / Artifact Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **Front Matter** (Title Page, Declaration, Acknowledgements, Abstract, TOC) | Front Matter (`00_Front_Matter.md`) | Pages i–viii | Ready to finalize | Exact RGU wording (Sec 14.2 & 14.3) |
| **1. Introduction and motivation** | Chapter 1: Introduction & Motivation | ~8–10 pages | **Drafted** | Clinical context, surgical biopsy risks, virtual biopsy motivation |
| **2. Review of relevant literature** | Chapter 2: Literature Review & Related Work | ~15–18 pages | Next to draft | Histopathology, Li et al. (2017) baseline, Causal AI, SCM, Pearl's *do*-calculus |
| **3. Discussion and evaluation of software tools to be used** | Chapter 3: Software Tools & Technology Evaluation | ~6–8 pages | Ready to draft | Python, PyTorch, Scikit-Learn, XGBoost, DoWhy, DuckDB, Streamlit, SQLite |
| **4. Problems identified and proposed solution(s)** | Chapter 1 (Sec 1.2–1.3) & Chapter 4 (Sec 4.1) | Integrated | **Drafted** | Correlational conflation, surgical biopsy risks, dual-track causal framework |
| **5. Specification of problem, aims, and objectives** | Chapter 1 (Sec 1.4–1.6) | Integrated | **Drafted** | Detailed specification, alternative interpretations, RQ1, RQ2, RQ3 |
| **6. List of functional and non-functional requirements** | Chapter 4: Requirements Specification | ~8–10 pages | Ready to draft | FR1–FR12, NFR1–NFR8, clinical user personas, SaMD usability requirements |
| **7. Approach / Methodology adopted** | Chapter 5: Research Methodology | ~20–25 pages | **Completed** | MIMIC-IV cohort, KDIGO eGFR target, ADASYN, Top 20 features, ML/DL architectures, SCM DAG |
| **8. Design of the solution** | Chapter 6: Architectural & Solution Design | ~12–15 pages | Ready to draft | Multi-tier architecture, DAG structural equations, DB schema, UI/UX workflow |
| **9. Implementation details (resources, facilities, architecture, components)** | Chapter 7: Implementation Details | ~15–18 pages | Ready to draft | DuckDB ETL, Scikit-learn/PyTorch pipeline, DoWhy backdoor engine, Streamlit app |
| **10. Critical comparison between original objectives and achieved** | Chapter 8: Results & Empirical Testing (Sec 8.5) | ~4–5 pages | Ready to draft | Quantitative matrix comparing RQ1–RQ3 targets vs final empirical results |
| **11. Critical evaluation (efficacy, performance, efficiency, cost-benefit, comparisons)** | Chapter 8: Results & Critical Evaluation | ~15–18 pages | Ready to draft | RQ1 (XGBoost 77.42% vs 62% baseline), McNemar tests, RQ2 (3 causes vs 17 symptoms), RQ3 comparison |
| **12. Review of professional, social, legal and ethical issues** | Chapter 9: Professional, Legal, Social & Ethical Issues | ~8–10 pages | Ready to draft | SPER ethical form, CITI Cert #76046135, PhysioNet DUA, HIPAA, FDA SaMD guidelines, algorithmic fairness |
| **13. Conclusions and further developments** | Chapter 10: Conclusions & Future Work | ~6–8 pages | Ready to draft | Summary of contributions, ICU selection bias, unobserved confounding, prospective trials |
| **14. List of references** | References Bibliography | ~8–10 pages | Ready | Harvard referencing style (Sloman 1992, RGU standard) |
| **15. Mandatory Appendices** | Appendices A, B, C, D, E, F | ~25–35 pages | Ready | Single-spaced, 10pt font: |
| • *Project plan (~1000 words, ~3 pages, diagrammatic Gantt)* | Appendix A: Project Plan & Schedule | ~3–4 pages | Ready | Rationale, timeline, resources, Gantt diagram |
| • *Detailed design documentation* | Appendix B: Detailed Design Documentation | ~6–8 pages | Ready | UML component/sequence diagrams, SQLite DDL, DoWhy SCM equations |
| • *Detailed test data and results* | Appendix C: Comprehensive Test Data & Results | ~8–10 pages | Ready | Confusion matrices, classification reports, McNemar contingency tables, ATE data |
| • *User guide* | Appendix D: Clinical Portal User Guide | ~5–6 pages | Ready | Clinician manual: intake, damage grade interpretation, counterfactual intervention |
| • *Installation guide* | Appendix E: Software Installation & Deployment Guide | ~3–4 pages | Ready | Environment setup, Python virtualenv, DuckDB data ingestion, launching Streamlit |
| • *Ethical review documentation* | Appendix F: Student Project Ethical Review (SPER) Form | ~2–3 pages | Ready | Completed SPER documentation |

**Total Estimated Main Body Pages:** ~100–110 pages (excluding appendices), strictly satisfying RGU Section 14.4.

---

## 2. Deliverables Checklist (RGU Section 12)

1. [x] **Project Proposal** (Submitted & Approved)
2. [ ] **MSc Project Report** (100–110 pages, 1.5 line spacing, 11/12pt font, Harvard citations)
3. [x] **Working Prototype** (Streamlit Virtual Biopsy Portal with DuckDB & SQLite database)
4. [ ] **Student Project Ethical Review (SPER) Form** (CITI ID: 76046135, PhysioNet DUA)
5. [ ] **Presentation Slides** (10 to 12 slides prepared for demo & oral defense panel)
6. [ ] **Source Code / Demo Video** (Well-structured repository + demonstration video for marker inspection)
7. [ ] **Submission Package** (Turnitin upload + 2 printed copies + CD containing code, docs, slides, data)

---

## 3. Immediate Writing Schedule

* **Step 1:** Complete **Front Matter** (`00_Front_Matter.md`) with official RGU Title Page & signed Declaration verbatim.
* **Step 2:** Draft **Chapter 2: Literature Review** and **Chapter 3: Software Tools & Technology Evaluation**.
* **Step 3:** Draft **Chapter 4: Requirements Specification** (Functional & Non-Functional Requirements).
* **Step 4:** Integrate **Chapter 5: Methodology** (already verified against codebase) and **Chapter 6: Design of the Solution**.
* **Step 5:** Draft **Chapter 7: Implementation Details** and **Chapter 8: Results & Critical Evaluation**.
* **Step 6:** Draft **Chapter 9: Professional, Legal, Social & Ethical Issues** and **Chapter 10: Conclusions**.
* **Step 7:** Compile the **5 Mandatory Appendices** (Project Plan, Design Docs, Empirical Data, User Guide, Installation Guide).
