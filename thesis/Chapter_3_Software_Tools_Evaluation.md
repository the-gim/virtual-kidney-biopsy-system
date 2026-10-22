# Chapter 3: Software Tools and Technology Evaluation

---

## 3.1 Overview of Computational Requirements and Architectural Needs

Developing a clinical decision support system capable of non-invasively predicting renal parenchymal pathology while uncovering underlying causal mechanisms demands a modern, robust, and reproducible computational ecosystem. Unlike conventional laboratory machine learning experiments conducted on small, homogeneous datasets, the **Virtual Kidney Biopsy** framework operates at the intersection of four technically demanding computational domains:

1. **Big Data Ingestion and Out-of-Core Processing:** The primary data source, the Medical Information Mart for Intensive Care (MIMIC-IV v3.1), comprises tens of millions of timestamped laboratory events, vital signs, demographic records, and billing codes totaling over 100 gigabytes of uncompressed tabular data. The ingestion engine must execute complex relational joins, aggregations, and cohort filtering efficiently on personal workstation hardware without causing memory exhaustion.
2. **Machine Learning and Deep Learning Modeling:** The analytical pipeline benchmarks six distinct predictive architectures—ranging from gradient-boosted decision trees to recurrent neural networks—requiring frameworks that support vectorized computation, flexible neural layer definitions, automatic differentiation, and GPU acceleration.
3. **Formal Structural Causal Inference:** Implementing Judea Pearl’s Structural Causal Models (SCM) requires specialized libraries capable of translating Directed Acyclic Graphs (DAGs) into non-parametric identification criteria (e.g., Backdoor and Frontdoor adjustments), estimating Average Treatment Effects (ATE), and executing statistical refutation tests.
4. **Clinical Web Prototype and Secure Data Persistence:** In compliance with Robert Gordon University (RGU) MSc guidelines mandating a fully realized software prototype, the system requires an intuitive, doctor-centric graphical interface coupled with an embedded relational database to support patient intake, risk gauge visualization, and interactive counterfactual simulation under strict data privacy constraints.

To satisfy these multifaceted requirements, candidate software tools and technologies were systematically evaluated across architectural suitability, computational performance, community maturity, and clinical deployment safety.

---

## 3.2 Evaluation of Core Programming Environments: Python vs. R

The choice of primary programming language represents a foundational architectural decision that impacts package availability, computational efficiency, and downstream deployment feasibility. In modern biomedical data science and clinical informatics, the two dominant programming languages are **Python** and **R** (Ihaka & Gentleman, 1996; Van Rossum & Drake, 2009).

*Table 3.1: Systematic Comparative Evaluation Between Python and R for the Virtual Biopsy Framework.*
| Evaluation Dimension | R (v4.3+) | Python (v3.11+) | Selection Rationale for Virtual Biopsy |
| :--- | :--- | :--- | :--- |
| **Statistical Foundations** | Exceptional native statistical modeling (`stats`, `lme4`, `survival`). | Strong via `scipy.stats` and `statsmodels`. | R has historical strength in classical biostatistics, but Python fully satisfies all required non-parametric tests (e.g., McNemar's test). |
| **Machine Learning Ecosystem** | Fragmented across `caret`, `tidymodels`, `randomForest`. | Highly unified under `scikit-learn` and `xgboost`. | Python’s `scikit-learn` provides standardized API pipelines, hyperparameter tuning grids, and transformer chains. |
| **Deep Learning Frameworks** | R wrappers around Python backends (`keras`, `torch`). | Native primary APIs for `PyTorch` and `TensorFlow`. | Native Python execution avoids abstraction overhead, debugging friction, and virtual environment conflicts. |
| **Causal AI Ecosystem** | Strong classical econometrics (`MatchIt`, `dagitty`). | Cutting-edge SCM engines (`DoWhy`, `EconML`, `CausalNex`). | Microsoft’s `DoWhy` provides the only unified framework supporting Pearl's 4-stage causal inference lifecycle. |
| **Web Prototyping** | `R Shiny` (mature, reactive, but resource-heavy). | `Streamlit`, `Dash`, `FastAPI`. | `Streamlit` allows pure Python scripting with instant reactive UI components and effortless multi-page clinical navigation. |
| **Data Ingestion Performance** | `data.table` (fast, memory-efficient). | `DuckDB`, `Polars`, `pandas`. | Direct native bindings to `DuckDB` enable seamless out-of-core SQL execution over Parquet files. |

### Justification for Selecting Python 3.11
While R offers unmatched historical depth in classical epidemiologic biostatistics, **Python 3.11** was selected as the sole primary development language for this project for three decisive reasons:
1. **Unification of ML and Deep Learning:** Python provides first-class, native support for both Scikit-learn (for tabular tree ensembles) and PyTorch (for sequential 1D-CNN and LSTM architectures) within a single execution runtime, eliminating cross-language bridging.
2. **Availability of Microsoft DoWhy:** The core causal inference objective of this research (RQ2 and RQ3) depends on the DoWhy causal engine, which is written natively in Python and has no complete, actively maintained counterpart in R.
3. **End-to-End Prototyping:** Python enables a single, cohesive codebase spanning ETL, causal modeling, database operations, and the user-facing Streamlit dashboard, drastically reducing architectural complexity and deployment overhead.

---

## 3.3 Evaluation of Big Data Storage and Analytical Query Engines

Extracting a clean, leakage-free Chronic Kidney Disease cohort from MIMIC-IV involves filtering millions of admission records, matching ICD-10 diagnostic codes, and computing historical summary statistics (min, max, mean, standard deviation) across 20+ laboratory analytes. Four candidate data storage and processing technologies were benchmarked:

```
+---------------------------------------------------------------------------------------------------+
|                        BIG DATA QUERY ENGINE BENCHMARKING FOR MIMIC-IV                            |
+--------------------------+---------------------+-------------------+------------------------------+
| TECHNOLOGY CANDIDATE     | ARCHITECTURE TYPE   | MEMORY MODEL      | CLINICAL SUITABILITY SCORE   |
+--------------------------+---------------------+-------------------+------------------------------+
| **Pandas (v2.1+)**       | Row-oriented (RAM)  | In-memory only    | Moderate (Fails on >8GB RAM) |
| **Apache Spark (PySpark)**| Distributed Cluster | Distributed RAM   | Poor (Severe cluster overhead)|
| **PostgreSQL (v15+)**    | Client-Server RDBMS | Disk with Buffer  | Moderate (Requires DBA setup)|
| **DuckDB (v0.9+)**       | Columnar Embedded   | Vectorized Out-of-| **Superior (Selected)**      |
|                          | OLAP Engine         | Core Streaming    | (Zero config, 10x faster)    |
+--------------------------+---------------------+-------------------+------------------------------+
```

1. **In-Memory Pandas:** While widely used for exploratory tabular analysis, Pandas operates exclusively in-memory, requiring 3 to 5 times the raw file size in available RAM. Loading the 30-gigabyte uncompressed `labevents.csv` table from MIMIC-IV triggers fatal `OutOfMemory` exceptions on standard workstation hardware.
2. **Apache Spark (PySpark):** Spark provides massive horizontal scalability across distributed server clusters. However, for a single workstation or local hospital server, Spark introduces excessive configuration overhead, JVM runtime dependencies, and substantial network serialization latency for modest datasets (<100GB).
3. **PostgreSQL:** As a traditional client-server relational database, PostgreSQL easily handles multi-gigabyte datasets via disk-based B-tree indexes. However, it requires a dedicated daemon process, administrative credentials, network socket configuration, and substantial storage pre-allocation, complicating portable clinical deployment.
4. **DuckDB (Selected):** DuckDB is a purpose-built, high-performance columnar analytical database designed to execute embedded within the host Python process (Raasveldt & Mühleisen, 2019). DuckDB combines zero-dependency embedded deployment (like SQLite) with a state-of-the-art vectorized execution engine capable of executing complex SQL queries directly over compressed Apache Parquet and CSV files out-of-core.

### Justification for Selecting DuckDB
DuckDB was selected as the core ETL engine (`01_load_data_duckdb.py`) because it can query multi-gigabyte MIMIC-IV tables while streaming data from disk using less than 2 gigabytes of active RAM. Benchmarking demonstrated that DuckDB aggregated 12 million laboratory event rows in under 18 seconds on an Apple Silicon M-series processor, providing an order-of-magnitude speedup over traditional Pandas pipelines without requiring an external database server.

---

## 3.4 Evaluation of Machine Learning and Deep Learning Frameworks

Addressing Research Question 1 requires implementing and comparing six distinct machine learning architectures. Framework selection was guided by modularity, training efficiency, and support for clinical tabular data:

*Table 3.2: Evaluation of Machine Learning and Deep Learning Frameworks.*
| Tool / Framework | Target Algorithms | Architectural Role | Strengths and Selection Justification |
| :--- | :--- | :--- | :--- |
| **Scikit-Learn (v1.3+)** | Decision Trees, Support Vector Machines (SVM), Multi-Layer Perceptron (ANN) | Baseline Predictive Modeling & Evaluation | Provides battle-tested, standardized implementations of `SVC`, `DecisionTreeClassifier`, and `MLPClassifier`. Features comprehensive preprocessing transformers (`StandardScaler`), metric suites (`f1_score`, `roc_auc_score`), and calibration tools. |
| **Imbalanced-Learn (v0.11+)** | ADASYN (Adaptive Synthetic Oversampling) | Class Imbalance Mitigation | High-quality implementation of the ADASYN algorithm (He et al., 2008), enabling adaptive synthetic generation of minority severe CKD cases based on local boundary difficulty. |
| **XGBoost (v2.0+)** | Extreme Gradient Boosted Trees | High-Performance Non-Linear Modeling | The gold standard for tabular clinical data. Employs second-order Taylor expansion loss approximation, built-in sparsity awareness, and regularized objective functions (`reg_lambda`, `reg_alpha`) to prevent overfitting on clinical indicators. |
| **PyTorch (v2.1+)** | 1D Convolutional Neural Network (CNN1D), Long Short-Term Memory Network (LSTM) | Deep Sequential & Temporal Representation | Dynamic computation graphs, native GPU/MPS hardware acceleration, and granular control over tensor operations. Enables clean implementation of the 1D convolution feature extractor and recurrent LSTM memory cells. |

### Framework Harmonization
Rather than relying on a single monolithic library, the pipeline harmonizes **Scikit-learn** for preprocessing, data splitting, and classical baselines; **XGBoost** for state-of-the-art tabular ensemble learning; and **PyTorch** for deep temporal architectures. All models are unified under a common inference interface, allowing the clinical dashboard to toggle seamlessly between algorithms during patient evaluation.

---

## 3.5 Evaluation of Causal Inference Frameworks

Executing causal discovery and average treatment effect estimation (Research Question 2 and 3) requires specialized algorithmic tooling capable of formalizing causal assumptions beyond observational regression. Three leading open-source Python causal libraries were evaluated:

```
                       EVALUATION OF CAUSAL INFERENCE LIBRARIES
  
      DoWhy (Microsoft Research)                CausalNex (McKinsey QuantumBlack)
  ┌─────────────────────────────────┐        ┌───────────────────────────────────┐
  │ • Formal 4-step Pearl workflow  │        │ • Focuses strictly on structure   │
  │ • Explicit DAG specification    │   vs   │   learning (NOTEARS algorithm)    │
  │ • Backdoor/Frontdoor solvers    │        │ • Discretization of continuous    │
  │ • Automated refutation suite    │        │   biomarkers required (info loss) │
  └─────────────────────────────────┘        └───────────────────────────────────┘
                  │                                           │
                  ▼                                           ▼
         [RECOMMENDED: DoWhy]                       [REJECTED: CausalNex]
         Direct support for continuous              Information loss due to
         biomarkers & ATE calculation               binning clinical blood tests
```

1. **CausalNex:** A library dedicated to structure learning using the NOTEARS continuous optimization algorithm to infer DAG structures from observational data. However, CausalNex relies heavily on Bayesian Networks that require continuous laboratory biomarkers (e.g., serum glucose, potassium) to be discretized into categorical bins, causing severe information loss and boundary artifacts.
2. **EconML:** Developed by Microsoft Research for estimating heterogeneous treatment effects using machine learning (e.g., Double Machine Learning, Causal Forests). While powerful for econometrics, EconML does not provide native graphical DAG visualization or built-in causal identification logic, assuming the user has already resolved confounding outside the library.
3. **DoWhy (Selected):** Developed by Microsoft Research, DoWhy is the premier causal AI framework grounded explicitly in Judea Pearl’s Graphical Causal Models and Donald Rubin’s Potential Outcomes framework (Sharma & Kiciman, 2020). DoWhy enforces an end-to-end, four-stage causal inference lifecycle:
   * **Stage 1 (Model):** Encode domain nephrological knowledge as a formal causal Directed Acyclic Graph (DAG).
   * **Stage 2 (Identify):** Use graph theory and $d$-separation to prove whether the target Average Treatment Effect is identifiable under the Backdoor or Frontdoor criteria.
   * **Stage 3 (Estimate):** Apply statistical estimators (e.g., Backdoor linear regression, propensity score matching) to compute the numerical Average Treatment Effect (ATE).
   * **Stage 4 (Refute):** Stress-test the validity of the estimated effect using placebo treatments, random common confounder additions, and subset data refuters.

### Justification for Selecting DoWhy
DoWhy was chosen because it allows continuous clinical variables (e.g., electrolyte concentrations, glucose levels) to be modeled without artificial discretization, integrates domain nephrology rules into the DAG, and outputs statistically defensible ATE estimates with corresponding confidence intervals and $p$-values.

---

## 3.6 Evaluation of Web Interface and Persistence Technologies

To fulfill RGU's core requirement that the MSc project culminate in a functional software prototype, the analytical engine must be accessible to clinical practitioners through an intuitive interface with underlying patient record persistence:

### 3.6.1 Web Presentation Tier: Streamlit vs. Dash vs. Flask/React
* **Dash (Plotly):** Highly customizable and capable of rendering complex analytical charts. However, Dash requires verbose boilerplate code for simple state management and callback chains, extending development cycles unnecessarily.
* **Flask / FastAPI with React:** The enterprise standard for production web applications, offering complete separation of concerns. However, developing a decoupled REST API and custom React frontend introduces significant JavaScript build tooling, state synchronization complexity, and authentication overhead that distracts from the core data science objectives.
* **Streamlit (Selected):** A modern, reactive Python web framework designed specifically for data science and clinical tools. Streamlit renders reactive, interactive widgets (sliders, number inputs, dynamic gauges) with zero HTML/CSS/JavaScript boilerplate. It natively handles session state, integrates smoothly with Matplotlib/Seaborn visualizations, and re-executes reactively upon user input, making it ideal for the real-time **Counterfactual Treatment Simulator**.

### 3.6.2 Persistence Tier: SQLite vs. PostgreSQL
* **PostgreSQL:** Excellent for high-concurrency multi-hospital deployments, but requires hosting an external database server, configuring connection strings, and managing credentials, which complicates local grading and clinical portability.
* **SQLite (Selected):** A lightweight, self-contained, serverless relational database engine stored as a single disk file (`virtual_biopsy.db`). SQLite provides full ACID compliance, zero setup, and lightning-fast read/write operations for patient intake records, damage grade predictions, and simulated counterfactual adjustments. It satisfies clinical data privacy guidelines by keeping all simulated patient records strictly local to the clinician's machine.

---

## 3.7 Technology Stack Summary and Architecture Matrix

The final computational ecosystem harmonizes specialized, high-performance tools across every layer of the Virtual Kidney Biopsy pipeline:

*Table 3.3: Complete Technology Stack and Component Allocation.*
| Layer / Subsystem | Technology Selected | Version | Specific Project Implementation Role |
| :--- | :--- | :--- | :--- |
| **Programming Runtime** | Python | 3.11.x | Core execution environment across all pipeline modules. |
| **Big Data Query Engine** | DuckDB | 0.9.x | Out-of-core SQL ingestion of MIMIC-IV Parquet tables (`01_load_data_duckdb.py`). |
| **Tabular Data Processing** | Pandas & NumPy | 2.1.x / 1.26.x | In-memory feature manipulation, scaling, and array operations. |
| **Class Imbalance** | Imbalanced-Learn | 0.11.x | Adaptive Synthetic (ADASYN) minority oversampling (`02_feature_engineering_clean.py`). |
| **Machine Learning** | Scikit-Learn | 1.3.x | Decision Trees, SVM, MLP ANN, preprocessing pipelines, and evaluation metrics. |
| **Tree Ensembles** | XGBoost | 2.0.x | Extreme Gradient Boosting classifier with regularized objective loss. |
| **Deep Learning** | PyTorch | 2.1.x | 1D Convolutional Neural Network (CNN) and LSTM recurrent networks. |
| **Causal AI & SCM** | DoWhy | 0.11.x | Pearl DAG encoding, Backdoor identification, and ATE calculation (`07_causal_analysis_complete.py`). |
| **Statistical Testing** | Statsmodels & SciPy | 0.14.x / 1.11.x | Non-parametric McNemar hypothesis testing and $p$-value computation (`06_statistical_testing.py`). |
| **Clinical Presentation** | Streamlit | 1.28.x | Interactive web prototype, risk gauge rendering, and counterfactual UI (`10_clinical_dashboard.py`). |
| **Local Persistence** | SQLite | 3.x | Serverless local relational storage for patient profiles and simulated treatments. |
| **Visualization** | Matplotlib & Seaborn | 3.8.x / 0.13.x | Publication-quality confusion matrices, ROC curves, and causal graph plots. |

This cohesive technological architecture ensures end-to-end reproducibility, rigorous computational performance, and immediate clinical accessibility, providing a defensible foundation for the empirical investigations that follow.
