================================================================================
VIRTUAL KIDNEY BIOPSY: RESEARCH TEST RESULTS & EMPIRICAL BENCHMARKS BUNDLE
================================================================================
Student: Gimhani Wijesuriya (MSc Big Data Analytics)
Institution: Robert Gordon University (Aberdeen, UK) / Informatics Institute of Technology

CONTENTS OF THIS RESEARCH BUNDLE:

1. RQ1: PREDICTIVE MACHINE LEARNING & DEEP LEARNING BENCHMARKS
   - 01_RQ1_all_models_benchmark.csv: Master comparison of all 6 architectures
     (XGBoost, SVM, ANN, 1D-CNN, Bi-LSTM, Decision Tree) + Li et al. (Baseline).
   - 01b_RQ1_advanced_deep_learning_metrics.csv: 10-Fold CV metrics for XGBoost, CNN, LSTM.
   - 01c_RQ1_basic_classifiers_metrics.csv: 10-Fold CV metrics for Decision Tree, SVM, ANN.

2. STATISTICAL HYPOTHESIS TESTING (MCNEMAR TESTS)
   - 02_McNemar_hypothesis_tests_summary.csv: Pairwise discordance matrices, chi-square,
     and non-parametric p-values proving XGBoost's statistical superiority.
   - 02b_McNemar_pairwise_p_values.csv: Full pairwise p-value matrix across top models.
   - 02c_McNemar_contingency_counts.csv: Raw 2x2 contingency counts for discordant pairs.

3. RQ2: CAUSAL AI DISCOVERY & STATISTICAL REFUTATION TESTS (MICROSOFT DOWHY)
   - Causal Model Framework: Judea Pearl's Structural Causal Models (SCM) via Microsoft DoWhy.
   - Estimation Algorithm: Backdoor Criterion Adjustment via Linear Regression.
   - 03_RQ2_causal_effects_all_20_features.csv: Exact Average Treatment Effect (ATE)
     for all 20 biomarkers vs. Pearson correlation coefficients.
   - 03b_RQ2_causal_refutation_tests.csv: Statistical validation for Causal AI:
       * Placebo Treatment Refutation Test: Drops effect to 0.0000 (p < 0.001, PASS).
       * Random Common Cause Refutation Test: Effect remains invariant (PASS).
   - 03c_RQ2_treatable_actionable_causes.csv: Clinically actionable root causes
     (Hypertension: ATE = -0.0828; Fasting Glucose: ATE = -0.0483).

4. RQ3: ASSOCIATIONAL VS. CAUSAL AI EMPIRICAL BENCHMARK
   - Causal AI ML Model: Regularized XGBoost Classifier trained strictly on 3 root causes.
   - 04_RQ3_causal_vs_noncausal_benchmark.csv: Empirical comparison between Model A
     (20 features, diagnostic accuracy 77.42%) and Model B (3 root causes, 96.24% recall).
   - Paired McNemar Test: Chi-square = 339.43, p = 8.47e-76 (statistically significant).

5. PUBLICATION-QUALITY FIGURES
   - figures/causal_dag_20features.png: Complete Pearl DAG of all 20 features.
   - figures/causal_effects_forest_plot.png: Forest plot contrasting ATE vs. correlation.
   - figures/rq3_confusion_matrix_comparison.png: Side-by-side confusion matrix benchmark.
================================================================================
