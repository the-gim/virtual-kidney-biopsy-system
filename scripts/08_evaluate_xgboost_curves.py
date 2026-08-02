import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
ROC_PLOT_OUT = os.path.join(DATA_DIR, "tuned_xgboost_roc_curve.png")
PR_PLOT_OUT = os.path.join(DATA_DIR, "tuned_xgboost_pr_curve.png")

def main():
    if not os.path.exists(FEATURES_IN):
        print(f"Error: {FEATURES_IN} not found.")
        return
        
    print(f"Loading {FEATURES_IN}...")
    df = pd.read_csv(FEATURES_IN, index_col=0)
    
    y = df['severity']
    X = df.drop(columns=['severity']).fillna(0)
    
    # Binarize labels for multiclass ROC/PR curves
    classes = np.unique(y)
    y_bin = label_binarize(y, classes=classes)
    n_classes = y_bin.shape[1]
    
    print(f"Dataset shape: {X.shape}")
    
    # 2. Tuned XGBoost
    xgb_tuned = Pipeline([
        ('scaler', StandardScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('xgb', XGBClassifier(
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42, 
            n_jobs=1,
            max_depth=3,             
            learning_rate=0.05,      
            n_estimators=200,        
            subsample=0.8,           
            colsample_bytree=0.8,    
            gamma=1,                 
            reg_lambda=10,           
            reg_alpha=1              
        ))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Generating out-of-fold probabilities via cross-validation...", flush=True)
    # Get probability predictions instead of hard class predictions
    y_prob = cross_val_predict(xgb_tuned, X, y, cv=cv, method='predict_proba', n_jobs=1)
    
    # Calculate Macro AUROC
    macro_auroc = roc_auc_score(y, y_prob, multi_class='ovr', average='macro')
    # Calculate Macro AUPRC
    macro_auprc = average_precision_score(y_bin, y_prob, average='macro')
    
    print("\nEvaluation Metrics (Cross-Validated):")
    print("=========================================")
    print(f"Macro AUROC: {macro_auroc:.4f}")
    print(f"Macro AUPRC: {macro_auprc:.4f}")
    
    # ---------------------------------------------------------
    # Plot ROC Curves
    # ---------------------------------------------------------
    print("\nGenerating ROC Curves...")
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {classes[i]} (area = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (AUROC) - Tuned XGBoost')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_OUT, dpi=300)
    print(f"Saved ROC plot to {ROC_PLOT_OUT}")
    
    # ---------------------------------------------------------
    # Plot Precision-Recall Curves
    # ---------------------------------------------------------
    print("Generating Precision-Recall Curves...")
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        pr_auc = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {classes[i]} (area = {pr_auc:.2f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (AUPRC) - Tuned XGBoost')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PR_PLOT_OUT, dpi=300)
    print(f"Saved PR plot to {PR_PLOT_OUT}")

if __name__ == "__main__":
    main()
