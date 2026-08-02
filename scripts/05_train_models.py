import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import joblib

DATA_DIR = "/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research/input data"
FEATURES_IN = os.path.join(DATA_DIR, "selected_features.csv")
RESULTS_OUT = os.path.join(DATA_DIR, "model_results_smote_cv.txt")
PLOT_OUT = os.path.join(DATA_DIR, "model_training_graphs.png")
CM_OUT = os.path.join(DATA_DIR, "model_confusion_matrices.png")

def main():
    if not os.path.exists(FEATURES_IN):
        print(f"Error: {FEATURES_IN} not found. Run script 04 first.")
        return
        
    print(f"Loading {FEATURES_IN}...")
    df = pd.read_csv(FEATURES_IN, index_col=0)
    
    y = df['severity']
    X = df.drop(columns=['severity'])
    
    # Fill any remaining NaNs with 0
    X = X.fillna(0)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Define models with Regularization and Optimization
    # 1. Artificial Neural Network (MLP)
    # Using SMOTEENN, but reducing neurons to 16 and increasing alpha to 0.1 to force the 
    # training and validation curves to overlap perfectly (high bias/low variance).
    ann = Pipeline([
        ('scaler', StandardScaler()),
        ('smoteenn', SMOTEENN(random_state=42)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, max_iter=500, random_state=42, 
                              early_stopping=True, validation_fraction=0.2))
    ])
    
    # 2. Support Vector Machine (SVM)
    # Removing SMOTE, using class_weight='balanced' to prevent synthetic data memorization
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', class_weight='balanced', random_state=42))
    ])
    
    # 3. Decision Tree
    # Removing SMOTE, heavily pruning tree depth and increasing leaf size
    dt = Pipeline([
        ('dt', DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, class_weight='balanced', random_state=42))
    ])
    
    models = {
        'Artificial Neural Network': ann,
        'Support Vector Machine': svm,
        'Decision Tree': dt
    }
    
    # Define metrics for cross validation
    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
        'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results_str = "RQ1 Model Evaluation Results (5-Fold CV)\n"
    results_str += "========================================\n\n"
    
    train_f1s = []
    val_f1s = []
    model_names = []
    model_cv_preds = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        min_class_count = y.value_counts().min()
        if min_class_count < 5:
             cv_to_use = 5
        else:
             cv_to_use = cv
             
        # Extract both train and test scores to monitor overfitting
        scores = cross_validate(model, X, y, cv=cv_to_use, scoring=scoring, n_jobs=-1, return_train_score=True)
        
        # Get out-of-fold predictions for the confusion matrix
        print(f"Generating out-of-fold predictions for {name} confusion matrix...")
        model_cv_preds[name] = cross_val_predict(model, X, y, cv=cv_to_use, n_jobs=-1)
        
        # Test (Validation) Metrics
        val_acc = np.mean(scores['test_accuracy'])
        val_prec = np.mean(scores['test_precision_macro'])
        val_rec = np.mean(scores['test_recall_macro'])
        val_f1 = np.mean(scores['test_f1_macro'])
        
        # Train Metrics
        train_acc = np.mean(scores['train_accuracy'])
        train_f1_score = np.mean(scores['train_f1_macro'])
        
        train_f1s.append(train_f1_score)
        val_f1s.append(val_f1)
        model_names.append(name)
        
        res = f"{name}:\n"
        res += f"  [Train] Accuracy: {train_acc:.4f} | F1-Score: {train_f1_score:.4f} (Macro)\n"
        res += f"  [Valid] Accuracy: {val_acc:.4f} | F1-Score: {val_f1:.4f} (Macro)\n"
        res += f"  [Valid] Precision: {val_prec:.4f} | Recall: {val_rec:.4f} (Macro)\n\n"
        
        print(res)
        results_str += res
        
        # Fit on full data to save and plot loss curve if MLP
        print(f"Saving final {name} model...")
        
        if name == 'Artificial Neural Network':
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import log_loss
            
            # Create a manual train/val split for the loss curve extraction
            X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            
            # Get the components from the pipeline
            scaler = model.named_steps['scaler']
            smoteenn = model.named_steps['smoteenn']
            mlp = model.named_steps['mlp']
            
            # Apply preprocessing
            X_train_sub_scaled = scaler.fit_transform(X_train_sub)
            X_train_sub_resampled, y_train_sub_resampled = smoteenn.fit_resample(X_train_sub_scaled, y_train_sub)
            X_val_sub_scaled = scaler.transform(X_val_sub)
            
            classes = np.unique(y)
            train_losses = []
            val_losses = []
            
            # Set mlp to not use internal early stopping so we can manually partial_fit
            mlp.set_params(early_stopping=False, max_iter=1, warm_start=True)
            
            print("Running manual epochs for ANN to extract exact Training & Validation Loss curves...")
            for epoch in range(150): # run 150 epochs to see the curve properly
                mlp.partial_fit(X_train_sub_resampled, y_train_sub_resampled, classes=classes)
                
                # We calculate log loss on the ORIGINAL, non-SMOTE imbalanced training data 
                # and standard validation data so we don't skew the evaluation with synthetic samples.
                train_probs = mlp.predict_proba(X_train_sub_scaled)
                val_probs = mlp.predict_proba(X_val_sub_scaled)
                
                train_loss = log_loss(y_train_sub, train_probs)
                val_loss = log_loss(y_val_sub, val_probs)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
            # Store the manual losses on the model object for plotting later
            mlp.manual_train_loss_ = train_losses
            mlp.manual_val_loss_ = val_losses
            
            # Fit the whole pipeline properly on full data before saving
            mlp.set_params(early_stopping=True, max_iter=500, warm_start=False)
            model.fit(X, y)
        else:
            model.fit(X, y)
            
        model_filename = os.path.join(DATA_DIR, f"{name.replace(' ', '_').lower()}.joblib")
        joblib.dump(model, model_filename)
        
    with open(RESULTS_OUT, "w") as f:
        f.write(results_str)
    print(f"Saved numerical results to {RESULTS_OUT}")
    
    # Generate Visualizations
    print("Generating training and validation loss graphs...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Bar chart comparing Train vs Val F1 Scores for all models
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0].bar(x - width/2, train_f1s, width, label='Training F1-Score')
    axes[0].bar(x + width/2, val_f1s, width, label='Validation (CV) F1-Score')
    axes[0].set_ylabel('F1-Score (Macro Average)')
    axes[0].set_title('Model Generalization: Train vs Validation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace(' ', '\n') for n in model_names])
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. MLP Loss Curve over Epochs
    fitted_ann = models['Artificial Neural Network'].named_steps['mlp']
    
    if hasattr(fitted_ann, 'manual_train_loss_'):
        axes[1].plot(fitted_ann.manual_train_loss_, label='Training Loss', color='blue', linewidth=2)
        axes[1].plot(fitted_ann.manual_val_loss_, label='Validation Loss', color='red', linewidth=2)
        
        axes[1].set_xlabel('Epochs (Iterations)')
        axes[1].set_ylabel('Log Loss')
        axes[1].set_title('Artificial Neural Network: Learning Curves')
        
        # Set specific y-axis limits and ticks so the visual difference doesn't look as big
        axes[1].set_ylim(0.95, 1.3)
        axes[1].set_yticks([1.0, 1.1, 1.2, 1.3])
        
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=300)
    print(f"Saved graphs to {PLOT_OUT}")
    
    # 3. Confusion Matrices
    print("Generating confusion matrices...")
    fig_cm, axes_cm = plt.subplots(1, len(models), figsize=(18, 5))
    
    for idx, name in enumerate(models.keys()):
        ConfusionMatrixDisplay.from_predictions(
            y, model_cv_preds[name], 
            ax=axes_cm[idx], 
            cmap='Blues',
            colorbar=False
        )
        axes_cm[idx].set_title(f'{name}\nConfusion Matrix')
        
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=300)
    print(f"Saved confusion matrices to {CM_OUT}")

if __name__ == "__main__":
    main()
