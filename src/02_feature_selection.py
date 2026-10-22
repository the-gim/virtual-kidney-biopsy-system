import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 2: FEATURE SELECTION (Removing Data Leakage)")
print("=" * 80)

BASE_WORKSPACE = '/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research'
PROCESSED_DATA_PATH = os.path.join(BASE_WORKSPACE, 'data/processed/')

# Load cleaned data
print("📂 Loading cleaned data...")
input_path = os.path.join(PROCESSED_DATA_PATH, '01_cleaned_data.csv')
df = pd.read_csv(input_path)

# Drop leaky features
print("🧹 Removing leaky features (creatinine, bun, hemoglobin)...")
leaky_features = [col for col in df.columns if any(k in col for k in ['creatinine', 'bun', 'hemoglobin', 'anemia', 'egfr'])]

all_features = [col for col in df.columns
                if col not in ['subject_id', 'target_binary'] + leaky_features]

print(f"   Removed {len(leaky_features)} leaky features. Remaining candidate features: {len(all_features)}")

X_all = df[all_features].fillna(0)
y_all = df['target_binary']

# Split data
print("🔀 Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# Scale data
print("⚖️ Scaling features...")
scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Select K Best Features
print("🎯 Running SelectKBest to find top 20 features...")
selector = SelectKBest(f_classif, k=20)
selector.fit(X_train_sc, y_train)

top_feature_mask = selector.get_support()
top_20_features  = [f for f, s in zip(all_features, top_feature_mask) if s]

print(f"\n   Top 20 Features Selected:")
scores = selector.scores_
feature_scores = sorted(
    zip(all_features, scores),
    key=lambda x: x[1], reverse=True
)[:20]

for i, (feat, score) in enumerate(feature_scores, 1):
    print(f"   {i:2}. {feat:<35} (F-score: {score:.1f})")

# Filter datasets to top 20
X_train_final = X_train_sc[top_20_features]
X_test_final  = X_test_sc[top_20_features]

# Save datasets for modeling steps
print("\n💾 Saving processed train and test datasets...")
X_train_final.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_train_selected.csv'), index=False)
X_test_final.to_csv(os.path.join(PROCESSED_DATA_PATH, 'X_test_selected.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, 'y_test.csv'), index=False)

print("✅ Feature selection complete.")
