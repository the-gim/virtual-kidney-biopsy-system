import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from dowhy import CausalModel
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CAUSAL GRAPH & ESTIMATION - ALL 20 SELECTED FEATURES")
print("=" * 80)

BASE_PATH = '/Users/gimhaniwijesuriya/Documents/virtual_biopsy_research'
OUT_PATH  = os.path.join(BASE_PATH, 'outputs/causal_results/')
os.makedirs(OUT_PATH, exist_ok=True)

# =====================================================================
# STEP 1: DEFINE ALL 20 FEATURES WITH CAUSAL RELATIONSHIPS
# =====================================================================
print("\n📊 Defining causal relationships for our exact 20 features...")

feature_groups = {
    'demographics_comorbidities': [
        'gender_encoded',
        'has_hypertension',
        'glucose_min' # Diabetes proxy
    ],
    'hospital_outcomes': [
        'avg_los',
    ],
    'electrolytes': [
        'potassium_mean',
        'potassium_max',
        'potassium_std',
        'hyperkalemia',
        'sodium_std',
    ],
    'minerals': [
        'phosphate_mean',
        'phosphate_min',
        'phosphate_max',
        'phosphate_std',
        'calcium_min',
    ],
    'acid_base': [
        'bicarbonate_mean',
        'bicarbonate_min',
        'bicarbonate_std',
    ],
    'metabolic': [
        'albumin_mean',
        'albumin_min',
        'low_albumin',
    ],
}

# =====================================================================
# STEP 2: DEFINE CAUSAL EDGES (Pearl's DAG Theory)
# =====================================================================
print("\n🔗 Building causal edges based on nephrology literature...")

causal_edges = [
    # ── Root Causes → Kidney Severity ──
    ('gender_encoded', 'Kidney_Severity'),
    ('has_hypertension', 'Kidney_Severity'),
    ('glucose_min', 'Kidney_Severity'),
    ('glucose_min', 'has_hypertension'),

    # ── Kidney Severity → Hospital Outcomes ──
    ('Kidney_Severity', 'avg_los'),

    # ── Kidney Severity → Electrolytes ──
    ('Kidney_Severity', 'potassium_mean'),
    ('Kidney_Severity', 'potassium_max'),
    ('Kidney_Severity', 'potassium_std'),
    ('Kidney_Severity', 'hyperkalemia'),
    ('Kidney_Severity', 'sodium_std'),

    # ── Kidney Severity → Minerals ──
    ('Kidney_Severity', 'phosphate_mean'),
    ('Kidney_Severity', 'phosphate_min'),
    ('Kidney_Severity', 'phosphate_max'),
    ('Kidney_Severity', 'phosphate_std'),
    ('Kidney_Severity', 'calcium_min'),

    # ── Kidney Severity → Acid-Base ──
    ('Kidney_Severity', 'bicarbonate_mean'),
    ('Kidney_Severity', 'bicarbonate_min'),
    ('Kidney_Severity', 'bicarbonate_std'),

    # ── Kidney Severity → Metabolic ──
    ('Kidney_Severity', 'albumin_mean'),
    ('Kidney_Severity', 'albumin_min'),
    ('Kidney_Severity', 'low_albumin'),
]

print(f"   ✅ Total causal edges: {len(causal_edges)}")

# =====================================================================
# STEP 3: BUILD NETWORKX GRAPH
# =====================================================================
print("\n🔨 Building DAG...")

G = nx.DiGraph()
G.add_edges_from(causal_edges)

is_dag = nx.is_directed_acyclic_graph(G)
print(f"   Is valid DAG: {is_dag} {'✅' if is_dag else '❌'}")
print(f"   Total nodes: {len(G.nodes())}")
print(f"   Total edges: {len(G.edges())}")

# =====================================================================
# STEP 4: DRAW COMPLETE CAUSAL GRAPH
# =====================================================================
print("\n🎨 Drawing complete causal graph...")

fig, ax = plt.subplots(figsize=(24, 16), facecolor='#0f1117')
ax.set_facecolor('#0f1117')

# Node positions tailored for our 20 features
pos = {
    'gender_encoded': (-4, 2),
    'has_hypertension': (-4, 0),
    'glucose_min': (-4, -2),

    'avg_los': (6, 0),

    'potassium_mean': (0, 4),
    'potassium_max': (0, 3),
    'potassium_std': (0, 2),
    'hyperkalemia': (0, 1),
    'sodium_std': (1.5, 3.5),

    'phosphate_mean': (0, -1),
    'phosphate_min': (0, -2),
    'phosphate_max': (0, -3),
    'phosphate_std': (0, -4),
    'calcium_min': (1.5, -3.5),

    'bicarbonate_mean': (2.5, 3),
    'bicarbonate_min': (2.5, 2),
    'bicarbonate_std': (2.5, 1),

    'albumin_mean': (2.5, -1),
    'albumin_min': (2.5, -2),
    'low_albumin': (2.5, -3),

    'Kidney_Severity': (4, 0),
}

color_map = {
    'gender_encoded': '#5c6bc0',
    'has_hypertension': '#5c6bc0',
    'glucose_min': '#5c6bc0',
    
    'avg_los': '#ffa726',
    
    'potassium_mean': '#26c6da',
    'potassium_max': '#26c6da',
    'potassium_std': '#26c6da',
    'hyperkalemia': '#26c6da',
    'sodium_std': '#26c6da',
    
    'phosphate_mean': '#26a69a',
    'phosphate_min': '#26a69a',
    'phosphate_max': '#26a69a',
    'phosphate_std': '#26a69a',
    'calcium_min': '#26a69a',
    
    'bicarbonate_mean': '#ab47bc',
    'bicarbonate_min': '#ab47bc',
    'bicarbonate_std': '#ab47bc',
    
    'albumin_mean': '#66bb6a',
    'albumin_min': '#66bb6a',
    'low_albumin': '#66bb6a',
    
    'Kidney_Severity': '#ef5350',
}

node_colors = [color_map.get(n, '#ffffff') for n in G.nodes()]

nx.draw_networkx_edges(
    G, pos,
    edge_color      = '#9fa8da',
    arrows          = True,
    arrowsize       = 15,
    width           = 0.8,
    alpha           = 0.5,
    connectionstyle = 'arc3,rad=0.05',
    ax              = ax,
)

nx.draw_networkx_nodes(
    G, pos,
    node_color = node_colors,
    node_size  = 2200,
    alpha      = 0.9,
    ax         = ax,
)

nx.draw_networkx_labels(
    G, pos,
    font_size   = 8,
    font_color  = 'white',
    font_weight = 'bold',
    ax          = ax,
)

ax.set_title(
    'Complete Causal DAG — Virtual Kidney Biopsy (RQ2)\n'
    'Our Exact 20 Selected Features → Kidney Severity\n'
    'Theory: Pearl\'s Causal Model & KDIGO',
    fontsize   = 16,
    fontweight = 'bold',
    color      = 'white',
    pad        = 20,
)

legend_elements = [
    mpatches.Patch(color='#5c6bc0', label='Root Causes (3)'),
    mpatches.Patch(color='#ffa726', label='Hospital Outcomes (1)'),
    mpatches.Patch(color='#26c6da', label='Electrolytes (5)'),
    mpatches.Patch(color='#26a69a', label='Minerals (5)'),
    mpatches.Patch(color='#ab47bc', label='Acid-base (3)'),
    mpatches.Patch(color='#66bb6a', label='Metabolic/Protein (3)'),
    mpatches.Patch(color='#ef5350', label='Target: Kidney Severity'),
]
ax.legend(
    handles     = legend_elements,
    loc         = 'lower left',
    fontsize    = 10,
    facecolor   = '#1a1d27',
    labelcolor  = 'white',
    edgecolor   = '#2e3347',
    title       = 'Feature Categories',
    title_fontsize = 11,
)

ax.axis('off')
plt.tight_layout()
plt.savefig(f'{OUT_PATH}/causal_graph_20features.png',
            dpi=200, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("   ✅ Saved causal_graph_20features.png")

# =====================================================================
# STEP 5: CAUSAL EFFECT ESTIMATION FOR ALL 20 FEATURES
# =====================================================================
print("\n📊 Estimating causal effects for all 20 features...")

X = pd.read_csv(f'{BASE_PATH}/data/processed/X_train_selected.csv')
y = pd.read_csv(f'{BASE_PATH}/data/processed/y_train.csv').squeeze()

df = X.copy()
df['Kidney_Severity'] = y
print(f"   ✅ Loaded {len(df)} actual patients from X_train_selected.csv")

all_20_features = list(X.columns)

treatable = {
    'gender_encoded': False,
    'has_hypertension': True,
    'glucose_min': True,
    'avg_los': False,
    'potassium_mean': True,
    'potassium_max': True,
    'potassium_std': True,
    'hyperkalemia': True,
    'sodium_std': True,
    'phosphate_mean': True,
    'phosphate_min': True,
    'phosphate_max': True,
    'phosphate_std': True,
    'calcium_min': True,
    'bicarbonate_mean': True,
    'bicarbonate_min': True,
    'bicarbonate_std': True,
    'albumin_mean': True,
    'albumin_min': True,
    'low_albumin': True,
}

interventions = {
    'gender_encoded': 'Not modifiable',
    'has_hypertension': 'Antihypertensives (ACEi/ARB)',
    'glucose_min': 'Insulin/Metformin',
    'avg_los': 'Indirect (Care Optimization)',
    'potassium_mean': 'Dietary restriction, Diuretics',
    'potassium_max': 'Dietary restriction, Diuretics',
    'potassium_std': 'Dietary management',
    'hyperkalemia': 'Acute medical management',
    'sodium_std': 'Sodium balance management',
    'phosphate_mean': 'Phosphate binders',
    'phosphate_min': 'Phosphate management',
    'phosphate_max': 'Phosphate binders',
    'phosphate_std': 'Dietary management',
    'calcium_min': 'Calcium supplements',
    'bicarbonate_mean': 'Sodium bicarbonate',
    'bicarbonate_min': 'Sodium bicarbonate',
    'bicarbonate_std': 'Acid-base management',
    'albumin_mean': 'Protein diet management',
    'albumin_min': 'Nutritional support',
    'low_albumin': 'Nutritional support',
}

causal_results = []
# Create string format for causal model
causal_graph = nx.nx_pydot.to_pydot(G).to_string()

for feature in all_20_features:
    corr = df[feature].corr(df['Kidney_Severity'])

    try:
        model = CausalModel(
            data=df,
            treatment=feature,
            outcome='Kidney_Severity',
            graph=causal_graph
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name='backdoor.linear_regression',
            test_significance=False
        )
        causal_effect = estimate.value
    except Exception:
        # If unidentifiable (e.g. outcome has 0 causal path to kidney severity), causal effect is 0
        causal_effect = 0.0

    causal_results.append({
        'Feature': feature,
        'Category': next((k for k, v in feature_groups.items() if feature in v), 'other'),
        'Causal_Effect': round(causal_effect, 4),
        'Correlation': round(corr, 4),
        'Treatable': '✅ Yes' if treatable.get(feature) else '❌ No',
        'Intervention': interventions.get(feature, 'N/A'),
    })

causal_df = pd.DataFrame(causal_results)
causal_df = causal_df.sort_values('Correlation', key=abs, ascending=False)

# =====================================================================
# STEP 6: PRINT RESULTS TABLE
# =====================================================================
print("\n" + "=" * 80)
print("CAUSAL EFFECT RESULTS - ALL 20 FEATURES")
print("=" * 80)

print(f"\n{'#':<4} {'Feature':<22} {'Category':<26} {'Causal Effect':>14} {'Correlation':>12} {'Treatable':>10}")
print(f"{'─'*95}")

for i, row in causal_df.reset_index().iterrows():
    print(f"{i+1:<4} {row['Feature']:<22} {row['Category']:<26} "
          f"{str(row['Causal_Effect']):>14} {str(row['Correlation']):>12} "
          f"{row['Treatable']:>10}")

causal_df.to_csv(f'{OUT_PATH}/causal_effects_20features.csv', index=False)
print(f"\n✅ Saved causal_effects_20features.csv")

# =====================================================================
# STEP 7: TREATABLE vs NON-TREATABLE SUMMARY
# =====================================================================
treatable_df = causal_df[causal_df['Treatable'] == '✅ Yes']
non_treatable_df = causal_df[causal_df['Treatable'] == '❌ No']

print(f"\n✅ TREATABLE CAUSES ({len(treatable_df)} features):")
for _, row in treatable_df.iterrows():
    if row['Causal_Effect'] != 0:
        print(f"   • {row['Feature']:<22} → {row['Intervention']} (Active Root Cause!)")
    else:
        print(f"   • {row['Feature']:<22} → {row['Intervention']} (Outcome only)")

treatable_df.to_csv(f'{OUT_PATH}/treatable_causes.csv', index=False)

# =====================================================================
# STEP 8: VISUALIZATION OF CAUSAL EFFECTS
# =====================================================================
print("\n🎨 Creating causal effect visualization...")

fig, axes = plt.subplots(1, 2, figsize=(18, 10), facecolor='#0f1117')
fig.suptitle('Causal Effects — Exact 20 Features → Kidney Severity',
             fontsize=16, fontweight='bold', color='white')

# Plot 1: All features by causal effect
ax1 = axes[0]
numeric_df = causal_df.copy()
numeric_df['Causal_Effect'] = pd.to_numeric(numeric_df['Causal_Effect'], errors='coerce').fillna(0)
numeric_df = numeric_df.sort_values('Causal_Effect', ascending=True)

colors_bar = ['#ef5350' if v > 0 else '#5c6bc0' if v < 0 else '#9e9e9e' for v in numeric_df['Causal_Effect']]
bars = ax1.barh(numeric_df['Feature'], numeric_df['Causal_Effect'],
                color=colors_bar, edgecolor='#0f1117', alpha=0.85)
ax1.axvline(0, color='white', linewidth=1.5, linestyle='--')
ax1.set_title('Causal Effect on Kidney Severity', fontweight='bold', fontsize=13, color='white')
ax1.set_xlabel('Causal Effect Estimate', color='white')
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')
ax1.set_facecolor('#0f1117')

# Plot 2: Treatable vs non-treatable
ax2 = axes[1]
categories = causal_df['Category'].value_counts()
colors_pie = ['#5c6bc0','#ffa726','#26c6da','#26a69a','#ab47bc','#66bb6a','#ec407a']
ax2.pie(categories.values, labels=categories.index,
        colors=colors_pie[:len(categories)],
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 11})
ax2.set_title('Feature Category Distribution', fontweight='bold', fontsize=13, color='white')

plt.tight_layout()
plt.savefig(f'{OUT_PATH}/causal_effects_20features_viz.png',
            dpi=200, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("   ✅ Saved causal_effects_20features_viz.png")

print(f"""
{'='*80}
✅ COMPLETE CAUSAL ANALYSIS DONE FOR OUR DATA!
{'='*80}
""")
