"""
Script de comparaison finale et sauvegarde des meilleurs modeles
"""
import os
import pickle
import pandas as pd
from prepare_data import load_and_prepare_data
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ========================================
# 1. CHARGEMENT DES DONNEES
# ========================================
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "btc_hourly.csv")
X, y, scaler = load_and_prepare_data(data_path, scale=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ========================================
# 2. MODELES OPTIMISES
# ========================================

# Meilleurs params trouves par Bayesian Optimization
lgb_best = LGBMClassifier(
    n_estimators=129,
    max_depth=12,
    learning_rate=0.02071820001514905,
    num_leaves=42,
    random_state=42,
    verbose=-1
)

xgb_best = XGBClassifier(
    n_estimators=230,
    max_depth=6,
    learning_rate=0.14717976673069674,
    random_state=42,
    verbosity=0
)

cb_best = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0
)

models = {
    'LightGBM': lgb_best,
    'XGBoost': xgb_best,
    'CatBoost': cb_best,
}

# ========================================
# 3. EVALUATION DETAILLEE
# ========================================
print("\n" + "="*80)
print("EVALUATION DETAILLEE DES MODELES OPTIMISES")
print("="*80)

results_summary = []

for model_name, model in models.items():
    print(f"\n[{model_name.upper()}]")
    print("-" * 80)
    
    # Training
    model.fit(X_train, y_train)
    
    # Test predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
    }
    
    results_summary.append(metrics)
    
    # Print metrics
    for key, value in metrics.items():
        if key != 'Model':
            print(f"  {key:15} : {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:6d} | FP: {cm[0,1]:6d}")
    print(f"    FN: {cm[1,0]:6d} | TP: {cm[1,1]:6d}")
    
    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), f'{model_name.lower()}_best.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n  [SAVED] {model_path}")

# ========================================
# 4. RESUME COMPARATIF
# ========================================
print("\n" + "="*80)
print("RESUME COMPARATIF")
print("="*80)

df_results = pd.DataFrame(results_summary)
df_results = df_results.sort_values('F1-Score', ascending=False)

print("\n" + df_results.to_string(index=False))

print("\n" + "="*80)
print(f"MEILLEUR MODELE: {df_results.iloc[0]['Model']} (F1-Score: {df_results.iloc[0]['F1-Score']:.4f})")
print("="*80 + "\n")

# ========================================
# 5. FEATURE IMPORTANCE (si disponible)
# ========================================
if hasattr(lgb_best, 'feature_importances_'):
    print("\n[LIGHTGBM] Top 10 Features:")
    fi = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': lgb_best.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    for idx, row in fi.iterrows():
        print(f"  {row['Feature']:20} : {row['Importance']:.4f}")

print("\n[DONE] Evaluation complete!")
