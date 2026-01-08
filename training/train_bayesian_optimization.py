"""
Training OPTIMISE avec Bayesian Optimization + modeles GPU-ready
XGBoost, LightGBM, CatBoost + Voting Ensemble
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from sklearn.ensemble import VotingClassifier
from prepare_data import load_and_prepare_data
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
import warnings
warnings.filterwarnings('ignore')

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ========================================
# 1. CHARGEMENT DES DONNEES
# ========================================
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "btc_hourly.csv")
X, y, scaler = load_and_prepare_data(data_path, scale=True)

print(f"\n[DATA] Donnees chargees:")
print(f"   Shape: {X.shape}")
print(f"   Classes: {y.value_counts().to_dict()}")

# Split temporel
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}\n")

mlflow.set_experiment("BTC_Direction_V4_Bayesian_Optimization")

# ========================================
# 2. BAYESIAN OPTIMIZATION POUR LIGHTGBM
# ========================================
print("[LIGHTGBM] Bayesian Optimization en cours...\n")

def objective_lightgbm(params):
    """Fonction objective pour LightGBM"""
    n_estimators, max_depth, learning_rate, num_leaves = params
    
    model = LGBMClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        num_leaves=int(num_leaves),
        random_state=42,
        verbose=-1
    )
    
    # Cross-validation sur F1-score
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
    return -scores.mean()  # Minimiser = maximiser F1

# Espace de parametres
space_lightgbm = [
    Integer(100, 300, name='n_estimators'),
    Integer(5, 15, name='max_depth'),
    Real(0.01, 0.2, name='learning_rate'),
    Integer(20, 50, name='num_leaves'),
]

result_lightgbm = gp_minimize(
    objective_lightgbm,
    space_lightgbm,
    n_calls=10,  # 10 iterations
    random_state=42,
    verbose=0
)

best_params_lgb = {
    'n_estimators': int(result_lightgbm.x[0]),
    'max_depth': int(result_lightgbm.x[1]),
    'learning_rate': result_lightgbm.x[2],
    'num_leaves': int(result_lightgbm.x[3]),
}

print(f"[OK] Best params: {best_params_lgb}")

lgb_model = LGBMClassifier(**best_params_lgb, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]

lgb_metrics = {
    'acc': accuracy_score(y_test, y_pred_lgb),
    'prec': precision_score(y_test, y_pred_lgb),
    'rec': recall_score(y_test, y_pred_lgb),
    'f1': f1_score(y_test, y_pred_lgb),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_lgb),
}

print(f"LightGBM Results: F1={lgb_metrics['f1']:.4f}, AUC={lgb_metrics['roc_auc']:.4f}\n")

# ========================================
# 3. BAYESIAN OPTIMIZATION POUR XGBOOST
# ========================================
print("[XGBOOST] Bayesian Optimization en cours...\n")

def objective_xgboost(params):
    """Fonction objective pour XGBoost"""
    n_estimators, max_depth, learning_rate = params
    
    model = XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        random_state=42,
        verbosity=0
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
    return -scores.mean()

space_xgboost = [
    Integer(100, 300, name='n_estimators'),
    Integer(5, 15, name='max_depth'),
    Real(0.01, 0.2, name='learning_rate'),
]

result_xgboost = gp_minimize(
    objective_xgboost,
    space_xgboost,
    n_calls=10,
    random_state=42,
    verbose=0
)

best_params_xgb = {
    'n_estimators': int(result_xgboost.x[0]),
    'max_depth': int(result_xgboost.x[1]),
    'learning_rate': result_xgboost.x[2],
}

print(f"[OK] Best params: {best_params_xgb}")

xgb_model = XGBClassifier(**best_params_xgb, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

xgb_metrics = {
    'acc': accuracy_score(y_test, y_pred_xgb),
    'prec': precision_score(y_test, y_pred_xgb),
    'rec': recall_score(y_test, y_pred_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb),
}

print(f"XGBoost Results: F1={xgb_metrics['f1']:.4f}, AUC={xgb_metrics['roc_auc']:.4f}\n")

# ========================================
# 4. CATBOOST (fast, auto-tuned)
# ========================================
print("[CATBOOST] Training...\n")

cb_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=7,
    random_state=42,
    verbose=0
)

cb_model.fit(X_train, y_train)
y_pred_cb = cb_model.predict(X_test)
y_pred_proba_cb = cb_model.predict_proba(X_test)[:, 1]

cb_metrics = {
    'acc': accuracy_score(y_test, y_pred_cb),
    'prec': precision_score(y_test, y_pred_cb),
    'rec': recall_score(y_test, y_pred_cb),
    'f1': f1_score(y_test, y_pred_cb),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_cb),
}

print(f"CatBoost Results: F1={cb_metrics['f1']:.4f}, AUC={cb_metrics['roc_auc']:.4f}\n")

# ========================================
# 5. VOTING ENSEMBLE (LightGBM + XGBoost uniquement)
# ========================================
print("[ENSEMBLE] Voting Classifier...\n")

voting = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('xgb', xgb_model),
    ],
    voting='soft'
)

voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)
y_pred_proba_voting = voting.predict_proba(X_test)[:, 1]

voting_metrics = {
    'acc': accuracy_score(y_test, y_pred_voting),
    'prec': precision_score(y_test, y_pred_voting),
    'rec': recall_score(y_test, y_pred_voting),
    'f1': f1_score(y_test, y_pred_voting),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_voting),
}

print(f"Voting Results: F1={voting_metrics['f1']:.4f}, AUC={voting_metrics['roc_auc']:.4f}\n")

# ========================================
# 6. RESUME FINAL
# ========================================
print("="*70)
print("[FINAL RESULTS] BAYESIAN OPTIMIZATION + ENSEMBLE")
print("="*70)

all_results = {
    'LightGBM': lgb_metrics,
    'XGBoost': xgb_metrics,
    'CatBoost': cb_metrics,
    'VotingEnsemble': voting_metrics,
}

for model_name, metrics in sorted(all_results.items(), key=lambda x: x[1]['f1'], reverse=True):
    print(f"\n{model_name:18} | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f} | AUC: {metrics['roc_auc']:.4f}")
    print(f"                   | Prec: {metrics['prec']:.4f} | Recall: {metrics['rec']:.4f}")

print("\n" + "="*70)

# ========================================
# 7. LOG DANS MLFLOW
# ========================================
for model_name, metrics in all_results.items():
    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric("accuracy", metrics['acc'])
        mlflow.log_metric("precision", metrics['prec'])
        mlflow.log_metric("recall", metrics['rec'])
        mlflow.log_metric("f1_score", metrics['f1'])
        mlflow.log_metric("roc_auc", metrics['roc_auc'])

# Log des meilleurs params dans un run separe
with mlflow.start_run(run_name="BestParams_LightGBM"):
    for key, value in best_params_lgb.items():
        mlflow.log_param(key, value)

with mlflow.start_run(run_name="BestParams_XGBoost"):
    for key, value in best_params_xgb.items():
        mlflow.log_param(key, value)

print("\n[DONE] All models trained and logged to MLflow!")
