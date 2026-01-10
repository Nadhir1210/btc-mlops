"""
Training avec XGBoost, LightGBM, CatBoost (GPU-ready + rapide)
Version simplifiée et rapide (pas de GridSearch massif)
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from prepare_data import load_and_prepare_data
import numpy as np

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# ========================================
# 1. CHARGEMENT DES DONNEES
# ========================================
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "btc_hourly.csv")
X, y, scaler = load_and_prepare_data(data_path, scale=True)

print(f"\n[DATA] Donnees chargees:")
print(f"   X shape: {X.shape}")
print(f"   y distribution: {y.value_counts().to_dict()}")

# Split temporel
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"   Train: {X_train.shape}, Test: {X_test.shape}\n")

mlflow.set_experiment("BTC_Direction_V3_GPU_Optimized")

# ========================================
# 2. MODELES RAPIDES ET GPU-READY
# ========================================
models_to_train = []

# RandomForest (baseline rapide)
models_to_train.append((
    "RandomForest_Fast",
    RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
))

# LightGBM (ultra rapide, GPU-compatible)
if LIGHTGBM_AVAILABLE:
    models_to_train.append((
        "LightGBM_GPU",
        LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
            # device='gpu'  # Decommenter si GPU NVIDIA disponible
        )
    ))

# XGBoost (tres populaire, rapide)
if XGBOOST_AVAILABLE:
    models_to_train.append((
        "XGBoost_Fast",
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
            # tree_method='gpu_hist'  # Decommenter si GPU disponible
        )
    ))

# CatBoost (bonne precision, auto-optimization)
if CATBOOST_AVAILABLE:
    models_to_train.append((
        "CatBoost_Fast",
        CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=7,
            random_state=42,
            verbose=0,
            # gpu_devices=[0]  # Decommenter pour utiliser GPU
        )
    ))

# ========================================
# 3. ENTRAÎNEMENT SIMPLE (pas de GridSearch)
# ========================================
print("[TRAINING] Entraînement des modeles simples...\n")

results = {}
best_overall_model = None
best_overall_f1 = 0

for model_name, model in models_to_train:
    print(f"[{model_name}] Training en cours...")
    
    try:
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Avec probabilites pour ROC-AUC
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            y_pred_proba = None
            roc_auc = 0.0
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[model_name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "model": model
        }
        
        print(f"   [OK] Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        # Track meilleur modele
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_model = (model_name, model)
        
        # Log MLflow
        with mlflow.start_run(run_name=model_name):
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.sklearn.log_model(model, model_name.lower())
    
    except Exception as e:
        print(f"   [ERROR] {str(e)[:100]}")

# ========================================
# 4. VOTING ENSEMBLE (des meilleurs modeles)
# ========================================
if len(results) >= 2:
    print(f"\n[ENSEMBLE] Voting des meilleurs modeles...")
    
    from sklearn.ensemble import VotingClassifier
    
    voting_models = [
        (name, results[name]["model"]) 
        for name in list(results.keys())[:3]  # Top 3 models
    ]
    
    voting_clf = VotingClassifier(
        estimators=voting_models,
        voting="soft"
    )
    
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    
    try:
        y_pred_voting_proba = voting_clf.predict_proba(X_test)[:, 1]
        roc_auc_voting = roc_auc_score(y_test, y_pred_voting_proba)
    except:
        roc_auc_voting = 0.0
    
    acc_voting = accuracy_score(y_test, y_pred_voting)
    precision_voting = precision_score(y_test, y_pred_voting)
    recall_voting = recall_score(y_test, y_pred_voting)
    f1_voting = f1_score(y_test, y_pred_voting)
    
    print(f"   [OK] Acc: {acc_voting:.4f} | F1: {f1_voting:.4f} | ROC-AUC: {roc_auc_voting:.4f}")
    
    # Track meilleur
    if f1_voting > best_overall_f1:
        best_overall_f1 = f1_voting
        best_overall_model = ("VotingEnsemble", voting_clf)
    
    # Log MLflow
    with mlflow.start_run(run_name="VotingEnsemble"):
        mlflow.log_metric("accuracy", acc_voting)
        mlflow.log_metric("precision", precision_voting)
        mlflow.log_metric("recall", recall_voting)
        mlflow.log_metric("f1_score", f1_voting)
        mlflow.log_metric("roc_auc", roc_auc_voting)
        mlflow.sklearn.log_model(voting_clf, "voting_ensemble")
    
    results["VotingEnsemble"] = {
        "accuracy": acc_voting,
        "precision": precision_voting,
        "recall": recall_voting,
        "f1": f1_voting,
        "roc_auc": roc_auc_voting,
        "model": voting_clf
    }

# ========================================
# 5. RESUME FINAL
# ========================================
print("\n" + "="*70)
print("[SUMMARY] RESUME DE TOUS LES MODELES")
print("="*70)

for model_name, metrics in sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True):
    print(f"\n{model_name:25} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

print("\n" + "="*70)
print(f"[BEST MODEL] {best_overall_model[0]} avec F1-Score: {best_overall_f1:.4f}")
print("="*70 + "\n")

# ========================================
# 6. SAUVEGARDE DU MEILLEUR MODELE
# ========================================
import pickle

best_model_path = os.path.join(os.path.dirname(__file__), "best_model.pkl")
with open(best_model_path, "wb") as f:
    pickle.dump(best_overall_model[1], f)

print(f"[SAVED] Meilleur modele sauvegarde: {best_model_path}")
