"""
Entraînement avec tuning et ensemble methods
"""
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from prepare_data import load_and_prepare_data
import numpy as np

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

print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# ========================================
# 2. MODELS A TESTER
# ========================================
models = {
    "RandomForest": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.7, 0.9, 1.0],
        }
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.5, 1.0, 1.5],
        }
    ),
}

mlflow.set_experiment("BTC_Direction_Prediction_V2_Ensemble")

# ========================================
# 3. GRID SEARCH POUR CHAQUE MODEL
# ========================================
best_models = {}

for model_name, (model, params) in models.items():
    print(f"\n[GRID SEARCH] {model_name}...")
    
    grid_search = GridSearchCV(
        model, 
        params, 
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_models[model_name] = best_model
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n[OK] {model_name}")
    print(f"   Params: {grid_search.best_params_}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    # Log MLflow
    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(best_model, model_name.lower())

# ========================================
# 4. ENSEMBLE (VOTING CLASSIFIER)
# ========================================
print("\n[ENSEMBLE] Creation d'un Ensemble Voting Classifier...")

voting_clf = VotingClassifier(
    estimators=[
        ("rf", best_models["RandomForest"]),
        ("gb", best_models["GradientBoosting"]),
        ("ada", best_models["AdaBoost"]),
    ],
    voting="soft"
)

voting_clf.fit(X_train, y_train)

y_pred_ensemble = voting_clf.predict(X_test)
y_pred_ensemble_proba = voting_clf.predict_proba(X_test)[:, 1]

acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
precision_ensemble = precision_score(y_test, y_pred_ensemble)
recall_ensemble = recall_score(y_test, y_pred_ensemble)
f1_ensemble = f1_score(y_test, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_test, y_pred_ensemble_proba)

print(f"\n[OK] ENSEMBLE VOTING")
print(f"   Accuracy:  {acc_ensemble:.4f}")
print(f"   Precision: {precision_ensemble:.4f}")
print(f"   Recall:    {recall_ensemble:.4f}")
print(f"   F1-Score:  {f1_ensemble:.4f}")
print(f"   ROC-AUC:   {roc_auc_ensemble:.4f}")

# Log MLflow
with mlflow.start_run(run_name="Ensemble_Voting"):
    mlflow.log_metric("accuracy", acc_ensemble)
    mlflow.log_metric("precision", precision_ensemble)
    mlflow.log_metric("recall", recall_ensemble)
    mlflow.log_metric("f1_score", f1_ensemble)
    mlflow.log_metric("roc_auc", roc_auc_ensemble)
    mlflow.sklearn.log_model(voting_clf, "ensemble_voting")

# ========================================
# 5. STACKING SIMPLE (MOYENNE PONDEREE)
# ========================================
print("\n[STACKING] Creation d'un Stacking avec moyenne ponderee...")

# Predictions des modeles individuels
rf_proba = best_models["RandomForest"].predict_proba(X_test)[:, 1]
gb_proba = best_models["GradientBoosting"].predict_proba(X_test)[:, 1]
ada_proba = best_models["AdaBoost"].predict_proba(X_test)[:, 1]

# Moyenne ponderee (poids bases sur F1 scores)
weights = np.array([
    f1_score(y_test, best_models["RandomForest"].predict(X_test)),
    f1_score(y_test, best_models["GradientBoosting"].predict(X_test)),
    f1_score(y_test, best_models["AdaBoost"].predict(X_test)),
])
weights = weights / weights.sum()

y_pred_stack_proba = (
    weights[0] * rf_proba + 
    weights[1] * gb_proba + 
    weights[2] * ada_proba
)
y_pred_stack = (y_pred_stack_proba > 0.5).astype(int)

acc_stack = accuracy_score(y_test, y_pred_stack)
precision_stack = precision_score(y_test, y_pred_stack)
recall_stack = recall_score(y_test, y_pred_stack)
f1_stack = f1_score(y_test, y_pred_stack)
roc_auc_stack = roc_auc_score(y_test, y_pred_stack_proba)

print(f"\n[OK] STACKING PONDERE (poids: {weights.round(3)})")
print(f"   Accuracy:  {acc_stack:.4f}")
print(f"   Precision: {precision_stack:.4f}")
print(f"   Recall:    {recall_stack:.4f}")
print(f"   F1-Score:  {f1_stack:.4f}")
print(f"   ROC-AUC:   {roc_auc_stack:.4f}")

# Log MLflow
with mlflow.start_run(run_name="Stacking_Weighted"):
    mlflow.log_params({
        "rf_weight": weights[0],
        "gb_weight": weights[1],
        "ada_weight": weights[2],
    })
    mlflow.log_metric("accuracy", acc_stack)
    mlflow.log_metric("precision", precision_stack)
    mlflow.log_metric("recall", recall_stack)
    mlflow.log_metric("f1_score", f1_stack)
    mlflow.log_metric("roc_auc", roc_auc_stack)

# ========================================
# 6. RESUME COMPARATIF
# ========================================
print("\n" + "="*70)
print("[SUMMARY] RESUME COMPARATIF DE TOUS LES MODELES")
print("="*70)

results = {
    "RandomForest": {
        "Accuracy": acc,
        "F1": f1_score(y_test, best_models["RandomForest"].predict(X_test)),
        "ROC-AUC": roc_auc_score(y_test, best_models["RandomForest"].predict_proba(X_test)[:, 1]),
    },
    "GradientBoosting": {
        "Accuracy": accuracy_score(y_test, best_models["GradientBoosting"].predict(X_test)),
        "F1": f1_score(y_test, best_models["GradientBoosting"].predict(X_test)),
        "ROC-AUC": roc_auc_score(y_test, best_models["GradientBoosting"].predict_proba(X_test)[:, 1]),
    },
    "AdaBoost": {
        "Accuracy": accuracy_score(y_test, best_models["AdaBoost"].predict(X_test)),
        "F1": f1_score(y_test, best_models["AdaBoost"].predict(X_test)),
        "ROC-AUC": roc_auc_score(y_test, best_models["AdaBoost"].predict_proba(X_test)[:, 1]),
    },
    "Ensemble Voting": {
        "Accuracy": acc_ensemble,
        "F1": f1_ensemble,
        "ROC-AUC": roc_auc_ensemble,
    },
    "Stacking Pondere": {
        "Accuracy": acc_stack,
        "F1": f1_stack,
        "ROC-AUC": roc_auc_stack,
    },
}

for model, metrics in results.items():
    print(f"\n{model:20} | Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")

print("\n" + "="*70)
