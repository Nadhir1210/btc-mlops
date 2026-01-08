#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Register best model to MLflow Model Registry for production API
Ensures model is properly versioned and tracked
"""

import mlflow
from mlflow.tracking import MlflowClient
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from prepare_data import load_and_prepare_data

print("=" * 60)
print("   MLFLOW MODEL REGISTRATION & VERIFICATION")
print("=" * 60)

# Initialize MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Check existing experiments
print("\n[1] EXISTING EXPERIMENTS:")
experiments = client.search_experiments()
for exp in experiments:
    runs = client.search_runs(exp.experiment_id)
    print(f"   - {exp.name} (ID: {exp.experiment_id}) | Runs: {len(runs)}")

# Load data
print("\n[2] LOADING DATA...")
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'btc_hourly.csv')
X, y, scaler = load_and_prepare_data(data_path, scale=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Load best model
print("\n[3] LOADING CATBOOST BEST MODEL...")
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'catboost_best.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"   Model type: {type(model).__name__}")

# Calculate metrics
print("\n[4] CALCULATING METRICS...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_proba)
}

for name, value in metrics.items():
    print(f"   {name}: {value:.4f}")

# Get model parameters
print("\n[5] MODEL PARAMETERS:")
params = {
    'iterations': model.get_param('iterations'),
    'depth': model.get_param('depth'),
    'learning_rate': model.get_param('learning_rate'),
    'l2_leaf_reg': model.get_param('l2_leaf_reg'),
    'random_seed': model.get_param('random_seed'),
}
for name, value in params.items():
    print(f"   {name}: {value}")

# Create/Get experiment for production
EXPERIMENT_NAME = "BTC_Production_Model"
print(f"\n[6] REGISTERING TO EXPERIMENT: {EXPERIMENT_NAME}")

try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        exp_id = client.create_experiment(EXPERIMENT_NAME)
        print(f"   Created new experiment: {exp_id}")
    else:
        exp_id = experiment.experiment_id
        print(f"   Using existing experiment: {exp_id}")
except Exception as e:
    exp_id = client.create_experiment(EXPERIMENT_NAME)
    print(f"   Created experiment: {exp_id}")

mlflow.set_experiment(EXPERIMENT_NAME)

# Log complete run with model
print("\n[7] LOGGING RUN TO MLFLOW...")
with mlflow.start_run(run_name="catboost_production_v1") as run:
    
    # Log parameters
    print("   - Logging parameters...")
    for name, value in params.items():
        mlflow.log_param(name, value)
    
    # Log additional training info
    mlflow.log_param("model_type", "CatBoostClassifier")
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])
    mlflow.log_param("feature_scaling", "StandardScaler")
    
    # Log metrics
    print("   - Logging metrics...")
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
    
    # Log confusion matrix as metrics
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("true_negatives", int(cm[0, 0]))
    mlflow.log_metric("false_positives", int(cm[0, 1]))
    mlflow.log_metric("false_negatives", int(cm[1, 0]))
    mlflow.log_metric("true_positives", int(cm[1, 1]))
    
    # Log model
    print("   - Logging model artifact...")
    mlflow.sklearn.log_model(
        model, 
        artifact_path="model",
        registered_model_name="BTC_CatBoost_Production"
    )
    
    # Log feature names
    feature_names = list(X.columns)
    mlflow.log_param("features", ",".join(feature_names[:10]) + "...")
    
    # Log scaler
    scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact(scaler_path, "preprocessing")
    print("   - Logged scaler artifact")
    
    run_id = run.info.run_id
    print(f"\n   RUN ID: {run_id}")

# Verify registration
print("\n[8] VERIFYING MODEL REGISTRATION...")
try:
    registered_model = client.get_registered_model("BTC_CatBoost_Production")
    print(f"   Model Name: {registered_model.name}")
    
    versions = client.search_model_versions(f"name='{registered_model.name}'")
    print(f"   Versions: {len(versions)}")
    
    for v in versions:
        print(f"      - Version {v.version}: {v.current_stage} | Run: {v.run_id[:8]}...")
except Exception as e:
    print(f"   Warning: {e}")

# Final summary
print("\n" + "=" * 60)
print("   VERIFICATION COMPLETE")
print("=" * 60)
print(f"""
   Model: CatBoostClassifier
   Experiment: {EXPERIMENT_NAME}
   Run ID: {run_id}
   
   METRICS:
   - Accuracy:  {metrics['accuracy']:.4f}
   - Precision: {metrics['precision']:.4f}
   - Recall:    {metrics['recall']:.4f}
   - F1 Score:  {metrics['f1_score']:.4f}
   - ROC-AUC:   {metrics['roc_auc']:.4f}
   
   MLflow UI: http://127.0.0.1:5000
   API Ready: Model is versioned and ready for production
""")
