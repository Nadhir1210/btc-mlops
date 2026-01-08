#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Populate MLflow with trained models and metrics
"""

import mlflow
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from prepare_data import load_and_prepare_data

# Setup paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load data
print("[DATA] Chargement des données...")
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'btc_hourly.csv')
X, y, scaler = load_and_prepare_data(data_path, scale=True)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

print(f"[OK] Données: Train {X_train.shape}, Test {X_test.shape}")

# Load trained models
models = {}
model_files = ['catboost_best.pkl', 'lightgbm_best.pkl', 'xgboost_best.pkl']

for model_file in model_files:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_file)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            models[model_file.replace('_best.pkl', '')] = pickle.load(f)
        print(f"[OK] Loaded {model_file}")
    else:
        print(f"[WARNING] {model_file} not found")

if not models:
    print("[ERROR] No models found. Train models first!")
    sys.exit(1)

# Log each model to MLflow
mlflow.set_experiment("btc-prediction")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Log parameters
        if hasattr(model, 'get_params'):
            params = model.get_params()
            for key, val in params.items():
                if isinstance(val, (int, float, str, bool)):
                    mlflow.log_param(key, val)
        
        # Log metrics
        for key, val in metrics.items():
            mlflow.log_metric(key, val)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name)
        
        print(f"\n[OK] {model_name}")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")

print("\n[DONE] All models logged to MLflow!")
print("[INFO] Access MLflow UI at http://127.0.0.1:5000")
