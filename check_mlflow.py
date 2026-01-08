#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check MLflow experiments and populate if empty"""

from mlflow.tracking import MlflowClient
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

client = MlflowClient()
exps = client.search_experiments()

print(f"\n=== MLFLOW STATUS ===")
print(f"Total experiments: {len(exps)}")

for e in exps:
    print(f"\nExperiment: {e.name} (ID: {e.experiment_id})")
    runs = client.search_runs(e.experiment_id)
    print(f"  Runs: {len(runs)}")
    
if len(exps) == 0:
    print("\n[WARNING] MLflow is empty! Creating default experiment...")
    exp_id = client.create_experiment("btc-prediction")
    print(f"[OK] Created experiment with ID: {exp_id}")

print("\n=== MLflow ready for logging ===")
