"""
Analyse exploratoire des données BTC
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.prepare_data import load_and_prepare_data

# Charger les données
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "btc_hourly.csv")
X, y, scaler = load_and_prepare_data(data_path, scale=False)

print("\n" + "="*70)
print("📊 ANALYSE EXPLORATOIRE DES DONNÉES BTC")
print("="*70)

# 1️⃣ Statistiques générales
print("\n1️⃣ DISTRIBUTION DE LA TARGET")
print(f"   Classes: {y.value_counts().to_dict()}")
print(f"   Ratio positif/négatif: {y.value_counts()[1]/len(y):.2%}")

# 2️⃣ Statistiques des features
print("\n2️⃣ STATISTIQUES DES FEATURES")
print(X.describe())

# 3️⃣ Corrélations
print("\n3️⃣ TOP 10 FEATURES CORRÉLÉES À LA TARGET")
correlations = X.copy()
correlations["target"] = y
corr_with_target = correlations.corr()["target"].drop("target").abs().sort_values(ascending=False)
print(corr_with_target.head(10))

# 4️⃣ Distribution des features
print("\n4️⃣ NOMBRE DE FEATURES")
print(f"   Total: {X.shape[1]}")

# 5️⃣ Features avec données manquantes
print("\n5️⃣ DONNÉES MANQUANTES")
missing = X.isnull().sum()
if missing.sum() > 0:
    print(f"   Features avec NaN: {missing[missing > 0].to_dict()}")
else:
    print("   ✅ Aucune donnée manquante!")

# 6️⃣ Classe imbalancée?
print("\n6️⃣ BALANCE DES CLASSES")
balance = y.value_counts() / len(y) * 100
print(f"   Classe 0 (baisse): {balance[0]:.1f}%")
print(f"   Classe 1 (hausse): {balance[1]:.1f}%")

if abs(balance[0] - 50) > 10:
    print("   ⚠️  Classes imbalancées - considérer class_weight ou SMOTE")

print("\n" + "="*70)
