"""
Script d'inference - Comment utiliser le meilleur modele en production
"""
import os
import pickle
import pandas as pd
import numpy as np
from prepare_data import load_and_prepare_data

# ========================================
# 1. CHARGER LE MEILLEUR MODELE
# ========================================
model_path = os.path.join(os.path.dirname(__file__), 'catboost_best.pkl')

with open(model_path, 'rb') as f:
    best_model = pickle.load(f)

print(f"\n[LOADED] Best model: {model_path}")

# ========================================
# 2. CHARGER ET PREPARER LES DONNEES
# ========================================
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "btc_hourly.csv")
X, y, scaler = load_and_prepare_data(data_path, scale=True)

print(f"\n[DATA] Shape: {X.shape}")

# ========================================
# 3. FAIRE DES PREDICTIONS
# ========================================
# Sur les dernieres 10 observations
recent_data = X.tail(10)

predictions = best_model.predict(recent_data)
probabilities = best_model.predict_proba(recent_data)[:, 1]

print(f"\n[PREDICTIONS] Dernieres 10 observations:\n")
print(f"{'Index':>6} | {'Prediction':>12} | {'Probability (UP)':>20} | {'Direction':>12}")
print("-" * 65)

for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
    direction = "UP" if pred == 1 else "DOWN"
    confidence = "STRONG" if prob > 0.7 or prob < 0.3 else "WEAK"
    print(f"{idx:>6} | {pred:>12} | {prob:>19.2%} | {direction:>7} ({confidence:>6})")

# ========================================
# 4. PREDICTION POUR LA PROCHAINE HEURE
# ========================================
print(f"\n" + "="*65)
print("PREDICTION POUR LA PROCHAINE HEURE")
print("="*65)

latest_pred = predictions[-1]
latest_prob = probabilities[-1]

if latest_pred == 1:
    direction = "HAUSSIER (Prix UP)"
    confidence = latest_prob
else:
    direction = "BAISSIER (Prix DOWN)"
    confidence = 1 - latest_prob

print(f"\nDirection: {direction}")
print(f"Confiance: {confidence:.2%}")

if confidence > 0.6:
    print("\n[SIGNAL] *** SIGNAL FORT ***")
elif confidence > 0.55:
    print("\n[SIGNAL] Signal modere")
else:
    print("\n[SIGNAL] Signal faible (probabilite proche de 50%)")

# ========================================
# 5. STATISTIQUES
# ========================================
print(f"\n" + "="*65)
print("STATISTIQUES")
print("="*65)

print(f"\nSur les 10 dernieres heures:")
print(f"  Predictions UP: {(predictions == 1).sum()}/10")
print(f"  Predictions DOWN: {(predictions == 0).sum()}/10")
print(f"  Moyenne probabilite UP: {probabilities.mean():.2%}")
print(f"  Min probabilite: {probabilities.min():.2%}")
print(f"  Max probabilite: {probabilities.max():.2%}")

# ========================================
# 6. FEATURE IMPORTANCE
# ========================================
print(f"\n" + "="*65)
print("TOP 5 FEATURES POUR CETTE PREDICTION")
print("="*65)

if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(5)
    
    print(f"\n")
    for idx, row in fi.iterrows():
        print(f"  {row['Feature']:20} : {row['Importance']:.4f}")

print(f"\n[DONE] Inference complete!")
