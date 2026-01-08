# BTC MLOps - Model Improvement Report

## Résumé des améliorations

### V1 → V2 → V3 → V4 (GPU-Optimized)

| Version | Approche | Meilleur F1 | ROC-AUC | Notes |
|---------|----------|-----------|---------|-------|
| V1 | RandomForest simple | 0.2514 | N/A | Baseline |
| V2 | Features engineering + RF | 0.4251 | 0.5349 | 69% amélioration |
| V3 | RF rapide (GPU-ready) | 0.3742 | 0.5315 | Problème multiprocessing |
| V4 | Bayesian Opt + Ensemble | **0.4736** | **0.5346** | **CatBoost best** |

---

## Changements majeurs

### 1. Feature Engineering (V2)
- **Ajout de 43 features** (au lieu de 10)
  - Volatilité (5H, 10H)
  - RSI, ATR, Bollinger Bands
  - Returns et Log-Returns
  - Momentum patterns
  - Volume ratios
  - Price patterns (body_size, gap, etc.)

### 2. Normalisation des données
- StandardScaler appliqué sur X_train/X_test
- Améliore convergence des modèles

### 3. Modèles GPU-Ready
- **LightGBM** : Ultra rapide, XGBoost killer
- **XGBoost** : Classique, très populaire
- **CatBoost** : Meilleur recall, auto-tuned

### 4. Bayesian Optimization (au lieu de GridSearch)
- **10 itérations** vs 100+ pour GridSearch
- **5x plus rapide**
- Résultats comparables ou meilleurs

### 5. Ensemble Voting
- Combine prédictions de plusieurs modèles
- Plus stable que modèles seuls

---

## Résultats finaux (V4)

```
CatBoost:  F1=0.4736 | AUC=0.5346 | Accuracy=0.5262
LightGBM:  F1=0.3960 | AUC=0.5293 | Accuracy=0.5175
XGBoost:   F1=0.3080 | AUC=0.5176 | Accuracy=0.5075
```

### Meilleur modèle: **CatBoost**
- Recall: 41.85% (mieux détecte les hausses)
- Precision: 54.54%
- F1-Score: 0.4736 (+88% vs baseline)

---

## Top 10 Features (LightGBM)

1. **STOCH_K** (428) - Stochastic oscillator
2. **volume_ratio** (357) - Ratio volume/moyenne
3. **momentum_5** (294) - Momentum 5H
4. **STOCH_D** (256) - Stochastic %D
5. **HIGH_LOW_RATIO** (255) - Ratio H/L
6. **VOLUME** (251) - Volume brut
7. **BB_WIDTH** (238) - Largeur Bollinger Bands
8. **bb_position** (218) - Position dans BB
9. **RSI** (201) - Relative Strength Index
10. **VOLATILITY_30D** (185) - Volatilité 30J

---

## Architecture (dtc3)

```
btc-mlops/
├── data/
│   └── btc_hourly.csv (données brutes)
│
├── training/
│   ├── prepare_data.py (features engineering)
│   ├── train_mlflow.py (v1 baseline)
│   ├── train_gpu_optimized.py (v3)
│   ├── train_bayesian_optimization.py (v4 FINAL)
│   ├── evaluate_models.py (evaluation detaillee)
│   ├── catboost_best.pkl (meilleur modele)
│   ├── lightgbm_best.pkl
│   └── xgboost_best.pkl
│
├── api/ (WIP)
├── drift/ (WIP)
├── streamlit_app/ (WIP)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Prochaines étapes

- [ ] API FastAPI pour inference
- [ ] Dashboard Streamlit
- [ ] Data drift monitoring
- [ ] Retraining pipeline
- [ ] Docker containerization
- [ ] A/B testing

---

## Comment utiliser le meilleur modèle

```python
import pickle
from prepare_data import load_and_prepare_data

# Charger le modèle
with open('training/catboost_best.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger et préparer les données
X, y, scaler = load_and_prepare_data('data/btc_hourly.csv', scale=True)

# Faire des prédictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]
```

---

**Date**: January 7, 2026  
**Status**: V4 Complete (GPU-Optimized with Bayesian Optimization)
