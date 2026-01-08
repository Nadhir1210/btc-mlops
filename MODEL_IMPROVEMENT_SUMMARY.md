# BTC MLOps - Project Summary

## Amelioration du modele: V1 → V4

### Baseline (V1)
```
- Simple RandomForest
- 10 features basiques
- F1-Score: 0.2514
- Pas de tuning
```

### Final (V4 - GPU Optimized)
```
- Bayesian Optimization
- 43 features engineered
- 3 modeles (LightGBM, XGBoost, CatBoost)
- F1-Score: 0.4736 (+88%)
- Catboost MEILLEUR
```

---

## Files dans training/

### Scripts principaux
- `prepare_data.py` - Feature engineering (43 features)
- `train_mlflow.py` - V1 baseline
- `train_bayesian_optimization.py` - V4 FINAL with Bayesian Opt
- `evaluate_models.py` - Evaluation detaillee
- `inference.py` - Production inference

### Modeles entraines
- `catboost_best.pkl` - **MEILLEUR** (F1=0.4736)
- `lightgbm_best.pkl` (F1=0.3960)
- `xgboost_best.pkl` (F1=0.3080)

---

## Resultats CatBoost (Meilleur)

```
Accuracy:  52.62%
Precision: 54.54%
Recall:    41.85%
F1-Score:  0.4736
ROC-AUC:   0.5346

Confusion Matrix:
  TN: 6005 | FP: 3408
  FN: 5681 | TP: 4088
```

**Interpretation:**
- Detecte 41.85% des hausses (Recall)
- 54.54% des signaux UP sont corrects (Precision)
- Meilleur equilibre entre Precision/Recall

---

## Top 10 Features (importants pour predictions)

1. STOCH_K (Stochastic oscillator K)
2. volume_ratio (Ratio volume/moyenne)
3. momentum_5 (Momentum 5H)
4. STOCH_D (Stochastic %D)
5. HIGH_LOW_RATIO (Ratio Haut/Bas)
6. VOLUME (Volume de trading)
7. BB_WIDTH (Largeur Bollinger Bands)
8. bb_position (Position dans BB)
9. RSI (Relative Strength Index)
10. VOLATILITY_30D (Volatilite 30 jours)

---

## Architecture

```
btc-mlops/
├── data/
│   └── btc_hourly.csv (95,906 rows x 22 cols)
│
├── training/
│   ├── prepare_data.py *** Feature Engineering ***
│   ├── train_bayesian_optimization.py *** MAIN ***
│   ├── evaluate_models.py *** Model Evaluation ***
│   ├── inference.py *** Production Use ***
│   ├── catboost_best.pkl *** BEST MODEL ***
│   ├── lightgbm_best.pkl
│   └── xgboost_best.pkl
│
├── api/ (TODO: FastAPI inference service)
├── drift/ (TODO: Data drift monitoring)
├── streamlit_app/ (TODO: Dashboard)
│
├── Dockerfile
├── requirements.txt
├── IMPROVEMENTS.md (detailed report)
└── README.md
```

---

## Comment utiliser

### 1. Evaluation detaillee
```bash
cd training
python evaluate_models.py
```

### 2. Inference sur nouvelles donnees
```bash
python inference.py
```

### 3. Voir les experiements MLflow
```bash
mlflow ui  # http://localhost:5000
```

---

## Prochaines ameliorations

- [ ] API FastAPI pour serving le modele
- [ ] Dashboard Streamlit pour visualizations
- [ ] Monitoring de data drift
- [ ] Retraining automatique (tous les mois?)
- [ ] Docker containerization
- [ ] Tests unitaires
- [ ] A/B testing entre modeles

---

## Techniques utilisees

✅ **Feature Engineering** - 43 features avancees (RSI, ATR, Bollinger Bands, etc.)
✅ **Data Normalization** - StandardScaler pour stabilite
✅ **Bayesian Optimization** - Plus rapide que GridSearch
✅ **Ensemble Methods** - Voting classifier
✅ **MLflow Tracking** - Experimentation et versioning
✅ **Multiple Models** - LightGBM, XGBoost, CatBoost
✅ **Time Series Split** - Respect de la temporalite

---

**Status**: COMPLETE ✅  
**Best Model**: CatBoost  
**F1-Score**: 0.4736 (+88% vs baseline)  
**Date**: January 7, 2026  
