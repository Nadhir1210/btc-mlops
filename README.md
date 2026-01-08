# 🪙 BTC MLOps - Bitcoin Direction Prediction

Classification binaire pour prédire si le prix du Bitcoin va **monter** ou **baisser** à l'heure suivante.

## 📊 Données

- **Source** : `data/btc_hourly.csv`
- **Lignes** : ~95,925 observations
- **Fréquence** : Données horaires

### Features utilisées
- `OPEN`, `HIGH`, `LOW`, `CLOSE`, `VOLUME`
- `SMA_20`, `EMA_12`, `EMA_26`
- `MACD`, `MACD_SIGNAL`

### Target
- `1` → Prix va **monter** (CLOSE(t+1) > CLOSE(t))
- `0` → Prix va **baisser ou stagner**

---

## 🏗️ Architecture

```
btc-mlops/
├── data/
│   └── btc_hourly.csv          # Données brutes
│
├── training/
│   ├── prepare_data.py         # Préparation & nettoyage
│   └── train_mlflow.py         # Entraînement + tracking
│
├── api/                        # API FastAPI (WIP)
├── drift/                      # Data drift monitoring (WIP)
├── streamlit_app/              # Dashboard (WIP)
├── Dockerfile
└── README.md
```

---

## 🚀 Démarrage rapide

### 1️⃣ Installation des dépendances

```bash
pip install pandas scikit-learn mlflow
```

### 2️⃣ Entraînement du modèle

```bash
cd training
python train_mlflow.py
```

### 3️⃣ Visualiser les résultats (MLflow UI)

```bash
mlflow ui
```

Puis ouvrir : `http://localhost:5000`

---

## 📈 Étapes suivantes

- [ ] API FastAPI pour inférence
- [ ] Monitoring de data drift
- [ ] Dashboard Streamlit
- [ ] Docker containerization
- [ ] CI/CD pipeline

---

## ✅ État du projet

- ✔️ Préparation des données
- ✔️ Modèle RandomForest
- ✔️ MLflow tracking
- ⏳ API + Monitoring + UI

---

## 👨‍💻 Author

BTC MLOps Project - 2026
