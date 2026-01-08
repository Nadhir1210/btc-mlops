# Data Drift Detection & Simulation Guide

## 📊 Vue d'ensemble

Le **Data Drift** est un problème critique en MLOps. Quand les données en production changent de distribution, les performances du modèle dégradent. Ce projet implémente une détection **multi-méthodes** et des simulations de drift.

---

## 🔍 Modules créés

### 1. **detect_drift.py** - Détecteur de Drift

Classe `DataDriftDetector` avec 3 méthodes statistiques :

#### **Kolmogorov-Smirnov (KS) Test**
```
Usage: Détecte différences dans les distributions
Sensibilité: Queues de distribution
Seuil: p-value < 0.05
Cas: Excellente base de comparaison
```

#### **Population Stability Index (PSI)**
```
Formula: Σ (curr% - ref%) * ln(curr% / ref%)
Interprétation:
  PSI < 0.05      → Pas de drift
  PSI 0.05-0.1    → Drift faible
  PSI 0.1-0.25    → Drift modéré
  PSI > 0.25      → Drift significatif
```

#### **Welch t-test**
```
Usage: Compare means de deux distributions
Avantage: Pas d'assomption d'égalité des variances
Robustesse: Bon pour données réelles
```

#### Exemple :
```python
from detect_drift import DataDriftDetector

# Charger données de référence (entraînement)
reference_data = pd.read_csv('training/data_processed.csv')

# Charger données actuelles (production)
current_data = pd.read_csv('production_data.csv')

# Créer détecteur
detector = DataDriftDetector(reference_data)

# Détecter drift avec toutes les méthodes
results = detector.detect_drift(
    current_data,
    methods=['ks', 'psi', 'ttest'],
    ks_threshold=0.05,
    psi_threshold=0.1,
    ttest_threshold=0.05
)

# Afficher rapport
detector.print_report()

# Sauvegarder résultats
detector.save_results('drift_results.json')
```

---

### 2. **simulate_drift.py** - Simulateur de Drift

Classe `DriftSimulator` pour générer différents types de drift :

#### **Mean Shift**
```
Scenario: Prix commencent à monter/descendre systématiquement
Impact: Changement de tendance du marché
```

#### **Variance Shift**
```
Scenario: Marché devient plus volatil
Impact: Incertitude accrue, oscillations plus larges
```

#### **Outlier Injection**
```
Scenario: Événements extrêmes (crash, pump)
Impact: 5-10% d'outliers injectés
```

#### **Covariate Shift**
```
Scenario: Corrélations entre features changent
Impact: Distributions marginales changent mais pas conditionnelles
```

#### **Concept Drift**
```
Scenario: Relation features-target change
Impact: Transformation non-linéaire appliquée
```

#### **Gradual Drift**
```
Scenario: Changement lent mais continu
Impact: 5 batches avec shift progressif
```

#### **Seasonal Shift**
```
Scenario: Patterns de marché changent selon la saison
Impact: Composante sinusoïdale ajoutée
```

#### Exemple :
```python
from simulate_drift import DriftSimulator, generate_drift_scenarios

# Charger données
data = pd.read_csv('training/data_processed.csv')

# Générer tous les scénarios
scenarios = generate_drift_scenarios(data)

# Accéder à un scénario
mean_shift_data = scenarios['mean_shift']

# Utiliser pour tester le détecteur
detector = DataDriftDetector(data)
results = detector.detect_drift(mean_shift_data)
```

---

### 3. **test_drift_detection.py** - Tests d'intégration

Script de test complet :

```bash
cd training
python test_drift_detection.py
```

**Résultats attendus** :

| Scénario | Drift Attendu | Résultat |
|----------|---------------|----------|
| baseline | Non | ✓ PASS |
| mean_shift | Oui | ✓ PASS |
| variance_shift | Oui | ✓ PASS |
| outlier_injection | Oui | ✓ PASS |
| concept_drift | Oui | ✓ PASS |
| gradual_drift | Oui | ✓ PASS |
| seasonal_shift | Oui | ✓ PASS |

---

## 🔄 Intégration CI/CD

### Workflow: `ml-training-pipeline.yml`

**JOB 1: check-drift**
```yaml
Steps:
  1. Checkout code
  2. Install dependencies (pandas, scipy)
  3. Detect data drift (run detect_drift.py)
  4. Parse results JSON
  5. Set retrain flag if drift detected
  6. Upload drift report as artifact
```

**Déclencheur** :
- Tous les **dimanches à 02:00 UTC**
- **Manuellement** avec `force_retrain=true`

**Outputs** :
- `should_retrain`: true/false
- `drift_detection_report`: JSON report
- Historique: 30 jours de rapports

---

## 📈 Workflow de production

```
┌─────────────────────┐
│   Production Data   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Drift Detection    │─────► JSON Report
│  (Weekly)           │       (Artifact)
└──────────┬──────────┘
           │
      ┌────┴────┐
      │          │
     ✓           ✗
     │          Drift Detected
     │           │
     │           ▼
     │    ┌──────────────────┐
     │    │ RETRAIN MODEL    │
     │    │ (Bayesian Opt)   │
     │    └────────┬─────────┘
     │             │
     │             ▼
     │    ┌──────────────────┐
     ▼    │ Test & Validate  │
  Continue │   New Model      │
     │     └────────┬─────────┘
     │              │
     └──────┬───────┘
            │
            ▼
     ┌──────────────────┐
     │ Deploy to Azure  │
     │ Container Apps   │
     └──────────────────┘
```

---

## 🎯 Seuils recommandés

### Pour données financières (BTC) :

```python
# Détection stricte (plus de retrainings)
ks_threshold = 0.01      # KS p-value très strict
psi_threshold = 0.05     # PSI très strict
ttest_threshold = 0.01   # t-test très strict

# Détection modérée (balance)
ks_threshold = 0.05      # KS p-value standard
psi_threshold = 0.10     # PSI modéré
ttest_threshold = 0.05   # t-test standard

# Détection souple (peu de retrainings)
ks_threshold = 0.10      # KS p-value souple
psi_threshold = 0.25     # PSI souple
ttest_threshold = 0.10   # t-test souple
```

---

## 🚀 Commandes utiles

### Tester localement :
```bash
cd training

# Test unitaire
python test_drift_detection.py

# Détection simple
python detect_drift.py

# Simulation simple
python simulate_drift.py
```

### Générer scénarios de drift :
```bash
python simulate_drift.py
# Génère: training/drift_scenarios/{scenario}.csv
```

### Checker les résultats :
```bash
# Afficher le JSON des résultats
cat drift_detection_results.json | jq .

# Ou juste les features avec drift
cat drift_detection_results.json | jq '.summary.drifted_features'
```

---

## 📊 Métriques de monitoring

Pour Azure Log Analytics, on peut envoyer :

```python
{
    "timestamp": "2026-01-08T12:30:00",
    "detection_method": "psi",
    "drifted_features": ["volume_sma", "rsi_14"],
    "drift_severity": "moderate",
    "recommended_action": "Monitor closely",
    "psi_scores": {
        "volume_sma": 0.15,
        "rsi_14": 0.12,
        "price_close": 0.03
    }
}
```

---

## 🔐 Bonnes pratiques

1. **Frequency** : Vérifier le drift **hebdomadairement** minimum
2. **Thresholds** : Adapter les seuils au domaine (finance = stricte)
3. **Actions** : Avoir une procédure de réaction définie
4. **Logging** : Tracker tous les drifts détectés
5. **Feedback** : Valider que le retraining améliore les performances

---

## ❌ Pièges courants

```python
# ❌ MAUVAIS: Ignorer le drift
# Modèles se dégradent silencieusement

# ✓ BON: Monitorer régulièrement
detector = DataDriftDetector(ref_data)
results = detector.detect_drift(prod_data)

# ❌ MAUVAIS: Un seul test statistique
# Peut être faux positif

# ✓ BON: Combiner plusieurs méthodes
methods=['ks', 'psi', 'ttest']  # Consensus requis

# ❌ MAUVAIS: Seuils trop souples
# Manque les drifts importants

# ✓ BON: Seuils adaptés au domaine
ks_threshold=0.05, psi_threshold=0.10
```

---

## 📝 Next Steps

- [ ] Intégrer alertes Slack sur drift détecté
- [ ] Ajouter métriques de performance modèle
- [ ] Implémenter récupération de données en batch
- [ ] Ajouter adaptive thresholds basés sur l'historique
- [ ] Dashboard de monitoring en temps réel
