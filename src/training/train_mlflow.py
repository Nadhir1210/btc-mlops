import os
import sys

# Ensure root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.prepare_data import load_and_prepare_data

# Charger données
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "btc_hourly.csv")
X, y, scaler = load_and_prepare_data(data_path, scale=True)

# Split temporel (IMPORTANT pour séries temporelles)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Configuration MLflow
mlflow_db_path = os.path.join(os.path.dirname(__file__), "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")
mlflow.set_experiment("BTC-Direction-Prediction")

with mlflow.start_run(run_name="RandomForest_v2"):

    # Update v2: class_weight="balanced" + more estimators
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcul des métriques
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Logs MLflow
    mlflow.log_param("model", "RandomForest_v2")
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("num_features", X.shape[1])
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="BTC_CatBoost_Production"
    )

    print(f"\n✅ Entraînement v2 terminé!")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\n📊 Modèle sauvegardé dans MLflow")
