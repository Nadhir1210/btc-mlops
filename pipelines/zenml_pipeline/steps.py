"""ZenML Pipeline Steps for BTC MLOps."""

import os
import pickle
from pathlib import Path
from typing import Tuple, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler
from zenml import step
import logging

logger = logging.getLogger(__name__)


@step
def prepare_data_step() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Prepare and engineer features for BTC price prediction.
    
    Returns:
        Tuple containing:
        - X_train: Training features
        - X_test: Test features
        - y_train: Training targets
        - y_test: Test targets
        - scaler: Fitted scaler object
    """
    from src.data.prepare_data import load_and_prepare_data

    logger.info("📊 Step 1: Preparing data...")

    # Load and prepare data
    X, y, scaler = load_and_prepare_data(
        "data/raw/btc_hourly.csv",
        scale=True,
    )

    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Ensure y is a DataFrame (ZenML expects DataFrame outputs)
    if isinstance(y, pd.Series):
        y = y.to_frame(name=getattr(y, "name", "target"))
    elif not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y)

    # Split data 80/20 and reset indices
    split_idx = int(0.8 * len(X))
    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_test = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_test = y.iloc[split_idx:].reset_index(drop=True)

    # Create output directory
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save processed data
    with open(processed_dir / "X_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    with open(processed_dir / "X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(processed_dir / "y_train.pkl", "wb") as f:
        pickle.dump(y_train, f)
    with open(processed_dir / "y_test.pkl", "wb") as f:
        pickle.dump(y_test, f)
    with open(processed_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"✅ Data prepared: {len(X_train)} train, {len(X_test)} test samples")

    return X_train, X_test, y_train, y_test, scaler


@step
def train_model_step(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Any, str]:
    """
    Train BTC price prediction model using CatBoost.
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Tuple containing:
        - model: Trained CatBoost model
        - model_path: Path to saved model
    """
    from catboost import CatBoostClassifier
    
    logger.info("🤖 Step 2: Training model...")
    
    # Initialize model
    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.03,
        depth=8,
        verbose=0,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    model_path = Path("src/training/catboost_model.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    logger.info(f"✅ Model trained and saved to {model_path}")
    
    return model, str(model_path)


@step
def evaluate_model_step(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score,
        recall_score, precision_score, confusion_matrix
    )
    import json
    
    logger.info("📈 Step 3: Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    
    # Save metrics
    metrics_path = Path("data/processed/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"✅ Metrics: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
    
    return metrics


@step
def export_model_step(
    model: Any,
    model_path: str,
    metrics: dict
) -> str:
    """
    Export model to MLflow registry for production deployment.
    
    Args:
        model: Trained model
        model_path: Path to saved model file
        metrics: Model evaluation metrics
        
    Returns:
        MLflow model URI
    """
    import mlflow
    import mlflow.sklearn
    from pathlib import Path
    
    logger.info("🚀 Step 4: Exporting model to MLflow...")
    
    # Set MLflow tracking to local file system
    mlflow_dir = Path("mlruns").absolute()
    mlflow_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlflow_dir}")
    
    # Create or get experiment
    experiment_name = "BTC_Price_Prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("algorithm", "CatBoost")
        mlflow.log_param("iterations", 200)
        mlflow.log_param("learning_rate", 0.03)
        mlflow.log_param("depth", 8)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
        
        # Register model
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="BTC_Price_Predictor"
        )
        
        model_uri = f"runs:/{run.info.run_id}/model"
        logger.info(f"✅ Model exported to MLflow: {model_uri}")
    
    return model_uri
