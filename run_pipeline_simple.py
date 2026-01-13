"""
Simple Pipeline Runner for BTC MLOps.

This script runs the ML pipeline steps directly without ZenML complexity.
Use this if ZenML has database or initialization issues.

Usage:
    python run_pipeline_simple.py
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Run the 4-step ML pipeline directly."""
    
    print("\n" + "=" * 70)
    print("🚀 BTC MLOps Pipeline - Direct Execution")
    print("=" * 70)
    
    try:
        # ========== Step 1: Prepare Data ==========
        print("\n📊 Step 1/4: Preparing data...")
        from src.data.prepare_data import load_and_prepare_data
        import pickle
        
        X, y, scaler = load_and_prepare_data(
            "data/raw/btc_hourly.csv",
            scale=True
        )
        
        # Split 80/20
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Save processed data
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        print(f"   ✅ Data prepared: {len(X_train)} train, {len(X_test)} test samples")
        print(f"   ✅ Features: {X_train.shape[1]} columns")
        print(f"   ✅ Saved to: data/processed/")
        
        # ========== Step 2: Train Model ==========
        print("\n🤖 Step 2/4: Training CatBoost model...")
        from catboost import CatBoostClassifier
        
        model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.03,
            depth=8,
            verbose=50,  # Show progress every 50 iterations
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Save model
        model_path = Path("src/training/catboost_model.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        print(f"   ✅ Model trained and saved to: {model_path}")
        
        # ========== Step 3: Evaluate Model ==========
        print("\n📈 Step 3/4: Evaluating model...")
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            recall_score, precision_score, classification_report
        )
        import json
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "recall": float(recall_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
        }
        
        # Save metrics
        with open(processed_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   ✅ Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   ✅ F1 Score:  {metrics['f1']:.4f}")
        print(f"   ✅ ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"   ✅ Recall:    {metrics['recall']:.4f}")
        print(f"   ✅ Precision: {metrics['precision']:.4f}")
        
        # ========== Step 4: Export to MLflow ==========
        print("\n🚀 Step 4/4: Exporting to MLflow...")
        try:
            import mlflow
            import mlflow.sklearn
            
            # Set experiment
            mlflow.set_experiment("BTC_Price_Prediction")
            
            with mlflow.start_run(run_name="pipeline_run") as run:
                # Log parameters
                mlflow.log_param("algorithm", "CatBoost")
                mlflow.log_param("iterations", 200)
                mlflow.log_param("learning_rate", 0.03)
                mlflow.log_param("depth", 8)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("features", X_train.shape[1])
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name="BTC_Price_Predictor"
                )
                
                model_uri = f"runs:/{run.info.run_id}/model"
                print(f"   ✅ Model logged to MLflow")
                print(f"   ✅ Run ID: {run.info.run_id}")
                print(f"   ✅ Model URI: {model_uri}")
                
        except Exception as e:
            print(f"   ⚠️ MLflow export skipped: {e}")
            print(f"   ✅ Model saved locally at: {model_path}")
        
        # ========== Summary ==========
        print("\n" + "=" * 70)
        print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\n📁 Output Files:")
        print("   - data/processed/X_train.pkl")
        print("   - data/processed/X_test.pkl")
        print("   - data/processed/y_train.pkl")
        print("   - data/processed/y_test.pkl")
        print("   - data/processed/scaler.pkl")
        print("   - data/processed/metrics.json")
        print("   - src/training/catboost_model.pkl")
        
        print("\n📊 Final Metrics:")
        print(f"   - Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   - F1 Score:  {metrics['f1']:.4f}")
        print(f"   - ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
