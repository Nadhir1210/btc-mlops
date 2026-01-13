"""ZenML Pipeline Orchestrator for BTC MLOps."""

from zenml import pipeline
from .steps import (
    prepare_data_step,
    train_model_step,
    evaluate_model_step,
    export_model_step,
)


@pipeline
def btc_training_pipeline():
    """
    ZenML pipeline for BTC price prediction model training and deployment.
    
    Pipeline Flow:
    1. prepare_data_step()
       └─> X_train, X_test, y_train, y_test, scaler
    2. train_model_step(X_train, y_train)
       └─> model, model_path
    3. evaluate_model_step(model, X_test, y_test)
       └─> metrics
    4. export_model_step(model, model_path, metrics)
       └─> model_uri
    
    Each step is independent and can be cached/reused.
    """
    # Step 1: Prepare Data
    X_train, X_test, y_train, y_test, scaler = prepare_data_step()
    
    # Step 2: Train Model
    model, model_path = train_model_step(
        X_train=X_train,
        y_train=y_train
    )
    
    # Step 3: Evaluate Model
    metrics = evaluate_model_step(
        model=model,
        X_test=X_test,
        y_test=y_test
    )
    
    # Step 4: Export Model
    model_uri = export_model_step(
        model=model,
        model_path=model_path,
        metrics=metrics
    )
    
    return model_uri


if __name__ == "__main__":
    # Run pipeline
    pipeline_instance = btc_training_pipeline()
    pipeline_instance.run()
