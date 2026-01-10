#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI service for BTC price direction prediction
Loads model from MLflow Model Registry for versioned production deployment
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

# Add training path for imports
# Add training path for imports
TRAINING_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'training')
sys.path.insert(0, TRAINING_PATH)

# MLflow configuration
MLFLOW_TRACKING_URI = f"sqlite:///{TRAINING_PATH}/mlflow.db"
MODEL_NAME = "BTC_CatBoost_Production"
MODEL_FALLBACK_PATH = os.path.join(TRAINING_PATH, 'catboost_best.pkl')

app = FastAPI(
    title="BTC Direction Prediction API",
    description="API pour predire la direction du prix Bitcoin (UP/DOWN) - Modele versionne via MLflow",
    version="1.1.0"
)

# Global model and metadata
model = None
model_info = {
    "source": None,
    "version": None,
    "run_id": None,
    "metrics": {}
}

# Feature names (43 features from prepare_data.py)
FEATURE_NAMES = [
    'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USD',
    'returns_1h', 'returns_2h', 'returns_4h', 'returns_8h', 'returns_24h',
    'volatility_4h', 'volatility_8h', 'volatility_24h',
    'ma_5', 'ma_10', 'ma_20', 'ma_50',
    'ema_5', 'ema_10', 'ema_20',
    'momentum_5', 'momentum_10', 'momentum_20',
    'roc_5', 'roc_10',
    'RSI', 'ATR',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'BB_position',
    'STOCH_K', 'STOCH_D',
    'volume_ma_5', 'volume_ma_20', 'volume_ratio',
    'HIGH_LOW_RATIO', 'CLOSE_OPEN_RATIO',
    'hour', 'day_of_week', 'is_weekend'
]


class PredictionInput(BaseModel):
    """Input features for prediction"""
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [50000.0] * 43  # 43 features
            }
        }


class PredictionOutput(BaseModel):
    """Prediction result"""
    direction: str
    probability_up: float
    probability_down: float
    confidence: float
    signal_strength: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    model_source: str
    model_version: Optional[str]
    run_id: Optional[str]


class ModelInfoResponse(BaseModel):
    """Model information response"""
    name: str
    version: str
    source: str
    run_id: str
    metrics: dict


@app.on_event("startup")
async def load_model():
    """Load model from MLflow Model Registry or fallback to pickle"""
    global model, model_info
    
    # Try loading from MLflow Model Registry first
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        # Get latest version of registered model
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if versions:
            latest_version = max(versions, key=lambda x: int(x.version))
            model_uri = f"models:/{MODEL_NAME}/{latest_version.version}"
            
            model = mlflow.sklearn.load_model(model_uri)
            
            # Get run metrics
            run = client.get_run(latest_version.run_id)
            metrics = {k: round(v, 4) for k, v in run.data.metrics.items()}
            
            model_info = {
                "source": "MLflow Model Registry",
                "version": latest_version.version,
                "run_id": latest_version.run_id,
                "metrics": metrics
            }
            
            print(f"[OK] Model loaded from MLflow Registry: {MODEL_NAME} v{latest_version.version}")
            print(f"     Run ID: {latest_version.run_id}")
            print(f"     Metrics: F1={metrics.get('f1_score', 'N/A')}, AUC={metrics.get('roc_auc', 'N/A')}")
            return
            
    except Exception as e:
        print(f"[WARNING] Could not load from MLflow: {e}")
    
    # Fallback to pickle file
    try:
        with open(MODEL_FALLBACK_PATH, 'rb') as f:
            model = pickle.load(f)
        model_info = {
            "source": "Local pickle file",
            "version": "local",
            "run_id": None,
            "metrics": {}
        }
        print(f"[OK] Model loaded from fallback: {MODEL_FALLBACK_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model = None


@app.get("/", tags=["Info"])
async def root():
    """API root endpoint"""
    return {
        "name": "BTC Direction Prediction API",
        "version": "1.1.0",
        "model_source": model_info.get("source", "Unknown"),
        "model_version": model_info.get("version", "Unknown"),
        "endpoints": {
            "/predict": "POST - Make prediction",
            "/health": "GET - Health check",
            "/features": "GET - List required features",
            "/model-info": "GET - Model details from MLflow"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API health and model status"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type=type(model).__name__ if model else "None",
        model_source=model_info.get("source", "Unknown"),
        model_version=str(model_info.get("version", "")),
        run_id=model_info.get("run_id")
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
async def get_model_info():
    """Get detailed model information from MLflow"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        name=MODEL_NAME,
        version=str(model_info.get("version", "unknown")),
        source=model_info.get("source", "unknown"),
        run_id=model_info.get("run_id") or "unknown",
        metrics=model_info.get("metrics", {})
    )


@app.get("/features", tags=["Info"])
async def get_features():
    """Get list of required features"""
    return {
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
async def predict(input_data: PredictionInput):
    """
    Predict BTC price direction
    
    - **features**: List of 43 numerical features
    
    Returns prediction with probability and confidence
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate input
    if len(input_data.features) != 43:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 43 features, got {len(input_data.features)}"
        )
    
    try:
        # Prepare features
        X = np.array(input_data.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        prob_down = probabilities[0]
        prob_up = probabilities[1]
        confidence = max(prob_up, prob_down) * 100
        
        # Determine signal strength
        if confidence >= 70:
            signal_strength = "STRONG"
        elif confidence >= 60:
            signal_strength = "MODERATE"
        else:
            signal_strength = "WEAK"
        
        return PredictionOutput(
            direction="UP" if prediction == 1 else "DOWN",
            probability_up=round(prob_up, 4),
            probability_down=round(prob_down, 4),
            confidence=round(confidence, 2),
            signal_strength=signal_strength
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
