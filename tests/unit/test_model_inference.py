"""
Unit tests for model inference and serving
Tests the FastAPI serving module
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.mark.unit
class TestModelLoading:
    """Test model loading and initialization"""
    
    def test_trained_model_exists(self):
        """Test that trained model file exists"""
        model_path = Path(__file__).parent.parent.parent / "src" / "training" / "catboost_best.pkl"
        assert model_path.exists(), f"Model not found at {model_path}"
    
    def test_scaler_exists(self):
        """Test that scaler file exists"""
        scaler_path = Path(__file__).parent.parent.parent / "data" / "processed" / "scaler.pkl"
        assert scaler_path.exists(), f"Scaler not found at {scaler_path}"
    
    def test_model_can_be_loaded(self, trained_model):
        """Test that model can be loaded successfully"""
        assert trained_model is not None, "Model should load successfully"


@pytest.mark.unit
class TestModelPrediction:
    """Test model prediction functionality"""
    
    def test_model_prediction_output_shape(self, trained_model, sample_features):
        """Test that model returns correct output shape"""
        # Reshape single sample to 2D array
        X = sample_features.reshape(1, -1)
        
        prediction = trained_model.predict(X)
        
        assert prediction.shape == (1,), "Prediction should return single value per sample"
        assert prediction[0] in [0, 1], "Prediction should be binary (0 or 1)"
    
    def test_model_probability_output(self, trained_model, sample_features):
        """Test that model returns valid probabilities"""
        X = sample_features.reshape(1, -1)
        
        proba = trained_model.predict_proba(X)
        
        assert proba.shape == (1, 2), "Probabilities should be (n_samples, 2)"
        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities should be in [0, 1]"
        assert np.isclose(proba.sum(axis=1)[0], 1.0), "Probabilities should sum to 1"
    
    def test_model_batch_prediction(self, trained_model):
        """Test model prediction with batch input"""
        np.random.seed(42)
        batch_size = 10
        X = np.random.randn(batch_size, 43)
        
        predictions = trained_model.predict(X)
        probas = trained_model.predict_proba(X)
        
        assert len(predictions) == batch_size, "Should return prediction for each sample"
        assert probas.shape == (batch_size, 2), "Should return probabilities for each sample"
    
    def test_model_deterministic_output(self, trained_model, sample_features):
        """Test that model gives same output for same input"""
        X = sample_features.reshape(1, -1)
        
        pred1 = trained_model.predict(X)[0]
        pred2 = trained_model.predict(X)[0]
        
        assert pred1 == pred2, "Model should give deterministic output"


@pytest.mark.unit
class TestFeatureScaling:
    """Test feature scaling for predictions"""
    
    def test_scaler_transform(self, scaler, sample_features):
        """Test that scaler can transform features"""
        features = sample_features.reshape(1, -1)
        
        scaled = scaler.transform(features)
        
        assert scaled.shape == features.shape, "Scaled shape should match input"
        assert not np.isnan(scaled).any(), "Scaled features should not contain NaN"
    
    def test_scaled_features_properties(self, scaler, sample_features):
        """Test properties of scaled features"""
        features = np.random.randn(100, 43)
        
        scaled = scaler.transform(features)
        
        # Scaled features should have different statistics than raw
        assert np.abs(scaled.mean()) < 1.0, "Scaled features should have mean near 0"
        assert np.abs(scaled.std() - 1.0) < 0.5, "Scaled features should have std near 1"


@pytest.mark.unit
class TestPredictionValidation:
    """Test prediction input/output validation"""
    
    def test_feature_count_validation(self, trained_model):
        """Test that model rejects wrong number of features"""
        # Wrong feature count
        X_wrong = np.random.randn(1, 10)  # Should be 43
        
        with pytest.raises(ValueError):
            trained_model.predict(X_wrong)
    
    def test_prediction_with_extreme_values(self, trained_model):
        """Test model handles extreme feature values"""
        X_extreme = np.ones((1, 43)) * 1e6
        
        prediction = trained_model.predict(X_extreme)
        
        assert prediction[0] in [0, 1], "Should handle extreme values"
    
    def test_prediction_with_zero_features(self, trained_model):
        """Test model handles zero feature values"""
        X_zeros = np.zeros((1, 43))
        
        prediction = trained_model.predict(X_zeros)
        
        assert prediction[0] in [0, 1], "Should handle zero values"
