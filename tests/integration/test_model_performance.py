"""
Model performance regression tests
Tests that model metrics meet minimum requirements
"""

import pytest
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)


@pytest.mark.integration
class TestModelPerformance:
    """Test trained model performance metrics"""
    
    def test_model_accuracy_minimum(self, trained_model, processed_data_dir):
        """Test that model achieves minimum accuracy"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        y_test_path = processed_data_dir / 'y_test.pkl'
        
        if not X_test_path.exists() or not y_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        
        # Make predictions
        y_pred = trained_model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # CatBoost should achieve at least 50% accuracy (better than random)
        assert accuracy >= 0.50, f"Accuracy {accuracy:.4f} is below 50% threshold"
        
        # Report actual performance
        print(f"\nModel Accuracy: {accuracy:.4f}")
    
    def test_model_f1_score_minimum(self, trained_model, processed_data_dir):
        """Test that model achieves minimum F1 score"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        y_test_path = processed_data_dir / 'y_test.pkl'
        
        if not X_test_path.exists() or not y_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        
        # Make predictions
        y_pred = trained_model.predict(X_test)
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        
        # Should achieve at least 0.30 F1 (has seen better than 0.2514 baseline)
        assert f1 >= 0.30, f"F1 score {f1:.4f} is below 0.30 threshold"
        
        print(f"\nModel F1 Score: {f1:.4f}")
    
    def test_model_roc_auc_minimum(self, trained_model, processed_data_dir):
        """Test that model achieves minimum ROC-AUC"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        y_test_path = processed_data_dir / 'y_test.pkl'
        
        if not X_test_path.exists() or not y_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        
        # Get probability predictions
        y_proba = trained_model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC-AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Should achieve at least 0.51 (better than random 0.50)
        assert roc_auc >= 0.51, f"ROC-AUC {roc_auc:.4f} is below 0.51 threshold"
        
        print(f"\nModel ROC-AUC: {roc_auc:.4f}")
    
    def test_model_recall_minimum(self, trained_model, processed_data_dir):
        """Test that model achieves minimum recall"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        y_test_path = processed_data_dir / 'y_test.pkl'
        
        if not X_test_path.exists() or not y_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        
        # Make predictions
        y_pred = trained_model.predict(X_test)
        
        # Calculate recall
        recall = recall_score(y_test, y_pred)
        
        # Should detect at least 25% of positive cases
        assert recall >= 0.25, f"Recall {recall:.4f} is below 0.25 threshold"
        
        print(f"\nModel Recall: {recall:.4f}")
    
    def test_model_precision_not_worse_than_baseline(self, trained_model, processed_data_dir):
        """Test that model precision is reasonable"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        y_test_path = processed_data_dir / 'y_test.pkl'
        
        if not X_test_path.exists() or not y_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        
        # Make predictions
        y_pred = trained_model.predict(X_test)
        
        # Calculate precision
        precision = precision_score(y_test, y_pred)
        
        # Should be above random (0.5) and have reasonable signal
        assert precision >= 0.40, f"Precision {precision:.4f} is suspiciously low"
        
        print(f"\nModel Precision: {precision:.4f}")


@pytest.mark.integration
class TestModelConsistency:
    """Test model consistency across different test samples"""
    
    def test_model_predictions_consistent_class_balance(self, trained_model, processed_data_dir):
        """Test that model predictions have reasonable class balance"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        
        if not X_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        
        # Make predictions
        y_pred = trained_model.predict(X_test)
        
        # Calculate class distribution
        positive_rate = np.mean(y_pred)
        
        # Should not be all zeros or all ones
        assert 0.1 < positive_rate < 0.9, \
            f"Model predicts all same class (positive_rate={positive_rate:.2%})"
        
        print(f"\nModel positive class rate: {positive_rate:.2%}")
    
    def test_model_probability_distribution(self, trained_model, processed_data_dir):
        """Test that model probability predictions are well-distributed"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        
        if not X_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        
        # Get probability predictions
        y_proba = trained_model.predict_proba(X_test)[:, 1]
        
        # Check distribution
        mean_proba = np.mean(y_proba)
        std_proba = np.std(y_proba)
        
        # Probabilities should have reasonable variance
        assert std_proba > 0.05, "Probability distribution is too narrow"
        
        # Mean should be close to class balance (near 0.5 for balanced data)
        assert 0.3 < mean_proba < 0.7, f"Mean probability {mean_proba:.4f} is extreme"
        
        print(f"\nProbability mean: {mean_proba:.4f}, std: {std_proba:.4f}")


@pytest.mark.integration
class TestModelRegressionOnBaseline:
    """Test that model doesn't regress from baseline"""
    
    def test_f1_better_than_baseline(self, trained_model, processed_data_dir):
        """Test that F1 score is better than v1 baseline (0.2514)"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        y_test_path = processed_data_dir / 'y_test.pkl'
        
        if not X_test_path.exists() or not y_test_path.exists():
            pytest.skip("Test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        with open(y_test_path, 'rb') as f:
            y_test = pickle.load(f)
        
        # Make predictions
        y_pred = trained_model.predict(X_test)
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        
        # Should be significantly better than v1 baseline
        baseline_f1 = 0.2514
        assert f1 > baseline_f1 * 1.1, \
            f"F1 {f1:.4f} should be at least 10% better than baseline {baseline_f1:.4f}"
        
        improvement = (f1 - baseline_f1) / baseline_f1 * 100
        print(f"\nF1 improvement over baseline: +{improvement:.1f}%")
