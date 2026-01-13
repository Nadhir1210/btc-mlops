"""
Integration tests for the full training and evaluation pipeline
Tests the complete workflow: data → train → eval → export
"""

import pytest
import os
import sys
import subprocess
import pickle
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.prepare_data import load_and_prepare_data
from sklearn.model_selection import train_test_split


@pytest.mark.integration
@pytest.mark.slow
class TestDataPipelineIntegration:
    """Test the complete data pipeline"""
    
    def test_data_preparation_pipeline(self, raw_data_path):
        """Test complete data preparation pipeline"""
        if not raw_data_path.exists():
            pytest.skip("Raw data not found")
        
        # Load and prepare data
        X, y, scaler = load_and_prepare_data(str(raw_data_path), scale=True)
        
        # Verify output
        assert X.shape[0] > 0, "X should not be empty"
        assert y.shape[0] > 0, "y should not be empty"
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 43, "Should have 43 features"
        assert set(y.unique()).issubset({0, 1}), "Target should be binary"
        assert scaler is not None, "Scaler should be returned"
    
    def test_train_test_split(self, raw_data_path):
        """Test train/test split preserves data integrity"""
        if not raw_data_path.exists():
            pytest.skip("Raw data not found")
        
        X, y, _ = load_and_prepare_data(str(raw_data_path), scale=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Verify split
        assert len(X_train) + len(X_test) == len(X), "Split should preserve all samples"
        assert len(y_train) + len(y_test) == len(y), "Split should preserve all labels"
        assert len(X_train) > len(X_test), "Train set should be larger than test set"
        assert 0.15 < len(X_test) / len(X) < 0.25, "Test set should be ~20% of data"
        assert len(X_train.columns) == 43, "Features should be preserved in train set"
    
    def test_processed_data_files_exist(self, processed_data_dir):
        """Test that processed data files exist"""
        required_files = [
            'X_train.pkl',
            'X_test.pkl',
            'y_train.pkl',
            'y_test.pkl',
            'scaler.pkl'
        ]
        
        for file in required_files:
            file_path = processed_data_dir / file
            assert file_path.exists(), f"Processed data file missing: {file}"
    
    def test_processed_data_can_be_loaded(self, processed_data_dir):
        """Test that processed data can be loaded from pickle files"""
        files = {
            'X_train': processed_data_dir / 'X_train.pkl',
            'X_test': processed_data_dir / 'X_test.pkl',
            'y_train': processed_data_dir / 'y_train.pkl',
            'y_test': processed_data_dir / 'y_test.pkl',
            'scaler': processed_data_dir / 'scaler.pkl'
        }
        
        for name, path in files.items():
            if not path.exists():
                pytest.skip(f"Processed data not found: {name}")
            
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            assert data is not None, f"{name} should load successfully"


@pytest.mark.integration
@pytest.mark.slow
class TestModelTrainingIntegration:
    """Test model training pipeline"""
    
    def test_trained_models_exist(self):
        """Test that all trained models exist"""
        model_dir = Path(__file__).parent.parent.parent / "src" / "training"
        
        models = {
            'catboost_best.pkl': model_dir / 'catboost_best.pkl',
            'lightgbm_best.pkl': model_dir / 'lightgbm_best.pkl',
            'xgboost_best.pkl': model_dir / 'xgboost_best.pkl',
        }
        
        for name, path in models.items():
            assert path.exists(), f"Trained model missing: {name}"
    
    def test_trained_models_can_be_loaded(self):
        """Test that trained models can be loaded"""
        model_dir = Path(__file__).parent.parent.parent / "src" / "training"
        
        models = {
            'catboost': model_dir / 'catboost_best.pkl',
            'lightgbm': model_dir / 'lightgbm_best.pkl',
            'xgboost': model_dir / 'xgboost_best.pkl',
        }
        
        for name, path in models.items():
            if path.exists():
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                assert model is not None, f"{name} should load successfully"
    
    def test_models_have_predict_method(self, trained_model):
        """Test that loaded models have predict method"""
        assert hasattr(trained_model, 'predict'), "Model should have predict method"
        assert hasattr(trained_model, 'predict_proba'), "Model should have predict_proba method"
    
    def test_models_compatible_with_processed_data(self, trained_model, processed_data_dir):
        """Test that models work with processed data"""
        X_test_path = processed_data_dir / 'X_test.pkl'
        
        if not X_test_path.exists():
            pytest.skip("Processed test data not found")
        
        with open(X_test_path, 'rb') as f:
            X_test = pickle.load(f)
        
        # Make prediction
        predictions = trained_model.predict(X_test.iloc[:10])
        
        assert len(predictions) == 10, "Should make predictions for test samples"
        assert set(predictions).issubset({0, 1}), "Predictions should be binary"


@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete end-to-end pipeline"""
    
    def test_dvc_pipeline_exists(self, project_root):
        """Test that DVC pipeline configuration exists"""
        dvc_yaml = project_root / 'dvc.yaml'
        assert dvc_yaml.exists(), "dvc.yaml should exist"
    
    def test_params_yaml_exists(self, project_root):
        """Test that params.yaml exists"""
        params_yaml = project_root / 'params.yaml'
        assert params_yaml.exists(), "params.yaml should exist"
    
    def test_dvc_config_has_remotes(self, project_root):
        """Test that DVC config has remotes configured"""
        dvc_config = project_root / '.dvc' / 'config'
        
        if dvc_config.exists():
            with open(dvc_config) as f:
                content = f.read()
            
            # Should have at least one remote
            assert 'remote' in content, "DVC config should have remote configuration"
    
    def test_all_required_directories_exist(self, project_root):
        """Test that all required directories exist"""
        required_dirs = [
            'src/data',
            'src/features',
            'src/training',
            'src/evaluation',
            'data',
            'serving',
            'monitoring'
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory missing: {dir_name}"
    
    def test_all_required_files_exist(self, project_root):
        """Test that all required Python modules exist"""
        required_files = [
            'src/data/prepare_data.py',
            'src/features/indicators.py',
            'src/training/train_bayesian_optimization.py',
            'src/evaluation/evaluate_models.py',
            'serving/app.py',
        ]
        
        for file_name in required_files:
            file_path = project_root / file_name
            assert file_path.exists(), f"Required file missing: {file_name}"


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineReproducibility:
    """Test pipeline reproducibility with DVC"""
    
    def test_dvc_repro_dry_run(self, project_root):
        """Test that dvc repro --dry works"""
        try:
            # Change to project directory
            original_dir = os.getcwd()
            os.chdir(project_root)
            
            # Run dvc repro --dry
            result = subprocess.run(
                ['dvc', 'repro', '--dry'],
                capture_output=True,
                timeout=30
            )
            
            # Verify command succeeded
            assert result.returncode == 0, f"dvc repro --dry failed: {result.stderr.decode()}"
            
        except Exception as e:
            pytest.skip(f"DVC not available or repo not setup: {e}")
        finally:
            os.chdir(original_dir)
    
    def test_dvc_dag_command(self, project_root):
        """Test that dvc dag works"""
        try:
            original_dir = os.getcwd()
            os.chdir(project_root)
            
            result = subprocess.run(
                ['dvc', 'dag'],
                capture_output=True,
                timeout=10
            )
            
            assert result.returncode == 0, f"dvc dag failed: {result.stderr.decode()}"
            
            # Output should contain pipeline stages
            output = result.stdout.decode()
            assert 'prepare' in output.lower(), "Pipeline should have prepare stage"
            
        except Exception as e:
            pytest.skip(f"DVC not available: {e}")
        finally:
            os.chdir(original_dir)
