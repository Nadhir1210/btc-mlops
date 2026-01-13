"""
Pytest configuration and shared fixtures for BTC-MLOps tests
"""

import pytest
import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def data_dir(project_root):
    """Get data directory"""
    return project_root / "data"

@pytest.fixture(scope="session")
def raw_data_path(data_dir):
    """Get path to raw BTC data"""
    return data_dir / "raw" / "btc_hourly.csv"

@pytest.fixture(scope="session")
def processed_data_dir(data_dir):
    """Get processed data directory"""
    return data_dir / "processed"

@pytest.fixture(scope="session")
def sample_raw_data():
    """Create sample raw BTC data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    
    data = {
        'DATETIME': dates,
        'UNIX_TIMESTAMP': [int(d.timestamp()) for d in dates],
        'OPEN': np.random.uniform(40000, 50000, 100),
        'HIGH': np.random.uniform(40500, 50500, 100),
        'LOW': np.random.uniform(39500, 49500, 100),
        'CLOSE': np.random.uniform(40000, 50000, 100),
        'VOLUME BTC': np.random.uniform(50, 200, 100),
        'VOLUME USD': np.random.uniform(2000000, 10000000, 100),
    }
    
    df = pd.DataFrame(data)
    # Ensure HIGH > CLOSE > LOW
    df['HIGH'] = df[['OPEN', 'HIGH', 'CLOSE']].max(axis=1) + 100
    df['LOW'] = df[['OPEN', 'CLOSE', 'LOW']].min(axis=1) - 100
    
    return df

@pytest.fixture
def sample_features():
    """Create sample feature data for predictions"""
    np.random.seed(42)
    # 43 features as per prepare_data.py
    return np.random.randn(43)

@pytest.fixture(scope="session")
def trained_model(processed_data_dir):
    """Load trained CatBoost model"""
    try:
        model_path = Path(__file__).parent.parent / "src" / "training" / "catboost_best.pkl"
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        pytest.skip(f"Cannot load trained model: {e}")

@pytest.fixture(scope="session")
def scaler(processed_data_dir):
    """Load feature scaler"""
    try:
        scaler_path = processed_data_dir / "scaler.pkl"
        if not scaler_path.exists():
            scaler_path = Path(__file__).parent.parent / "src" / "training" / "scaler.pkl"
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        pytest.skip(f"Cannot load scaler: {e}")

# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )
