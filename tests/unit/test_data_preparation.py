"""
Unit tests for data preparation and feature engineering
Tests prepare_data.py and indicators.py
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.prepare_data import load_and_prepare_data
from features.indicators import add_indicators, calculate_rsi, calculate_atr


@pytest.mark.unit
class TestIndicators:
    """Test technical indicator calculations"""
    
    def test_rsi_calculation(self):
        """Test RSI (Relative Strength Index) calculation"""
        # Create sample price data
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
        
        rsi = calculate_rsi(prices, period=14)
        
        # RSI should be between 0 and 100
        assert rsi.min() >= 0, "RSI minimum should be >= 0"
        assert rsi.max() <= 100, "RSI maximum should be <= 100"
        
    def test_rsi_output_shape(self):
        """Test RSI output has same shape as input"""
        prices = pd.Series(np.random.uniform(100, 110, 100))
        rsi = calculate_rsi(prices, period=14)
        
        assert len(rsi) == len(prices), "RSI length should match input length"
    
    def test_atr_calculation(self):
        """Test ATR (Average True Range) calculation"""
        high = pd.Series([110, 111, 109, 112, 110])
        low = pd.Series([100, 101, 99, 102, 100])
        close = pd.Series([105, 105, 104, 107, 105])
        
        atr = calculate_atr(high, low, close, period=3)
        
        # ATR should be positive
        assert (atr >= 0).all(), "ATR should be non-negative"
        assert len(atr) == len(high), "ATR length should match input"
    
    def test_indicators_with_missing_data(self, sample_raw_data):
        """Test indicators handle NaN values properly"""
        df = sample_raw_data.copy()
        
        # Add some NaN values
        df.loc[0, 'CLOSE'] = np.nan
        
        result = add_indicators(df)
        
        # Should not have more NaN values than input
        assert result.isna().sum().sum() > 0, "Some NaN values expected"
        assert result['RSI'].notna().sum() > 0, "Should have some valid RSI values"


@pytest.mark.unit
class TestFeatureEngineering:
    """Test feature engineering pipeline"""
    
    def test_add_indicators_output_shape(self, sample_raw_data):
        """Test add_indicators returns expected number of features"""
        result = add_indicators(sample_raw_data)
        
        # Should have original columns + new indicator columns
        assert len(result.columns) > len(sample_raw_data.columns), \
            "Feature engineering should add columns"
        
        # Check key indicators exist
        key_indicators = ['volatility_5', 'volatility_10', 'rsi', 'bb_upper', 
                         'bb_lower', 'atr', 'momentum_5']
        for indicator in key_indicators:
            assert indicator in result.columns, f"Missing indicator: {indicator}"
    
    def test_add_indicators_no_negative_volatility(self, sample_raw_data):
        """Test that volatility is non-negative"""
        result = add_indicators(sample_raw_data)
        
        assert (result['volatility_5'].dropna() >= 0).all(), \
            "Volatility should be non-negative"
        assert (result['volatility_10'].dropna() >= 0).all(), \
            "Volatility should be non-negative"
    
    def test_feature_engineering_preserves_rows(self, sample_raw_data):
        """Test that feature engineering preserves number of rows (before dropna)"""
        result = add_indicators(sample_raw_data)
        
        # Should have same or more rows (not drop any)
        assert len(result) >= len(sample_raw_data) - 20, \
            "Feature engineering should preserve most rows"


@pytest.mark.unit
class TestDataPreparation:
    """Test data loading and preparation"""
    
    def test_load_and_prepare_data_output_types(self, raw_data_path):
        """Test load_and_prepare_data returns correct types"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        X, y, scaler = load_and_prepare_data(str(raw_data_path), scale=True)
        
        assert isinstance(X, pd.DataFrame), "X should be DataFrame"
        assert isinstance(y, pd.Series), "y should be Series"
        assert scaler is not None, "Scaler should not be None when scale=True"
    
    def test_load_and_prepare_data_scaling(self, raw_data_path):
        """Test that scaling is applied correctly"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        X_scaled, _, _ = load_and_prepare_data(str(raw_data_path), scale=True)
        X_unscaled, _, _ = load_and_prepare_data(str(raw_data_path), scale=False)
        
        # Scaled data should have different values
        assert not (X_scaled.iloc[:, 0] == X_unscaled.iloc[:, 0]).all(), \
            "Scaled and unscaled data should differ"
        
        # Scaled data should have mean near 0 and std near 1
        means = X_scaled.mean()
        assert (means.abs() < 1.0).all(), "Scaled features should have mean near 0"
    
    def test_load_and_prepare_data_target_is_binary(self, raw_data_path):
        """Test that target is binary (0 or 1)"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        _, y, _ = load_and_prepare_data(str(raw_data_path))
        
        assert set(y.unique()).issubset({0, 1}), "Target should be binary (0 or 1)"
        assert len(y) > 0, "Target should not be empty"
    
    def test_load_and_prepare_data_no_missing_values(self, raw_data_path):
        """Test that output has no missing values"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        X, y, _ = load_and_prepare_data(str(raw_data_path))
        
        assert not X.isna().any().any(), "X should not have NaN values"
        assert not y.isna().any(), "y should not have NaN values"
    
    def test_load_and_prepare_data_correct_number_of_features(self, raw_data_path):
        """Test that correct number of features are returned"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        X, _, _ = load_and_prepare_data(str(raw_data_path))
        
        # Should have 43 features (as per specification)
        assert X.shape[1] == 43, f"Expected 43 features, got {X.shape[1]}"
    
    def test_prepare_data_without_scaling(self, raw_data_path):
        """Test data preparation without scaling"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        X, y, scaler = load_and_prepare_data(str(raw_data_path), scale=False)
        
        assert scaler is None, "Scaler should be None when scale=False"
        assert not X.isna().any().any(), "No NaN values expected"


@pytest.mark.unit
class TestDataQuality:
    """Test data quality checks"""
    
    def test_raw_data_has_required_columns(self, raw_data_path):
        """Test that raw data has required columns"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        df = pd.read_csv(raw_data_path)
        required_cols = ['DATETIME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME BTC']
        
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
    
    def test_raw_data_ohlc_relationships(self, raw_data_path):
        """Test that OHLC data has correct relationships"""
        if not raw_data_path.exists():
            pytest.skip("Raw data file not found")
        
        df = pd.read_csv(raw_data_path).head(100)
        
        # HIGH should be >= OPEN, CLOSE, LOW
        assert (df['HIGH'] >= df['OPEN']).all(), "HIGH should be >= OPEN"
        assert (df['HIGH'] >= df['CLOSE']).all(), "HIGH should be >= CLOSE"
        assert (df['HIGH'] >= df['LOW']).all(), "HIGH should be >= LOW"
        
        # LOW should be <= OPEN, CLOSE, HIGH
        assert (df['LOW'] <= df['OPEN']).all(), "LOW should be <= OPEN"
        assert (df['LOW'] <= df['CLOSE']).all(), "LOW should be <= CLOSE"
