#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Docker API endpoints
"""

import requests
import json

API_URL = "http://localhost:8000"

print("=" * 60)
print("   TEST API DOCKER - BTC Prediction")
print("=" * 60)

# Test 1: Root
print("\n[TEST 1] Root endpoint...")
response = requests.get(f"{API_URL}/")
print(f"   Status: {response.status_code}")
data = response.json()
print(f"   Name: {data['name']}")
print(f"   Version: {data['version']}")
print(f"   Model Source: {data['model_source']}")

# Test 2: Health
print("\n[TEST 2] Health check...")
response = requests.get(f"{API_URL}/health")
print(f"   Status: {response.status_code}")
data = response.json()
print(f"   Model Loaded: {data['model_loaded']}")
print(f"   Model Type: {data['model_type']}")
print(f"   Source: {data['model_source']}")

# Test 3: Features
print("\n[TEST 3] Features list...")
response = requests.get(f"{API_URL}/features")
data = response.json()
print(f"   Feature count: {data['feature_count']}")

# Test 4: Prediction
print("\n[TEST 4] Prediction...")
# Example features (43 values)
features = [
    50000, 51000, 49000, 50500, 100, 5000000,  # OHLCV
    0.01, 0.02, 0.03, 0.04, 0.05,  # returns
    0.001, 0.002, 0.003,  # volatility
    50100, 50200, 50300, 50400,  # MA
    50100, 50200, 50300,  # EMA
    10, 20, 30,  # momentum
    0.5, 1.0,  # ROC
    55, 500,  # RSI, ATR
    51000, 50500, 50000, 1000, 0.5,  # BB
    50, 45,  # STOCH
    100, 200, 1.5,  # volume
    1.02, 1.01,  # ratios
    14, 2, 0  # time features
]

response = requests.post(
    f"{API_URL}/predict",
    json={"features": features}
)
print(f"   Status: {response.status_code}")
if response.status_code == 200:
    data = response.json()
    print(f"   Direction: {data['direction']}")
    print(f"   Probability UP: {data['probability_up']:.2%}")
    print(f"   Probability DOWN: {data['probability_down']:.2%}")
    print(f"   Confidence: {data['confidence']:.1f}%")
    print(f"   Signal: {data['signal_strength']}")
else:
    print(f"   Error: {response.json()}")

# Test 5: Error handling
print("\n[TEST 5] Error handling (wrong features)...")
response = requests.post(
    f"{API_URL}/predict",
    json={"features": [1, 2, 3]}
)
print(f"   Status: {response.status_code}")
print(f"   Expected 400, Got: {response.status_code}")

print("\n" + "=" * 60)
print("   DOCKER API TESTS COMPLETED!")
print("=" * 60)
print(f"\n   API URL: {API_URL}")
print(f"   Swagger: {API_URL}/docs")
print(f"   Container: btc-api")
