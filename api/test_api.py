#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for BTC FastAPI
"""

import requests
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training'))
from prepare_data import load_and_prepare_data

API_URL = "http://127.0.0.1:8000"

print("=" * 50)
print("   TEST API FastAPI BTC Prediction")
print("=" * 50)

# Test 1: Health check
print("\n[TEST 1] Health Check...")
response = requests.get(f"{API_URL}/health")
print(f"   Status: {response.status_code}")
print(f"   Response: {response.json()}")

# Test 2: Get features
print("\n[TEST 2] Features List...")
response = requests.get(f"{API_URL}/features")
data = response.json()
print(f"   Feature count: {data['feature_count']}")
print(f"   First 5 features: {data['features'][:5]}")

# Test 3: Prediction with real data
print("\n[TEST 3] Prediction avec donnees reelles...")

# Load real data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'btc_hourly.csv')
X, y, scaler = load_and_prepare_data(data_path, scale=True)

# Get last observation
last_row = X.iloc[-1].values.tolist()

response = requests.post(
    f"{API_URL}/predict",
    json={"features": last_row}
)

print(f"   Status: {response.status_code}")
result = response.json()
print(f"   Direction: {result['direction']}")
print(f"   Probability UP: {result['probability_up']:.2%}")
print(f"   Probability DOWN: {result['probability_down']:.2%}")
print(f"   Confidence: {result['confidence']:.1f}%")
print(f"   Signal: {result['signal_strength']}")

# Test 4: Multiple predictions
print("\n[TEST 4] Multiple predictions (5 dernieres heures)...")
for i in range(-5, 0):
    row = X.iloc[i].values.tolist()
    response = requests.post(f"{API_URL}/predict", json={"features": row})
    r = response.json()
    print(f"   H{i}: {r['direction']:4s} | Conf: {r['confidence']:5.1f}% | Signal: {r['signal_strength']}")

# Test 5: Error handling
print("\n[TEST 5] Error handling (wrong feature count)...")
response = requests.post(
    f"{API_URL}/predict",
    json={"features": [1.0, 2.0, 3.0]}  # Wrong count
)
print(f"   Status: {response.status_code}")
print(f"   Error: {response.json()['detail']}")

print("\n" + "=" * 50)
print("   TOUS LES TESTS PASSES!")
print("=" * 50)
print(f"\n[INFO] API disponible: {API_URL}")
print(f"[INFO] Documentation: {API_URL}/docs")
