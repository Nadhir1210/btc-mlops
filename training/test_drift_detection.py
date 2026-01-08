"""
Integration test pour la détection de drift
Teste tous les scénarios de drift
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from detect_drift import DataDriftDetector
from simulate_drift import DriftSimulator, generate_drift_scenarios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_drift_detection():
    """Test la détection de drift sur tous les scénarios"""
    
    # Charger données
    try:
        data = pd.read_csv('data_processed.csv')
    except FileNotFoundError:
        logger.error("data_processed.csv non trouvé - utilisation de données synthétiques")
        # Créer données synthétiques pour test
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            col: np.random.normal(0, 1, n_samples)
            for col in [f'feature_{i}' for i in range(20)]
        })
    
    # Générer scénarios
    logger.info("Génération des scénarios de drift...")
    scenarios = generate_drift_scenarios(data)
    
    # Créer détecteur
    detector = DataDriftDetector(data)
    
    # Tester chaque scénario
    results_summary = {}
    
    print("\n" + "="*80)
    print("DRIFT DETECTION TEST RESULTS")
    print("="*80)
    
    for scenario_name, scenario_data in scenarios.items():
        logger.info(f"\nTesting scenario: {scenario_name}")
        
        results = detector.detect_drift(
            scenario_data,
            methods=['ks', 'psi', 'ttest'],
            ks_threshold=0.05,
            psi_threshold=0.1,
            ttest_threshold=0.05
        )
        
        summary = results['summary']
        results_summary[scenario_name] = {
            'total_drifted_features': summary['total_drifted_features'],
            'action': summary['recommended_action'],
            'overall_drift': summary['overall_drift_detected']
        }
        
        print(f"\n{scenario_name.upper()}")
        print("-" * 80)
        print(f"  Overall Drift Detected: {summary['overall_drift_detected']}")
        print(f"  Number of Drifted Features: {summary['total_drifted_features']}")
        print(f"  Recommended Action: {summary['recommended_action']}")
        
        if summary['drifted_features']:
            print(f"  Drifted Features: {', '.join(summary['drifted_features'][:5])}...")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    
    summary_df = pd.DataFrame(results_summary).T
    print(summary_df.to_string())
    
    # Sauvegarder les résultats
    output_path = 'drift_detection_test_results.json'
    import json
    with open(output_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info(f"Résultats sauvegardés: {output_path}")
    
    return results_summary


if __name__ == '__main__':
    results = test_drift_detection()
    
    # Vérifier les résultats
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    
    # Le baseline ne doit pas avoir de drift
    assert not results['baseline']['overall_drift'], "Baseline should NOT have drift!"
    print("✓ Baseline: PASS")
    
    # Les autres scénarios doivent avoir du drift
    for scenario in ['mean_shift', 'variance_shift', 'outlier_injection', 
                     'concept_drift', 'gradual_drift', 'seasonal_shift']:
        if results[scenario]['overall_drift']:
            print(f"✓ {scenario.upper()}: PASS (drift detected)")
        else:
            print(f"⚠ {scenario.upper()}: WARNING (no drift detected)")
    
    print("="*80)
