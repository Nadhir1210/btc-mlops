"""
Data Drift Detection Module
Détecte les changements de distribution dans les données de production
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Détecteur de Data Drift utilisant plusieurs méthodes statistiques"""
    
    def __init__(self, reference_data, feature_list=None):
        """
        Args:
            reference_data: DataFrame de référence (données d'entraînement)
            feature_list: Liste des features à monitorer (None = toutes)
        """
        self.reference_data = reference_data
        self.feature_list = feature_list or reference_data.columns.tolist()
        self.reference_stats = self._compute_stats(reference_data)
        self.drift_results = {}
        
    def _compute_stats(self, data):
        """Calcule les statistiques pour chaque feature"""
        stats_dict = {}
        for col in self.feature_list:
            if col in data.columns:
                col_data = data[col].dropna()
                stats_dict[col] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median(),
                    'q1': col_data.quantile(0.25),
                    'q3': col_data.quantile(0.75),
                }
        return stats_dict
    
    def kolmogorov_smirnov_test(self, current_data, threshold=0.05):
        """
        Teste KS : détecte si les distributions ont changé
        Plus sensible pour les queues de distribution
        """
        results = {}
        for col in self.feature_list:
            if col not in current_data.columns:
                continue
                
            current_col = current_data[col].dropna()
            reference_col = self.reference_data[col].dropna()
            
            # Normaliser les données
            reference_normalized = (reference_col - reference_col.mean()) / reference_col.std()
            current_normalized = (current_col - current_col.mean()) / current_col.std()
            
            # KS test
            statistic, p_value = stats.ks_2samp(reference_normalized, current_normalized)
            
            results[col] = {
                'method': 'Kolmogorov-Smirnov',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_drift': p_value < threshold,
                'threshold': threshold,
            }
        
        return results
    
    def population_stability_index(self, current_data, threshold=0.1):
        """
        PSI : mesure les changements de distribution
        PSI > 0.1 = drift modéré, PSI > 0.25 = drift significatif
        """
        results = {}
        for col in self.feature_list:
            if col not in current_data.columns:
                continue
            
            current_col = current_data[col].dropna()
            reference_col = self.reference_data[col].dropna()
            
            # Créer des bins
            bins = np.histogram_bin_edges(
                reference_col,
                bins=10,
                range=(min(reference_col.min(), current_col.min()),
                       max(reference_col.max(), current_col.max()))
            )
            
            # Calculer les proportions
            ref_counts = np.histogram(reference_col, bins=bins)[0]
            curr_counts = np.histogram(current_col, bins=bins)[0]
            
            # Éviter division par zéro
            ref_pct = ref_counts / np.sum(ref_counts)
            curr_pct = curr_counts / np.sum(curr_counts)
            
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
            
            # Calculer PSI
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            
            results[col] = {
                'method': 'Population Stability Index',
                'psi': float(psi),
                'is_drift': psi > threshold,
                'threshold': threshold,
                'interpretation': self._interpret_psi(psi)
            }
        
        return results
    
    def _interpret_psi(self, psi):
        """Interprète la valeur PSI"""
        if psi < 0.05:
            return 'Pas de drift'
        elif psi < 0.1:
            return 'Drift faible'
        elif psi < 0.25:
            return 'Drift modéré'
        else:
            return 'Drift significatif'
    
    def statistical_test(self, current_data, threshold=0.05):
        """
        Tests statistiques simples (Welch t-test pour variables continues)
        """
        results = {}
        for col in self.feature_list:
            if col not in current_data.columns:
                continue
            
            current_col = current_data[col].dropna()
            reference_col = self.reference_data[col].dropna()
            
            # Welch t-test (pas d'assomption d'égalité des variances)
            statistic, p_value = stats.ttest_ind(reference_col, current_col, equal_var=False)
            
            results[col] = {
                'method': 'Welch t-test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'is_drift': p_value < threshold,
                'threshold': threshold,
            }
        
        return results
    
    def detect_drift(self, current_data, methods=['ks', 'psi', 'ttest'], 
                    ks_threshold=0.05, psi_threshold=0.1, ttest_threshold=0.05):
        """
        Détecte le drift en utilisant plusieurs méthodes
        
        Args:
            current_data: DataFrame des données actuelles
            methods: Liste des méthodes à utiliser
            ks_threshold: Seuil pour KS test
            psi_threshold: Seuil pour PSI
            ttest_threshold: Seuil pour t-test
        """
        logger.info(f"Détection de drift avec méthodes: {methods}")
        
        self.drift_results = {
            'timestamp': datetime.now().isoformat(),
            'methods': {},
            'summary': {}
        }
        
        if 'ks' in methods:
            self.drift_results['methods']['kolmogorov_smirnov'] = self.kolmogorov_smirnov_test(
                current_data, ks_threshold
            )
        
        if 'psi' in methods:
            self.drift_results['methods']['psi'] = self.population_stability_index(
                current_data, psi_threshold
            )
        
        if 'ttest' in methods:
            self.drift_results['methods']['statistical_test'] = self.statistical_test(
                current_data, ttest_threshold
            )
        
        # Résumé du drift
        self._compute_summary()
        
        return self.drift_results
    
    def _compute_summary(self):
        """Calcule un résumé global du drift"""
        all_drifted_features = set()
        feature_drift_count = {}
        
        for method, results in self.drift_results['methods'].items():
            for feature, result in results.items():
                if result.get('is_drift', False):
                    all_drifted_features.add(feature)
                    feature_drift_count[feature] = feature_drift_count.get(feature, 0) + 1
        
        self.drift_results['summary'] = {
            'total_drifted_features': len(all_drifted_features),
            'drifted_features': list(all_drifted_features),
            'feature_drift_consensus': {
                k: v for k, v in feature_drift_count.items() if v > 1
            },
            'overall_drift_detected': len(all_drifted_features) > 0,
            'recommended_action': self._recommend_action(all_drifted_features)
        }
    
    def _recommend_action(self, drifted_features):
        """Recommande une action basée sur le drift détecté"""
        if not drifted_features:
            return 'No action needed'
        elif len(drifted_features) <= 3:
            return 'Monitor closely'
        else:
            return 'RETRAIN RECOMMENDED'
    
    def save_results(self, filepath):
        """Sauvegarde les résultats de détection"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.drift_results, f, indent=2)
        
        logger.info(f"Résultats sauvegardés: {filepath}")
    
    def print_report(self):
        """Affiche un rapport lisible du drift détecté"""
        print("\n" + "="*70)
        print("DATA DRIFT DETECTION REPORT")
        print("="*70)
        print(f"Timestamp: {self.drift_results['timestamp']}\n")
        
        for method, results in self.drift_results['methods'].items():
            print(f"\n{method.upper()}")
            print("-" * 50)
            drifted = [f for f, r in results.items() if r.get('is_drift')]
            if drifted:
                print(f"  ⚠️  Drift détecté dans: {', '.join(drifted)}")
                for feature in drifted:
                    print(f"     {feature}: {results[feature]}")
            else:
                print("  ✓ Pas de drift détecté")
        
        print(f"\n{'SUMMARY':^70}")
        print("-" * 50)
        summary = self.drift_results['summary']
        print(f"Total features with drift: {summary['total_drifted_features']}")
        print(f"Action: {summary['recommended_action']}")
        print("="*70 + "\n")


def main():
    """Exemple d'utilisation"""
    # Charger les données de référence
    reference_data = pd.read_csv('training/data_processed.csv')
    
    # Simuler les données actuelles (avec drift)
    current_data = reference_data.sample(frac=0.5, random_state=42).copy()
    current_data.iloc[:int(len(current_data)*0.3), 0] += np.random.normal(0, 5, int(len(current_data)*0.3))
    
    # Créer le détecteur
    detector = DataDriftDetector(reference_data)
    
    # Détecter le drift
    results = detector.detect_drift(current_data, methods=['ks', 'psi', 'ttest'])
    
    # Afficher le rapport
    detector.print_report()
    
    # Sauvegarder les résultats
    detector.save_results('training/drift_detection_results.json')


if __name__ == '__main__':
    main()
