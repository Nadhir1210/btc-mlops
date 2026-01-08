"""
Data Drift Simulation
Simule différents types de drift pour tester la détection
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftSimulator:
    """Simule différents types de data drift"""
    
    def __init__(self, base_data):
        """
        Args:
            base_data: DataFrame de base
        """
        self.base_data = base_data.copy()
        self.original_means = base_data.mean()
        self.original_stds = base_data.std()
    
    def mean_shift(self, feature_list, shift_amount=1.0, sample_size=None):
        """
        Simule un changement de moyenne (shift graduel)
        Scenario: Les prix commencent à monter/descendre systématiquement
        """
        data = self.base_data.sample(frac=sample_size or 0.5, random_state=42).copy()
        
        for feature in feature_list:
            if feature in data.columns:
                data[feature] = data[feature] + (shift_amount * self.original_stds[feature])
                logger.info(f"Mean shift simulé pour {feature}: +{shift_amount} σ")
        
        return data
    
    def variance_shift(self, feature_list, variance_multiplier=2.0, sample_size=None):
        """
        Simule une augmentation de variance (volatilité)
        Scenario: Le marché devient plus volatil
        """
        data = self.base_data.sample(frac=sample_size or 0.5, random_state=42).copy()
        
        for feature in feature_list:
            if feature in data.columns:
                mean = data[feature].mean()
                data[feature] = mean + (data[feature] - mean) * np.sqrt(variance_multiplier)
                logger.info(f"Variance shift simulée pour {feature}: x{variance_multiplier}")
        
        return data
    
    def outlier_injection(self, feature_list, outlier_percentage=0.05, 
                         outlier_magnitude=3.0, sample_size=None):
        """
        Injecte des outliers (régime extrême)
        Scenario: Événements extrêmes du marché (crash, pump)
        """
        data = self.base_data.sample(frac=sample_size or 0.5, random_state=42).copy()
        
        for feature in feature_list:
            if feature in data.columns:
                num_outliers = int(len(data) * outlier_percentage)
                outlier_indices = np.random.choice(len(data), num_outliers, replace=False)
                
                for idx in outlier_indices:
                    data.iloc[idx, data.columns.get_loc(feature)] += (
                        outlier_magnitude * self.original_stds[feature]
                    )
                
                logger.info(f"Outliers injectés dans {feature}: {num_outliers} ({outlier_percentage*100}%)")
        
        return data
    
    def covariate_shift(self, feature_list, feature_to_shift=None, 
                       shift_amount=1.0, sample_size=None):
        """
        Simule un changement de covariables (feature interchanges)
        Scenario: Les corrélations entre features changent
        """
        data = self.base_data.sample(frac=sample_size or 0.5, random_state=42).copy()
        
        if feature_to_shift and feature_to_shift in data.columns:
            data[feature_to_shift] = data[feature_to_shift] * (1 + shift_amount * 0.1)
            logger.info(f"Covariate shift simulé pour {feature_to_shift}")
        
        return data
    
    def concept_drift(self, feature_list, sample_size=None):
        """
        Simule un concept drift (changement de la relation features-target)
        Scenario: La relation entre les features et la direction du prix change
        """
        data = self.base_data.sample(frac=sample_size or 0.5, random_state=42).copy()
        
        # Appliquer une transformation non-linéaire
        for feature in feature_list:
            if feature in data.columns:
                data[feature] = np.sin(data[feature] / self.original_stds[feature])
        
        logger.info(f"Concept drift simulé: transformation non-linéaire appliquée")
        
        return data
    
    def gradual_drift(self, feature_list, num_batches=5, 
                     final_shift=1.5, sample_size=None):
        """
        Simule un drift graduel (dérive progressive)
        Scenario: Changement lent mais continu des conditions de marché
        """
        all_data = []
        
        for batch in range(num_batches):
            shift_factor = (final_shift / num_batches) * (batch + 1)
            
            batch_data = self.base_data.sample(
                frac=sample_size or 0.1,
                random_state=42 + batch
            ).copy()
            
            for feature in feature_list:
                if feature in batch_data.columns:
                    batch_data[feature] = (
                        batch_data[feature] + 
                        (shift_factor * self.original_stds[feature])
                    )
            
            all_data.append(batch_data)
        
        logger.info(f"Gradual drift simulé sur {num_batches} batches")
        
        return pd.concat(all_data, ignore_index=True)
    
    def seasonal_shift(self, feature_list, pattern='sine', sample_size=None):
        """
        Simule un décalage saisonnier
        Scenario: Les patterns de marché changent selon la saison
        """
        data = self.base_data.sample(frac=sample_size or 0.5, random_state=42).copy()
        
        # Ajouter une composante saisonnière
        seasonal_factor = np.sin(np.arange(len(data)) * 2 * np.pi / 100)
        
        for feature in feature_list:
            if feature in data.columns:
                data[feature] = (
                    data[feature] + 
                    seasonal_factor * self.original_stds[feature]
                )
        
        logger.info(f"Seasonal shift simulé avec pattern: {pattern}")
        
        return data
    
    def no_drift(self, sample_size=None):
        """Pas de drift - baseline pour comparaison"""
        data = self.base_data.sample(frac=sample_size or 0.5, random_state=42).copy()
        logger.info("Baseline: pas de drift")
        return data


def generate_drift_scenarios(base_data, output_dir='training/drift_scenarios'):
    """
    Génère différents scénarios de drift pour test
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    simulator = DriftSimulator(base_data)
    
    # Tous les features numériques sauf la cible
    feature_list = [col for col in base_data.columns if col != 'target']
    
    scenarios = {
        'baseline': simulator.no_drift(),
        'mean_shift': simulator.mean_shift(feature_list, shift_amount=1.5),
        'variance_shift': simulator.variance_shift(feature_list, variance_multiplier=2.5),
        'outlier_injection': simulator.outlier_injection(feature_list, outlier_percentage=0.1),
        'concept_drift': simulator.concept_drift(feature_list),
        'gradual_drift': simulator.gradual_drift(feature_list, num_batches=3, final_shift=2.0),
        'seasonal_shift': simulator.seasonal_shift(feature_list),
    }
    
    # Sauvegarder les scénarios
    for scenario_name, scenario_data in scenarios.items():
        filepath = f"{output_dir}/{scenario_name}.csv"
        scenario_data.to_csv(filepath, index=False)
        logger.info(f"Scénario sauvegardé: {filepath}")
    
    return scenarios


def main():
    """Exemple d'utilisation"""
    # Charger les données
    try:
        data = pd.read_csv('training/data_processed.csv')
    except FileNotFoundError:
        logger.error("Fichier data_processed.csv non trouvé")
        return
    
    # Générer les scénarios
    scenarios = generate_drift_scenarios(data)
    
    print("\n" + "="*70)
    print("DRIFT SCENARIOS GENERATED")
    print("="*70)
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{scenario_name}:")
        print(f"  Shape: {scenario_data.shape}")
        print(f"  Mean change: {(scenario_data.mean() - data.mean()).mean():.4f}")
        print(f"  Std change: {(scenario_data.std() - data.std()).mean():.4f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
