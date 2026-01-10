import pandas as pd
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Tuple
from sklearn.preprocessing import StandardScaler
from src.features.indicators import add_indicators

def load_and_prepare_data(path: str, scale: bool = True) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Charge et prépare les données BTC pour l'entraînement.
    
    Args:
        path: Chemin vers le fichier CSV
        scale: Normaliser les features
        
    Returns:
        X, y, scaler: Features, target, et scaler
    """
    df = pd.read_csv(path)

    # 1️⃣ Trier par date
    df["DATETIME"] = pd.to_datetime(df["DATETIME"])
    df = df.sort_values("DATETIME").reset_index(drop=True)

    # 2️⃣ Target : direction du prix (à la prochaine heure)
    df["target"] = (df["CLOSE"].shift(-1) > df["CLOSE"]).astype(int)

    # 3️⃣ Feature engineering avancé
    df = add_indicators(df)

    # 4️⃣ Supprimer les lignes avec NaN
    df = df.dropna()

    # 5️⃣ Features sélectionnées
    feature_cols = [col for col in df.columns if col not in ["DATETIME", "target", "UNIX_TIMESTAMP"]]
    
    X = df[feature_cols]
    y = df["target"]

    # 6️⃣ Normaliser si demandé
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=feature_cols,
            index=X.index
        )

    return X, y, scaler

