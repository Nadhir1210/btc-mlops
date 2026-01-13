import pandas as pd
import numpy as np
import sys
import os
import pickle

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


if __name__ == "__main__":
    # Load and prepare data
    data_path = "data/raw/btc_hourly.csv"
    print(f"Loading data from {data_path}...")
    X, y, scaler = load_and_prepare_data(data_path)
    print(f"Data loaded: X shape = {X.shape}, y shape = {y.shape}")
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Data split: Train size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")
    
    # Create processed data directory
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Created directory: {processed_dir}")
    
    # Save processed data as pickle files
    with open(os.path.join(processed_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    print(f"✓ Saved X_train to {os.path.join(processed_dir, 'X_train.pkl')}")
    
    with open(os.path.join(processed_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)
    print(f"✓ Saved X_test to {os.path.join(processed_dir, 'X_test.pkl')}")
    
    with open(os.path.join(processed_dir, "y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    print(f"✓ Saved y_train to {os.path.join(processed_dir, 'y_train.pkl')}")
    
    with open(os.path.join(processed_dir, "y_test.pkl"), "wb") as f:
        pickle.dump(y_test, f)
    print(f"✓ Saved y_test to {os.path.join(processed_dir, 'y_test.pkl')}")
    
    with open(os.path.join(processed_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler to {os.path.join(processed_dir, 'scaler.pkl')}")
    
    print("\n✓ All data saved successfully!")

