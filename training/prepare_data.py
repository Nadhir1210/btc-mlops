import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler

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
    df = _create_advanced_features(df)

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


def _create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features avancées pour le trading."""
    
    # 1️⃣ Features d'ordre 1 (déjà présentes)
    df = df.copy()
    
    # 2️⃣ Volatilité
    df["volatility_5"] = df["CLOSE"].rolling(5).std()
    df["volatility_10"] = df["CLOSE"].rolling(10).std()
    
    # 3️⃣ Returns et log returns
    df["returns"] = df["CLOSE"].pct_change()
    df["log_returns"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))
    
    # 4️⃣ RSI (Relative Strength Index)
    df["rsi"] = _calculate_rsi(df["CLOSE"], period=14)
    
    # 5️⃣ Bollinger Bands
    sma20 = df["CLOSE"].rolling(20).mean()
    std20 = df["CLOSE"].rolling(20).std()
    df["bb_upper"] = sma20 + (std20 * 2)
    df["bb_lower"] = sma20 - (std20 * 2)
    df["bb_position"] = (df["CLOSE"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    
    # 6️⃣ ATR (Average True Range)
    df["atr"] = _calculate_atr(df["HIGH"], df["LOW"], df["CLOSE"], period=14)
    
    # 7️⃣ Volume features
    df["volume_sma"] = df["VOLUME"].rolling(20).mean()
    df["volume_ratio"] = df["VOLUME"] / (df["volume_sma"] + 1e-8)
    
    # 8️⃣ Momentum
    df["momentum_5"] = df["CLOSE"] - df["CLOSE"].shift(5)
    df["momentum_10"] = df["CLOSE"] - df["CLOSE"].shift(10)
    
    # 9️⃣ Price patterns
    df["high_low_ratio"] = df["HIGH"] / df["LOW"]
    df["close_open_ratio"] = df["CLOSE"] / df["OPEN"]
    df["body_size"] = abs(df["CLOSE"] - df["OPEN"]) / df["OPEN"]
    
    # 🔟 Gap
    df["gap"] = (df["OPEN"] - df["CLOSE"].shift(1)) / df["CLOSE"].shift(1)
    
    return df


def _calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcule le RSI."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calcule l'ATR."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr
