"""Feature engineering shared by the notebooks and the demo app (src/features.py)."""
import numpy as np
import pandas as pd

CATEGORICAL = ["merchant_category", "card_type"]
NUMERIC_TO_SCALE = ["log_amount", "log_gap", "log_ratio"]
# fixed category lists, so a new file with a missing category still gets every column
CATEGORIES = {
    "merchant_category": ["Electronics", "Fuel", "Grocery", "Healthcare", "Online", "Restaurant", "Travel"],
    "card_type": ["Credit", "Debit", "Prepaid"],
}
FEATURES = (NUMERIC_TO_SCALE + ["hour_sin", "hour_cos", "dow"]
            + [f"merchant_category_{c}" for c in CATEGORIES["merchant_category"]]
            + [f"card_type_{c}" for c in CATEGORIES["card_type"]])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df (sorted by time) with model features added. Uses only PAST
    information per card (previous transactions), so there is no look-ahead leakage."""
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    hour = d["timestamp"].dt.hour
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["dow"] = d["timestamp"].dt.dayofweek
    d["log_amount"] = np.log1p(d["amount"])
    gap = d.groupby("card_token")["timestamp"].diff().dt.total_seconds()
    d["log_gap"] = np.log1p(gap.fillna(7 * 86400))          # first transaction of a card: 7 days
    prev_mean = d.groupby("card_token")["amount"].transform(
        lambda s: s.shift().expanding().mean())
    d["log_ratio"] = np.log1p(d["amount"] / prev_mean).fillna(0)
    for col, cats in CATEGORIES.items():
        for c in cats:
            d[f"{col}_{c}"] = (d[col] == c).astype(int)
    return d


def to_matrix(d: pd.DataFrame, scaler) -> pd.DataFrame:
    """Select model columns and scale the numeric ones with a scaler fitted on TRAIN only."""
    X = d[FEATURES].copy()
    X[NUMERIC_TO_SCALE] = scaler.transform(X[NUMERIC_TO_SCALE])
    return X
