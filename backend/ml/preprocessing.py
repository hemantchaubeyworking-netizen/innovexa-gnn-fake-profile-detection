"""
INNOVEXA Preprocessing Pipeline
Handles data loading, validation, leakage-free feature scaling, and train/val/test splits.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'profile pic',
    'nums/length username',
    'fullname words',
    'nums/length fullname',
    'name==username',
    'description length',
    'external URL',
    'private',
    '#posts',
    '#followers',
    '#follows',
    'followers_following_ratio'
]

TARGET_COLUMN = 'fake'

COUNT_FEATURES = ['#posts', '#followers', '#follows', 'followers_following_ratio', 'description length']


class Preprocessor:
    def __init__(self, scaler=None, feature_stats=None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.feature_stats = feature_stats if feature_stats is not None else {}
        self.feature_names = FEATURE_COLUMNS
        self.fitted = False

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Applies feature transformations (log1p on high-skew counts, robust normalization)."""
        df_copy = df[self.feature_names].copy()
        
        # Ensure correct numeric types
        for col in self.feature_names:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0)
            
        # Log1p transform on heavy-tailed count/ratio features
        for col in COUNT_FEATURES:
            # Handle potential negative values safely
            df_copy[col] = np.log1p(np.maximum(df_copy[col].values, 0.0))
            
        values = df_copy.values.astype(np.float32)
        return values

    def fit(self, X_train: pd.DataFrame):
        """Fits scaler strictly on the training partition to prevent test data leakage."""
        transformed = self._transform_features(X_train)
        self.scaler.fit(transformed)
        self.fitted = True
        
        # Calculate summary statistics for documentation
        self.feature_stats = {
            col: {
                "mean": float(X_train[col].mean()),
                "std": float(X_train[col].std()),
                "min": float(X_train[col].min()),
                "max": float(X_train[col].max()),
                "median": float(X_train[col].median()),
            }
            for col in self.feature_names
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms feature matrix using parameters learned from training partition."""
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted yet.")
        transformed = self._transform_features(X)
        return self.scaler.transform(transformed).astype(np.float32)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def transform_single(self, sample_dict: Dict[str, Any]) -> np.ndarray:
        """Transforms a single profile record for real-time inference."""
        # Calculate followers_following_ratio if missing or zero
        if 'followers_following_ratio' not in sample_dict or sample_dict['followers_following_ratio'] is None:
            followers = float(sample_dict.get('#followers', 0))
            follows = float(sample_dict.get('#follows', 0))
            sample_dict['followers_following_ratio'] = followers / max(follows, 1.0)
            
        df = pd.DataFrame([sample_dict])
        return self.transform(df)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "feature_stats": self.feature_stats,
            "feature_names": self.feature_names,
            "fitted": self.fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        instance = cls(scaler=data["scaler"], feature_stats=data["feature_stats"])
        instance.feature_names = data["feature_names"]
        instance.fitted = data["fitted"]
        return instance


def load_dataset(data_path: str) -> pd.DataFrame:
    """Loads and validates the exact uploaded dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Verify exact required columns
    missing_cols = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
        
    return df


def prepare_dataset_splits(
    data_path: str,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs a stratified 70/15/15 split and scales features using training-set parameters.
    Guarantees no test-set information leaks into the preprocessing or graph construction.
    """
    df = load_dataset(data_path)
    
    # Exclude target from features
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].values.astype(np.int64)
    
    # Overall dataset statistics
    total_samples = len(df)
    fake_count = int(np.sum(y == 1))
    genuine_count = int(np.sum(y == 0))
    duplicate_count = int(df.duplicated().sum())
    
    # Step 1: Split into train (70%) and temporary holdout (30%)
    holdout_size = val_size + test_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        train_size=train_size,
        stratify=y,
        random_state=random_state
    )
    
    # Step 2: Split holdout 50/50 into validation (15%) and test (15%)
    val_ratio_of_holdout = val_size / holdout_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout,
        train_size=val_ratio_of_holdout,
        stratify=y_holdout,
        random_state=random_state
    )
    
    # Preprocessing: Fit scaler STRICTLY on training split
    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Keep full scaled matrix in original row ordering for transductive graph building
    X_full_scaled = preprocessor.transform(X)
    
    # Retain index masks for train/val/test partitions
    train_indices = X_train.index.values
    val_indices = X_val.index.values
    test_indices = X_test.index.values
    
    splits = {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "X_full_scaled": X_full_scaled,
        "y_full": y,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "preprocessor": preprocessor,
        "stats": {
            "total_samples": total_samples,
            "genuine_count": genuine_count,
            "fake_count": fake_count,
            "duplicate_count": duplicate_count,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "feature_names": FEATURE_COLUMNS,
            "feature_stats": preprocessor.feature_stats
        }
    }
    
    return splits


if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset.csv")
    print(f"Loading from: {data_file}")
    res = prepare_dataset_splits(data_file)
    print("Train shape:", res["X_train"].shape)
    print("Val shape:", res["X_val"].shape)
    print("Test shape:", res["X_test"].shape)
    print("Stats:", json.dumps(res["stats"], indent=2))
