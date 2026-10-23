"""
INNOVEXA Baseline Models Module
Implements traditional machine learning and non-graph deep learning baselines:
1. Logistic Regression
2. Support Vector Machine (RBF kernel)
3. Random Forest
4. Multilayer Perceptron (MLP)
"""

import os
import joblib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class MLBaseline:
    """Wrapper for scikit-learn models providing unified fit/predict/save/load API."""
    def __init__(self, model_name: str, model_instance):
        self.model_name = model_name
        self.model = model_instance
        self.train_time = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        start_t = time.perf_counter()
        self.model.fit(X_train, y_train)
        self.train_time = time.perf_counter() - start_t
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns 1D array of predicted probabilities for the positive ('fake') class."""
        probs = self.model.predict_proba(X)
        return probs[:, 1].astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model_name": self.model_name,
            "model": self.model,
            "train_time": self.train_time
        }, filepath)

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        instance = cls(data["model_name"], data["model"])
        instance.train_time = data.get("train_time", 0.0)
        return instance


def create_logistic_regression() -> MLBaseline:
    clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    return MLBaseline("Logistic Regression", clf)


def create_svm() -> MLBaseline:
    clf = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    return MLBaseline("Support Vector Machine", clf)


def create_random_forest() -> MLBaseline:
    clf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    return MLBaseline("Random Forest", clf)


class MLPNetwork(nn.Module):
    """3-layer feedforward neural network with BatchNorm and Dropout."""
    def __init__(self, in_features: int = 12, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout: float = 0.25):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(hidden_dim2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns (logits, embedding)
        h1 = self.dropout1(self.relu(self.bn1(self.fc1(x))))
        h2 = self.dropout2(self.relu(self.bn2(self.fc2(h1))))
        logits = self.fc_out(h2).squeeze(-1)
        return logits, h2


class MLPClassifier:
    """Trainer and inference wrapper for Multilayer Perceptron."""
    def __init__(self, in_features: int = 12, hidden_dim1: int = 64, hidden_dim2: int = 32, lr: float = 0.005, weight_decay: float = 1e-4):
        self.model_name = "Multilayer Perceptron (MLP)"
        self.device = torch.device("cpu")
        self.model = MLPNetwork(in_features, hidden_dim1, hidden_dim2).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_time = 0.0
        self.train_history = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 120,
        batch_size: int = 128,
        patience: int = 20
    ):
        start_t = time.perf_counter()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            val_x = torch.tensor(X_val, dtype=torch.float32)
            val_y = torch.tensor(y_val, dtype=torch.float32)
            
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        self.train_history = []
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            
            for bx, by in loader:
                optimizer.zero_grad()
                logits, _ = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(by)
                
            train_loss = epoch_loss / len(train_x)
            
            val_loss = None
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    v_logits, _ = self.model(val_x)
                    val_loss = criterion(v_logits, val_y).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
            record = {"epoch": epoch, "train_loss": float(train_loss)}
            if val_loss is not None:
                record["val_loss"] = float(val_loss)
            self.train_history.append(record)
            
            if patience_counter >= patience:
                # Early stopping triggered
                break
                
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        self.train_time = time.perf_counter() - start_t
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            logits, _ = self.model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(np.int64)

    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            _, embs = self.model(x_tensor)
        return embs.cpu().numpy()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "train_time": self.train_time,
            "train_history": self.train_history
        }, filepath)

    @classmethod
    def load(cls, filepath: str, in_features: int = 12):
        data = torch.load(filepath, map_location="cpu")
        instance = cls(in_features=in_features)
        instance.model.load_state_dict(data["state_dict"])
        instance.train_time = data.get("train_time", 0.0)
        instance.train_history = data.get("train_history", [])
        instance.model.eval()
        return instance
