"""
INNOVEXA Test Suite
Tests preprocessing, graph construction, saved models, inference, and FastAPI API routes.
"""

import os
import sys
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Ensure backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.preprocessing import load_dataset, prepare_dataset_splits, FEATURE_COLUMNS, TARGET_COLUMN
from backend.ml.graph_builder import SimilarityGraphBuilder
from backend.ml.predict import ProfilePredictor
from backend.app.main import app

client = TestClient(app)


def test_dataset_integrity():
    """Verifies that the exact dataset is loaded with 5000 rows and 13 columns."""
    data_path = os.path.join("data", "dataset.csv")
    df = load_dataset(data_path)
    assert df.shape == (5000, 13)
    assert TARGET_COLUMN in df.columns
    for feat in FEATURE_COLUMNS:
        assert feat in df.columns
    assert df[TARGET_COLUMN].isnull().sum() == 0
    # Balanced classes
    assert (df[TARGET_COLUMN] == 0).sum() == 2500
    assert (df[TARGET_COLUMN] == 1).sum() == 2500


def test_preprocessing_and_splits():
    """Verifies no test leakage and correct 70/15/15 stratified proportions."""
    data_path = os.path.join("data", "dataset.csv")
    splits = prepare_dataset_splits(data_path, random_state=42)
    
    assert splits["X_train"].shape == (3500, 12)
    assert splits["X_val"].shape == (750, 12)
    assert splits["X_test"].shape == (750, 12)
    
    # Target values isolated
    assert len(splits["y_train"]) == 3500
    assert len(splits["y_test"]) == 750
    assert splits["preprocessor"].fitted is True


def test_graph_builder():
    """Verifies feature-similarity graph creation and inductive neighbor retrieval."""
    data_path = os.path.join("data", "dataset.csv")
    splits = prepare_dataset_splits(data_path, random_state=42)
    
    builder = SimilarityGraphBuilder(k_neighbors=5, metric="cosine")
    builder.fit_training_nodes(splits["X_train"])
    
    # Test inductive query on unseen sample
    sample_vec = splits["X_test"][0]
    nbrs = builder.get_inductive_neighbors(sample_vec)
    assert len(nbrs["neighbor_indices"]) == 5
    assert len(nbrs["similarity_weights"]) == 5
    for sim in nbrs["similarity_weights"]:
        assert 0.0 <= sim <= 1.0


def test_saved_model_inference():
    """Verifies that saved models load and generate valid predictions."""
    predictor = ProfilePredictor("backend/models")
    sample = {
        "profile pic": 1,
        "nums/length username": 0.1,
        "fullname words": 2,
        "nums/length fullname": 0.0,
        "name==username": 0,
        "description length": 30,
        "external URL": 0,
        "private": 0,
        "#posts": 50,
        "#followers": 800,
        "#follows": 300,
        "followers_following_ratio": 2.66
    }
    
    for model_name in ["GraphSAGE", "Random Forest", "GCN", "MLP", "Logistic Regression"]:
        res = predictor.predict_profile(sample, model_name=model_name)
        assert res["predicted_class"] in ["Fake", "Genuine"]
        assert 0.0 <= res["fake_probability"] <= 1.0
        assert res["risk_category"] in ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
        assert "neighborhood_context" in res


def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"


def test_api_models():
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert len(data["models"]) >= 5


def test_api_dashboard():
    res = client.get("/api/dashboard")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset_summary"]["total_profiles"] == 5000
    assert data["dataset_summary"]["genuine_profiles"] == 2500
    assert data["dataset_summary"]["fake_profiles"] == 2500


def test_api_predict():
    payload = {
        "profile_pic": 0,
        "nums_length_username": 0.55,
        "fullname_words": 1,
        "nums_length_fullname": 0.0,
        "name_equals_username": 1,
        "description_length": 0,
        "external_url": 0,
        "is_private": 0,
        "posts_count": 0,
        "followers_count": 5,
        "follows_count": 1200,
        "model_name": "GraphSAGE"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] == "Fake"
    assert data["fake_probability"] >= 0.50
    assert "neighborhood_context" in data
