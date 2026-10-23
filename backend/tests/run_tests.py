"""
INNOVEXA Standalone Test Runner using Python Standard Library unittest.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.preprocessing import load_dataset, prepare_dataset_splits, FEATURE_COLUMNS, TARGET_COLUMN
from backend.ml.graph_builder import SimilarityGraphBuilder
from backend.ml.predict import ProfilePredictor
from backend.app.main import app
from fastapi.testclient import TestClient


class TestInnovexaPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_dataset_integrity(self):
        data_path = os.path.join("data", "dataset.csv")
        df = load_dataset(data_path)
        self.assertEqual(df.shape, (5000, 13))
        self.assertIn(TARGET_COLUMN, df.columns)
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, df.columns)
        self.assertEqual(df[TARGET_COLUMN].isnull().sum(), 0)
        self.assertEqual(int((df[TARGET_COLUMN] == 0).sum()), 2500)
        self.assertEqual(int((df[TARGET_COLUMN] == 1).sum()), 2500)

    def test_02_preprocessing_splits(self):
        data_path = os.path.join("data", "dataset.csv")
        splits = prepare_dataset_splits(data_path, random_state=42)
        self.assertEqual(splits["X_train"].shape, (3500, 12))
        self.assertEqual(splits["X_val"].shape, (750, 12))
        self.assertEqual(splits["X_test"].shape, (750, 12))
        self.assertEqual(len(splits["y_train"]), 3500)
        self.assertEqual(len(splits["y_test"]), 750)
        self.assertTrue(splits["preprocessor"].fitted)

    def test_03_graph_builder(self):
        data_path = os.path.join("data", "dataset.csv")
        splits = prepare_dataset_splits(data_path, random_state=42)
        builder = SimilarityGraphBuilder(k_neighbors=5, metric="cosine")
        builder.fit_training_nodes(splits["X_train"])
        
        sample_vec = splits["X_test"][0]
        nbrs = builder.get_inductive_neighbors(sample_vec)
        self.assertEqual(len(nbrs["neighbor_indices"]), 5)
        self.assertEqual(len(nbrs["similarity_weights"]), 5)
        for sim in nbrs["similarity_weights"]:
            self.assertTrue(0.0 <= sim <= 1.0)

    def test_04_saved_model_inference(self):
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
            self.assertIn(res["predicted_class"], ["Fake", "Genuine"])
            self.assertTrue(0.0 <= res["fake_probability"] <= 1.0)
            self.assertIn(res["risk_category"], ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"])
            self.assertIn("neighborhood_context", res)

    def test_05_api_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")

    def test_06_api_models(self):
        res = self.client.get("/api/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("models", data)
        self.assertGreaterEqual(len(data["models"]), 5)

    def test_07_api_dashboard(self):
        res = self.client.get("/api/dashboard")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["dataset_summary"]["total_profiles"], 5000)
        self.assertEqual(data["dataset_summary"]["genuine_profiles"], 2500)
        self.assertEqual(data["dataset_summary"]["fake_profiles"], 2500)

    def test_08_api_predict(self):
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
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_class"], "Fake")
        self.assertGreaterEqual(data["fake_probability"], 0.50)
        self.assertIn("neighborhood_context", data)


if __name__ == "__main__":
    unittest.main()
