"""
INNOVEXA FastAPI Router Endpoints
"""

import datetime
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from backend.app.schemas.profile import ProfileInputSchema, PredictionResponseSchema
from backend.app.services.detection_service import service
from backend.app.core.database import get_recent_history, clear_history

router = APIRouter()


@router.get("/health", summary="API Health Check")
def get_health():
    return {
        "status": "healthy",
        "service": "INNOVEXA GNN-Based Fake Profile Detection API",
        "version": "1.0.0",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "database": "SQLite (innovexa.db)",
        "models_loaded": len(service.get_available_models())
    }


@router.get("/models", summary="List Available Baseline & GNN Models")
def get_models():
    return {
        "default_model": "GraphSAGE",
        "models": service.get_available_models()
    }


@router.get("/dashboard", summary="Retrieve Dashboard Statistics & Highlights")
def get_dashboard():
    return service.get_dashboard_data()


@router.get("/metrics", summary="Retrieve Held-Out Test Evaluation Metrics")
def get_metrics():
    return {
        "dataset_protocol": "Held-Out Test Set (15% = 750 samples, Stratified)",
        "metrics": service.get_metrics(),
        "roc_curves": service.get_roc_curves(),
        "pr_curves": service.get_pr_curves(),
        "loss_history": service.get_loss_history()
    }


@router.get("/network", summary="Retrieve Interactive Similarity Graph Subnetwork & Cluster Analysis")
def get_network():
    return service.get_network_analysis()


@router.get("/robustness", summary="Retrieve Adversarial Robustness Test Results")
def get_robustness():
    return service.get_robustness_results()


@router.get("/samples", summary="Retrieve Preset Profile Test Samples from Dataset")
def get_sample_profiles():
    return {
        "samples": service.get_sample_profiles()
    }


@router.post("/predict", response_model=PredictionResponseSchema, summary="Predict Whether an Instagram Profile is Fake or Genuine")
def predict_profile(payload: ProfileInputSchema):
    try:
        model_name = payload.model_name or "GraphSAGE"
        features = payload.to_dataset_features()
        result = service.predict_profile(features, model_name=model_name)
        result["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/history", summary="Retrieve Prediction History Records")
def get_history(limit: int = Query(default=50, ge=1, le=200)):
    records = get_recent_history(limit=limit)
    return {
        "count": len(records),
        "history": records
    }


@router.delete("/history", summary="Clear Prediction History")
def delete_history():
    clear_history()
    return {"message": "Prediction history cleared successfully."}
