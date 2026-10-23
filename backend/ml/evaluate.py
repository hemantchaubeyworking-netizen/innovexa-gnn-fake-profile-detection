"""
INNOVEXA Evaluation Module
Computes genuine held-out test metrics, tunes decision thresholds strictly on validation data,
and produces structured result tables, ROC curves, PR curves, and confusion matrices.
"""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)


def find_optimal_threshold(y_val: np.ndarray, y_val_probs: np.ndarray) -> float:
    """
    Finds classification threshold that maximizes F1-score on the VALIDATION set.
    Prevents test-label contamination in threshold selection.
    """
    best_thresh = 0.50
    best_f1 = 0.0
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for th in thresholds:
        preds = (y_val_probs >= th).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(th)
            
    return round(best_thresh, 4)


def evaluate_predictions(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "Model",
    train_time: float = 0.0,
    inference_time_ms: float = 0.0
) -> Dict[str, Any]:
    """Calculates all scientific evaluation metrics on the held-out test set."""
    y_pred = (y_probs >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_probs))
    pr_auc = float(average_precision_score(y_true, y_probs))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    # Compute ROC Curve points (downsampled to 50 points for clean UI charting)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    idx_sample = np.linspace(0, len(fpr) - 1, min(50, len(fpr))).astype(int)
    roc_points = [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx_sample]
    
    # Compute PR Curve points
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
    idx_pr = np.linspace(0, len(r_curve) - 1, min(50, len(r_curve))).astype(int)
    pr_points = [{"recall": round(float(r_curve[i]), 4), "precision": round(float(p_curve[i]), 4)} for i in idx_pr]
    
    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "threshold": round(threshold, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": [[tn, fp], [fn, tp]]
        },
        "train_time_sec": round(train_time, 3),
        "inference_latency_ms": round(inference_time_ms, 3),
        "roc_curve": roc_points,
        "pr_curve": pr_points
    }


def save_experiment_results(all_metrics: List[Dict[str, Any]], output_dir: str):
    """Saves metrics in both JSON and CSV formats, and prepares curve datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary Metrics Table (without full curve points for concise tabular viewing)
    summary_list = []
    for m in all_metrics:
        item = {k: v for k, v in m.items() if k not in ["roc_curve", "pr_curve", "confusion_matrix"]}
        cm = m["confusion_matrix"]
        item["true_positives"] = cm["true_positives"]
        item["false_positives"] = cm["false_positives"]
        item["true_negatives"] = cm["true_negatives"]
        item["false_negatives"] = cm["false_negatives"]
        summary_list.append(item)
        
    df_metrics = pd.DataFrame(summary_list)
    df_metrics.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    # 2. Complete Metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
        
    # 3. Dedicated ROC and PR curve JSONs
    roc_data = {m["model_name"]: m["roc_curve"] for m in all_metrics}
    with open(os.path.join(output_dir, "roc_curves.json"), "w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2)
        
    pr_data = {m["model_name"]: m["pr_curve"] for m in all_metrics}
    with open(os.path.join(output_dir, "pr_curves.json"), "w", encoding="utf-8") as f:
        json.dump(pr_data, f, indent=2)
        
    print(f"Results successfully saved to {output_dir}")
