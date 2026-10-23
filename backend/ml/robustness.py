"""
INNOVEXA Adversarial Robustness and Suspicious Network Analysis Module
Implements:
1. Feature Masking Evaluation (testing resilience to attribute camouflage)
2. Edge Injection Evaluation (testing resilience to relational camouflage)
3. Suspicious Network & Cluster Analysis
"""

import os
import json
import copy
import numpy as np
import torch
import networkx as nx
from typing import Dict, Any, List, Tuple
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


# ============================================================================
# 1. FEATURE MASKING PERTURBATIONS
# ============================================================================

def run_feature_masking_experiment(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    graph_data: Dict[str, Any],
    mask_rates: List[float] = [0.0, 0.10, 0.20, 0.30, 0.50],
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Evaluates degradation of models when profile attributes are randomly masked.
    Replaces masked feature values with neutral zero (the scaled feature mean).
    """
    np.random.seed(random_seed)
    results = {name: [] for name in models_dict.keys()}
    
    test_mask = graph_data["test_mask"]
    
    for rate in mask_rates:
        # Create perturbed test features for tabular baselines
        X_test_masked = X_test.copy()
        if rate > 0.0:
            mask = np.random.rand(*X_test.shape) < rate
            X_test_masked[mask] = 0.0
            
        # Create perturbed graph data for GNNs
        graph_perturbed = copy.deepcopy(graph_data)
        if rate > 0.0:
            x_pert = graph_perturbed["x"].clone()
            # Perturb only the test nodes in graph to test inference degradation
            test_indices = torch.where(test_mask)[0]
            mask_t = torch.rand(len(test_indices), x_pert.shape[1]) < rate
            x_pert[test_indices] = torch.where(mask_t, torch.zeros_like(x_pert[test_indices]), x_pert[test_indices])
            graph_perturbed["x"] = x_pert
            
        for name, model_info in models_dict.items():
            model = model_info["model"]
            threshold = model_info.get("threshold", 0.5)
            is_gnn = model_info.get("is_gnn", False)
            
            if is_gnn:
                probs = model.predict_proba(graph_perturbed, indices=graph_data["test_indices"])
            else:
                probs = model.predict_proba(X_test_masked)
                
            preds = (probs >= threshold).astype(int)
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            auc = float(roc_auc_score(y_test, probs))
            
            results[name].append({
                "mask_rate": rate,
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4)
            })
            
    return results


# ============================================================================
# 2. EDGE INJECTION PERTURBATIONS
# ============================================================================

def run_edge_injection_experiment(
    gnn_models_dict: Dict[str, Any],
    graph_data: Dict[str, Any],
    y_test: np.ndarray,
    injection_ratios: List[float] = [0.0, 0.05, 0.10, 0.20, 0.30],
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Simulates adversarial relational camouflage by injecting random similarity edges
    between arbitrary pairs of nodes, evaluating GNN degradation.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    num_nodes = graph_data["num_nodes"]
    base_edge_index = graph_data["edge_index"]
    num_base_edges = base_edge_index.shape[1]
    test_mask = graph_data["test_mask"]
    
    results = {name: [] for name in gnn_models_dict.keys()}
    
    for ratio in injection_ratios:
        graph_pert = copy.deepcopy(graph_data)
        
        if ratio > 0.0:
            num_injected = int(num_base_edges * ratio)
            src_injected = torch.randint(0, num_nodes, (num_injected,))
            dst_injected = torch.randint(0, num_nodes, (num_injected,))
            
            # Combine undirected
            new_edges = torch.cat([
                base_edge_index,
                torch.stack([src_injected, dst_injected]),
                torch.stack([dst_injected, src_injected])
            ], dim=1)
            
            graph_pert["edge_index"] = new_edges
            
            # Recompute normalized adjacency for GCN
            all_s = new_edges[0].tolist()
            all_t = new_edges[1].tolist()
            from graph_builder import SimilarityGraphBuilder
            builder = SimilarityGraphBuilder()
            graph_pert["norm_adj"] = builder._compute_normalized_adjacency(num_nodes, all_s, all_t)
            
        for name, model_info in gnn_models_dict.items():
            model = model_info["model"]
            threshold = model_info.get("threshold", 0.5)
            probs = model.predict_proba(graph_pert, indices=graph_data["test_indices"])
            preds = (probs >= threshold).astype(int)
            
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, zero_division=0))
            auc = float(roc_auc_score(y_test, probs))
            
            results[name].append({
                "injection_ratio": ratio,
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(auc, 4)
            })
            
    return results


# ============================================================================
# 3. SUSPICIOUS NETWORK ANALYSIS
# ============================================================================

def analyze_suspicious_network(
    graph_data: Dict[str, Any],
    node_probs: np.ndarray,
    node_labels: np.ndarray,
    threshold: float = 0.50,
    max_export_nodes: int = 150
) -> Dict[str, Any]:
    """
    Analyzes graph topology, high-risk node connections, and candidate clusters.
    Constructs a representative subgraph with node risks, community clusters, and statistics
    for interactive frontend visualization.
    """
    edge_index = graph_data["edge_index"].cpu().numpy()
    num_nodes = len(node_probs)
    
    # Build NetworkX graph for topology analysis
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    sources, targets = edge_index[0], edge_index[1]
    for s, t in zip(sources, targets):
        if s < t:  # Undirected simple edge
            G.add_edge(int(s), int(t))
            
    # Node metrics
    degrees = dict(G.degree())
    
    # Neighborhood risk: ratio of neighbors predicted as fake
    predicted_fake = (node_probs >= threshold).astype(int)
    
    neigh_fake_ratios = {}
    for node in range(num_nodes):
        nbrs = list(G.neighbors(node))
        if nbrs:
            ratio = sum(predicted_fake[nbr] for nbr in nbrs) / len(nbrs)
        else:
            ratio = 0.0
        neigh_fake_ratios[node] = round(float(ratio), 3)
        
    # High-Risk Nodes: fake probability >= 0.70
    high_risk_node_ids = [i for i in range(num_nodes) if node_probs[i] >= 0.70]
    
    # Community detection on high-risk subnetwork using greedy modularity
    high_risk_subgraph = G.subgraph(high_risk_node_ids)
    communities = []
    if len(high_risk_subgraph) > 0:
        try:
            comm_generator = nx.community.greedy_modularity_communities(high_risk_subgraph)
            for c_id, comm in enumerate(comm_generator):
                comm_nodes = list(comm)
                if len(comm_nodes) >= 3:
                    avg_prob = float(np.mean([node_probs[n] for n in comm_nodes]))
                    true_fake_count = int(sum(node_labels[n] for n in comm_nodes))
                    communities.append({
                        "cluster_id": c_id + 1,
                        "size": len(comm_nodes),
                        "node_ids": comm_nodes[:20],  # sample IDs
                        "average_fake_prob": round(avg_prob, 4),
                        "ground_truth_fake_ratio": round(true_fake_count / len(comm_nodes), 3),
                        "risk_level": "Critical" if avg_prob >= 0.85 else "High"
                    })
        except Exception:
            pass
            
    # Subgraph extraction for rich interactive Cytoscape visualization:
    # Select top 75 high-risk fake nodes, top 50 genuine nodes, and 25 border/moderate-risk nodes
    high_risk_sample = sorted(high_risk_node_ids, key=lambda n: node_probs[n], reverse=True)[:75]
    genuine_nodes = [i for i in range(num_nodes) if node_probs[i] <= 0.25]
    genuine_sample = sorted(genuine_nodes, key=lambda n: degrees.get(n, 0), reverse=True)[:50]
    border_nodes = [i for i in range(num_nodes) if 0.40 <= node_probs[i] <= 0.60][:25]
    
    viz_node_ids = set(high_risk_sample + genuine_sample + border_nodes)
    viz_subgraph = G.subgraph(viz_node_ids)
    
    # Format nodes and edges for Cytoscape.js
    nodes_data = []
    for nid in viz_subgraph.nodes():
        prob = float(node_probs[nid])
        label = int(node_labels[nid])
        pred = int(prob >= threshold)
        
        # Risk tier definition
        if prob >= 0.80:
            risk_tier = "Critical"
        elif prob >= 0.60:
            risk_tier = "High"
        elif prob >= 0.40:
            risk_tier = "Moderate"
        else:
            risk_tier = "Low"
            
        nodes_data.append({
            "data": {
                "id": str(nid),
                "label": f"User #{nid}",
                "fake_prob": round(prob, 4),
                "risk_tier": risk_tier,
                "ground_truth": "Fake" if label == 1 else "Genuine",
                "predicted": "Fake" if pred == 1 else "Genuine",
                "degree": degrees.get(nid, 0),
                "neighbor_fake_ratio": neigh_fake_ratios.get(nid, 0.0)
            }
        })
        
    edges_data = []
    for s, t in viz_subgraph.edges():
        edges_data.append({
            "data": {
                "id": f"e_{s}_{t}",
                "source": str(s),
                "target": str(t),
                "weight": round(float(graph_data["edge_weight"][0].item() if len(graph_data["edge_weight"]) > 0 else 0.8), 2)
            }
        })
        
    summary = {
        "total_nodes_in_full_graph": num_nodes,
        "total_edges_in_full_graph": int(len(sources) / 2),
        "high_risk_nodes_count": len(high_risk_node_ids),
        "clusters_identified": len(communities),
        "clusters": communities,
        "visualization_subgraph": {
            "nodes": nodes_data,
            "edges": edges_data
        },
        "disclaimer": (
            "Cluster associations and high-risk classifications reflect model probability estimates "
            "over a feature-similarity graph. They do not constitute conclusive evidence of fraudulent activity."
        )
    }
    
    return summary
