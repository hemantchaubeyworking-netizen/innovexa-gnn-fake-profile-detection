"""
INNOVEXA Graph Construction Module
Constructs an attributed k-nearest-neighbor feature-similarity graph.

RESEARCH LIMITATION NOTICE:
The dataset does not contain actual social interaction or follower-following edges.
The constructed graph is a feature-similarity graph where an edge represents proximity
in the normalized 12-dimensional attribute space. It does NOT denote real-world social
connections. This distinction is preserved across all experiments and user interfaces.
"""

import os
import json
import torch
import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, Tuple, Optional
from sklearn.neighbors import NearestNeighbors


class SimilarityGraphBuilder:
    def __init__(self, k_neighbors: int = 5, metric: str = 'cosine'):
        """
        Args:
            k_neighbors: Number of nearest neighbors per node (excluding self).
            metric: Similarity metric ('cosine' or 'euclidean').
        """
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.nn_model = None
        self.training_features = None

    def fit_training_nodes(self, X_train: np.ndarray):
        """Fits nearest neighbor index on training nodes for inductive neighbor lookup."""
        self.training_features = X_train
        # n_neighbors=k+1 because the query node itself will be its closest neighbor
        self.nn_model = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1,
            metric=self.metric,
            algorithm='auto',
            n_jobs=-1
        )
        self.nn_model.fit(X_train)
        return self

    def build_transductive_graph(
        self,
        X_full: np.ndarray,
        y_full: np.ndarray,
        train_indices: np.ndarray,
        val_indices: np.ndarray,
        test_indices: np.ndarray
    ) -> Dict[str, Any]:
        """
        Builds the global feature-similarity graph over all nodes.
        Edges are constructed solely from normalized profile features WITHOUT target labels.
        """
        num_nodes = X_full.shape[0]
        
        # Fit kNN over full feature matrix
        full_nn = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1,
            metric=self.metric,
            algorithm='auto',
            n_jobs=-1
        )
        full_nn.fit(X_full)
        
        # Find k-nearest neighbors
        distances, indices = full_nn.kneighbors(X_full)
        
        # Build undirected edge list, excluding self-loops initially
        sources = []
        targets = []
        weights = []
        
        for i in range(num_nodes):
            for rank in range(1, self.k_neighbors + 1):  # Skip rank 0 (self)
                neighbor = indices[i, rank]
                dist = distances[i, rank]
                sim = 1.0 / (1.0 + dist) if self.metric == 'euclidean' else max(0.0, 1.0 - dist)
                
                # Directed edge i -> neighbor
                sources.append(i)
                targets.append(neighbor)
                weights.append(sim)
                
                # Ensure symmetry / undirected edge neighbor -> i
                sources.append(neighbor)
                targets.append(i)
                weights.append(sim)
                
        # Remove duplicate edges
        edge_set = set()
        clean_sources = []
        clean_targets = []
        clean_weights = []
        
        for s, t, w in zip(sources, targets, weights):
            edge_key = (s, t)
            if edge_key not in edge_set and s != t:
                edge_set.add(edge_key)
                clean_sources.append(s)
                clean_targets.append(t)
                clean_weights.append(w)
                
        edge_index = torch.tensor([clean_sources, clean_targets], dtype=torch.long)
        edge_weight = torch.tensor(clean_weights, dtype=torch.float32)
        
        # PyTorch Tensors for features and targets
        x_tensor = torch.tensor(X_full, dtype=torch.float32)
        y_tensor = torch.tensor(y_full, dtype=torch.float32)
        
        # Boolean masks for train / val / test sets
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        # Compute normalized adjacency matrix for GCN message passing: \hat{A} = D^{-1/2} (A + I) D^{-1/2}
        norm_adj = self._compute_normalized_adjacency(num_nodes, clean_sources, clean_targets)
        
        graph_data = {
            "x": x_tensor,
            "y": y_tensor,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "norm_adj": norm_adj,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "train_indices": torch.tensor(train_indices, dtype=torch.long),
            "val_indices": torch.tensor(val_indices, dtype=torch.long),
            "test_indices": torch.tensor(test_indices, dtype=torch.long),
            "num_nodes": num_nodes,
            "num_edges": len(clean_sources),
            "k_neighbors": self.k_neighbors,
            "metric": self.metric,
            "graph_type": "Feature-Similarity kNN Graph (Unsupervised Construction)",
            "limitations_notice": (
                "Edges represent metric proximity in the normalized feature space, "
                "not confirmed social relationships or follower interactions."
            )
        }
        
        return graph_data

    def _compute_normalized_adjacency(
        self,
        num_nodes: int,
        sources: list,
        targets: list
    ) -> torch.Tensor:
        r"""
        Computes symmetric normalized adjacency with self-loops:
        \tilde{A} = A + I
        \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}
        """
        # Create sparse matrix with self-loops
        all_s = sources + list(range(num_nodes))
        all_t = targets + list(range(num_nodes))
        data = np.ones(len(all_s), dtype=np.float32)
        
        adj = sp.coo_matrix((data, (all_s, all_t)), shape=(num_nodes, num_nodes))
        # Ensure symmetric
        adj = adj.maximum(adj.transpose())
        
        # Degree vector
        rowsum = np.array(adj.sum(1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj_sp = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()
        
        # Convert to PyTorch sparse tensor
        indices = torch.tensor(np.vstack((norm_adj_sp.row, norm_adj_sp.col)), dtype=torch.long)
        values = torch.tensor(norm_adj_sp.data, dtype=torch.float32)
        shape = torch.Size(norm_adj_sp.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def get_inductive_neighbors(self, query_feature: np.ndarray, k: Optional[int] = None) -> Dict[str, Any]:
        """
        For an unseen new profile, retrieves its k-nearest neighbors in the training graph.
        Used by GraphSAGE for inductive message-passing and real-time inference.
        """
        if self.nn_model is None:
            raise ValueError("Training nodes have not been fitted. Call fit_training_nodes first.")
            
        k_val = k if k is not None else self.k_neighbors
        query_vec = query_feature.reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=k_val)
        
        neighbor_indices = indices[0].tolist()
        neighbor_distances = distances[0].tolist()
        
        return {
            "neighbor_indices": neighbor_indices,
            "neighbor_distances": neighbor_distances,
            "similarity_weights": [
                float(1.0 / (1.0 + d) if self.metric == 'euclidean' else max(0.0, 1.0 - d))
                for d in neighbor_distances
            ]
        }


if __name__ == "__main__":
    from preprocessing import prepare_dataset_splits
    data_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset.csv")
    splits = prepare_dataset_splits(data_file)
    
    builder = SimilarityGraphBuilder(k_neighbors=5, metric='cosine')
    builder.fit_training_nodes(splits["X_train"])
    
    graph = builder.build_transductive_graph(
        splits["X_full_scaled"],
        splits["y_full"],
        splits["train_indices"],
        splits["val_indices"],
        splits["test_indices"]
    )
    
    print("Graph built successfully:")
    print("Total Nodes:", graph["num_nodes"])
    print("Total Directed Edges:", graph["num_edges"])
    print("Edge index shape:", graph["edge_index"].shape)
    print("Normalized adjacency shape:", graph["norm_adj"].shape)
    
    # Test inductive neighbor query
    test_sample = splits["X_test"][0]
    sample_neighbors = builder.get_inductive_neighbors(test_sample)
    print("Inductive neighbors for unseen sample:", sample_neighbors)
