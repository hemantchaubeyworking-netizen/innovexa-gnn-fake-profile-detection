"""
INNOVEXA Graph Neural Network Architectures
Implements:
1. Graph Convolutional Network (GCN)
2. GraphSAGE (with inductive neighborhood aggregation)
3. Graph Attention Network (GAT) with multi-head self-attention
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List


# ============================================================================
# 1. GRAPH CONVOLUTIONAL NETWORK (GCN)
# ============================================================================

class GCNLayer(nn.Module):
    r"""
    Graph Convolutional Layer using renormalized adjacency:
    H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})
    where \hat{A} = \tilde{D}^{-1/2} (A + I) \tilde{D}^{-1/2}
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # Support = X * W
        support = torch.mm(x, self.weight)
        # Output = \hat{A} * Support
        output = torch.sparse.mm(norm_adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCNModel(nn.Module):
    """2-layer GCN with dropout and classification head."""
    def __init__(self, in_features: int = 12, hidden_dim: int = 64, out_features: int = 32, dropout: float = 0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNLayer(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNLayer(hidden_dim, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.classifier = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.dropout(self.relu(self.bn1(self.conv1(x, norm_adj))))
        h2 = self.dropout(self.relu(self.bn2(self.conv2(h1, norm_adj))))
        logits = self.classifier(h2).squeeze(-1)
        return logits, h2


# ============================================================================
# 2. GRAPHSAGE (SAMPLE AND AGGREGATE)
# ============================================================================

class SAGEConvLayer(nn.Module):
    r"""
    GraphSAGE Layer with mean aggregator:
    h_{N(u)} = Mean({h_v : v \in N(u)})
    h_u^{(l+1)} = \sigma(W_self * h_u + W_neigh * h_{N(u)})
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(SAGEConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_self = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_neigh = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_self)
        nn.init.xavier_uniform_(self.weight_neigh)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        source_nodes, target_nodes = edge_index[0], edge_index[1]
        
        # Aggregate messages via scatter mean
        neigh_sum = torch.zeros_like(x)
        neigh_count = torch.zeros((num_nodes, 1), device=x.device, dtype=x.dtype)
        
        neigh_sum.index_add_(0, target_nodes, x[source_nodes])
        neigh_count.index_add_(0, target_nodes, torch.ones((len(source_nodes), 1), device=x.device, dtype=x.dtype))
        
        neigh_mean = neigh_sum / torch.clamp(neigh_count, min=1.0)
        
        # Transform self and neighborhood
        out_self = torch.mm(x, self.weight_self)
        out_neigh = torch.mm(neigh_mean, self.weight_neigh)
        out = out_self + out_neigh
        
        if self.bias is not None:
            out = out + self.bias
            
        # L2-normalization for GraphSAGE representation stability
        out = F.normalize(out, p=2, dim=-1)
        return out


class GraphSAGEModel(nn.Module):
    """2-layer GraphSAGE architecture supporting transductive and inductive inference."""
    def __init__(self, in_features: int = 12, hidden_dim: int = 64, out_features: int = 32, dropout: float = 0.3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConvLayer(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = SAGEConvLayer(hidden_dim, out_features)
        self.classifier = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.dropout(self.relu(self.conv1(x, edge_index)))
        h2 = self.dropout(self.relu(self.conv2(h1, edge_index)))
        logits = self.classifier(h2).squeeze(-1)
        return logits, h2

    def inductive_inference(
        self,
        query_feature: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[float, np.ndarray]:
        """
        Inductive inference for an unseen node:
        Aggregates context from its nearest training neighbors without retraining the graph.
        """
        self.eval()
        with torch.no_grad():
            q_feat = query_feature.view(1, -1)
            # Level 1 aggregation: Mean of neighbor input features
            neigh_feat_mean = neighbor_features.mean(dim=0, keepdim=True)
            
            # Conv1 forward for query node
            h1_self = torch.mm(q_feat, self.conv1.weight_self)
            h1_neigh = torch.mm(neigh_feat_mean, self.conv1.weight_neigh)
            h1 = F.normalize(F.relu(h1_self + h1_neigh), p=2, dim=-1)
            
            # Conv2 forward: Use neighbor hidden representations if available, else approximate
            if neighbor_embeddings is not None and len(neighbor_embeddings) > 0:
                neigh_emb_mean = neighbor_embeddings.mean(dim=0, keepdim=True)
            else:
                # Forward neighbors through conv1
                n_self = torch.mm(neighbor_features, self.conv1.weight_self)
                n_neigh = torch.mm(q_feat.expand(len(neighbor_features), -1), self.conv1.weight_neigh)
                neigh_emb_mean = F.normalize(F.relu(n_self + n_neigh), p=2, dim=-1).mean(dim=0, keepdim=True)
                
            h2_self = torch.mm(h1, self.conv2.weight_self)
            h2_neigh = torch.mm(neigh_emb_mean, self.conv2.weight_neigh)
            h2 = F.normalize(F.relu(h2_self + h2_neigh), p=2, dim=-1)
            
            logit = self.classifier(h2).squeeze(-1)
            prob = torch.sigmoid(logit).item()
            
            return prob, h2.cpu().numpy().flatten()


# ============================================================================
# 3. GRAPH ATTENTION NETWORK (GAT)
# ============================================================================

class GATLayer(nn.Module):
    """Multi-Head Graph Attention Layer with LeakyReLU and Softmax attention."""
    def __init__(self, in_features: int, out_features: int, heads: int = 4, dropout: float = 0.2, concat: bool = True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        
        self.W = nn.Parameter(torch.FloatTensor(heads, in_features, out_features))
        self.a_src = nn.Parameter(torch.FloatTensor(heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.FloatTensor(heads, out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # Shape: [heads, num_nodes, out_features]
        h = torch.matmul(x.unsqueeze(0), self.W)
        
        # Attention scores: alpha_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        score_src = torch.matmul(h, self.a_src).squeeze(-1)  # [heads, num_nodes]
        score_dst = torch.matmul(h, self.a_dst).squeeze(-1)  # [heads, num_nodes]
        
        edge_scores = score_src[:, src] + score_dst[:, dst]  # [heads, num_edges]
        edge_scores = self.leaky_relu(edge_scores)
        
        # Softmax over incoming edges for each destination node
        # Compute exp and aggregate
        exp_scores = torch.exp(edge_scores - edge_scores.max(dim=1, keepdim=True)[0])
        
        # Scatter sum denominator per target node
        denom = torch.zeros((self.heads, num_nodes), device=x.device, dtype=x.dtype)
        denom.scatter_add_(1, dst.unsqueeze(0).expand(self.heads, -1), exp_scores)
        denom = torch.clamp(denom, min=1e-8)
        
        alpha = exp_scores / denom.gather(1, dst.unsqueeze(0).expand(self.heads, -1))
        alpha = self.dropout(alpha)  # [heads, num_edges]
        
        # Aggregate messages: \sum_j \alpha_ij * h_j
        weighted_msgs = h[:, src, :] * alpha.unsqueeze(-1)  # [heads, num_edges, out_features]
        
        out = torch.zeros((self.heads, num_nodes, self.out_features), device=x.device, dtype=x.dtype)
        for head in range(self.heads):
            out[head].index_add_(0, dst, weighted_msgs[head])
            
        if self.concat:
            # Flatten multi-head outputs: [num_nodes, heads * out_features]
            out = out.permute(1, 0, 2).reshape(num_nodes, self.heads * self.out_features)
        else:
            # Average heads: [num_nodes, out_features]
            out = out.mean(dim=0)
            
        return out


class GATModel(nn.Module):
    """Graph Attention Network with 4-head attention and classification head."""
    def __init__(self, in_features: int = 12, hidden_dim: int = 16, heads: int = 4, out_features: int = 32, dropout: float = 0.3):
        super(GATModel, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.gat2 = GATLayer(hidden_dim * heads, out_features, heads=1, dropout=dropout, concat=False)
        self.classifier = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.dropout(self.relu(self.gat1(x, edge_index)))
        h2 = self.dropout(self.relu(self.gat2(h1, edge_index)))
        logits = self.classifier(h2).squeeze(-1)
        return logits, h2


# ============================================================================
# UNIFIED GNN TRAINER WRAPPER
# ============================================================================

class GNNTrainer:
    """Trainer and evaluation manager for GCN, GraphSAGE, and GAT models."""
    def __init__(self, model_type: str, in_features: int = 12, lr: float = 0.005, weight_decay: float = 5e-4):
        self.model_type = model_type
        self.device = torch.device("cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_time = 0.0
        self.train_history = []
        
        if model_type == "GCN":
            self.model = GCNModel(in_features=in_features).to(self.device)
        elif model_type == "GraphSAGE":
            self.model = GraphSAGEModel(in_features=in_features).to(self.device)
        elif model_type == "GAT":
            self.model = GATModel(in_features=in_features).to(self.device)
        else:
            raise ValueError(f"Unknown GNN model type: {model_type}")

    def fit(
        self,
        graph_data: Dict[str, Any],
        epochs: int = 120,
        patience: int = 25
    ):
        start_t = time.perf_counter()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        x = graph_data["x"].to(self.device)
        y = graph_data["y"].to(self.device)
        edge_index = graph_data["edge_index"].to(self.device)
        norm_adj = graph_data["norm_adj"].to(self.device)
        
        train_indices = graph_data["train_indices"].to(self.device) if "train_indices" in graph_data else graph_data["train_mask"].to(self.device)
        val_indices = graph_data["val_indices"].to(self.device) if "val_indices" in graph_data else graph_data["val_mask"].to(self.device)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        self.train_history = []
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            
            if self.model_type == "GCN":
                logits, _ = self.model(x, norm_adj)
            else:
                logits, _ = self.model(x, edge_index)
                
            loss = criterion(logits[train_indices], y[train_indices])
            loss.backward()
            optimizer.step()
            
            # Validation step
            self.model.eval()
            with torch.no_grad():
                if self.model_type == "GCN":
                    v_logits, _ = self.model(x, norm_adj)
                else:
                    v_logits, _ = self.model(x, edge_index)
                val_loss = criterion(v_logits[val_indices], y[val_indices]).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                
            self.train_history.append({
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss)
            })
            
            if patience_counter >= patience:
                break
                
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        self.train_time = time.perf_counter() - start_t
        return self

    def predict_proba(
        self,
        graph_data: Dict[str, Any],
        indices: Optional[Any] = None,
        mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = graph_data["x"].to(self.device)
            if self.model_type == "GCN":
                norm_adj = graph_data["norm_adj"].to(self.device)
                logits, _ = self.model(x, norm_adj)
            else:
                edge_index = graph_data["edge_index"].to(self.device)
                logits, _ = self.model(x, edge_index)
                
            probs = torch.sigmoid(logits)
            if indices is not None:
                if isinstance(indices, np.ndarray):
                    indices = torch.tensor(indices, dtype=torch.long, device=x.device)
                elif not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(list(indices), dtype=torch.long, device=x.device)
                probs = probs[indices]
            elif mask is not None:
                probs = probs[mask]
            return probs.cpu().numpy().astype(np.float32)

    def extract_node_embeddings(self, graph_data: Dict[str, Any]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = graph_data["x"].to(self.device)
            if self.model_type == "GCN":
                norm_adj = graph_data["norm_adj"].to(self.device)
                _, embs = self.model(x, norm_adj)
            else:
                edge_index = graph_data["edge_index"].to(self.device)
                _, embs = self.model(x, edge_index)
            return embs.cpu().numpy()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_type": self.model_type,
            "state_dict": self.model.state_dict(),
            "train_time": self.train_time,
            "train_history": self.train_history
        }, filepath)

    @classmethod
    def load(cls, filepath: str, in_features: int = 12):
        data = torch.load(filepath, map_location="cpu")
        instance = cls(model_type=data["model_type"], in_features=in_features)
        instance.model.load_state_dict(data["state_dict"])
        instance.train_time = data.get("train_time", 0.0)
        instance.train_history = data.get("train_history", [])
        instance.model.eval()
        return instance
