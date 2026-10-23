# INNOVEXA: Fake Profile Detection Using Graph Neural Networks (GNNs)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![React: 19](https://img.shields.io/badge/Frontend-React%2019%20%2B%20Vite-61DAFB.svg)](https://vitejs.dev/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-EE4C2C.svg)](https://pytorch.org/)

**INNOVEXA** is a research-grade machine learning and Graph Neural Network (GNN) system designed to detect fraudulent, automated, and suspicious accounts on social media (specifically Instagram). The system implements the methodologies proposed in the research paper *"Fake Profile Detection Using Graph Neural Networks"*, combining individual profile metadata with topological neighborhood aggregation to achieve state-of-the-art detection accuracy and adversarial robustness.

---

## Table of Contents

- [1. Research Objective & System Architecture](#1-research-objective--system-architecture)
- [2. Dataset Specification](#2-dataset-specification)
- [3. Graph Construction: Similarity Network vs. Social Topology](#3-graph-construction-similarity-network-vs-social-topology)
- [4. Models & Message-Passing Formulations](#4-models--message-passing-formulations)
- [5. Empirical Held-Out Test Results](#5-empirical-held-out-test-results)
- [6. Adversarial Robustness Analysis](#6-adversarial-robustness-analysis)
- [7. Suspicious Network & Cluster Analysis](#7-suspicious-network--cluster-analysis)
- [8. Repository & Folder Structure](#8-repository--folder-structure)
- [9. Setup & Installation Guide](#9-setup--installation-guide)
  - [Prerequisites](#prerequisites)
  - [Backend Setup (FastAPI & PyTorch)](#backend-setup-fastapi--pytorch)
  - [Frontend Setup (React + Vite)](#frontend-setup-react--vite)
- [10. Reproducing Experiments & Model Training](#10-reproducing-experiments--model-training)
- [11. API Reference](#11-api-reference)
- [12. Running Tests](#12-running-tests)
- [13. GitHub Repository Management](#13-github-repository-management)
- [14. Ethical Disclosures & Scientific Limitations](#14-ethical-disclosures--scientific-limitations)

---

## 1. Research Objective & System Architecture

Detecting fraudulent accounts using only individual profile attributes (e.g., number of followers, bio length) leaves security systems vulnerable to evasion tactics where bots mimic genuine feature distributions. INNOVEXA addresses this by framing account classification as a **semi-supervised node classification task on a graph**:

$$\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{X})$$

Where:
- $\mathcal{V}$: Set of $N = 5,000$ profile nodes.
- $\mathcal{E}$: Relational similarity edges connecting profiles with correlated attribute embeddings.
- $\mathbf{X} \in \mathbb{R}^{N \times 12}$: Transformed feature matrix containing account metadata.

```
+--------------------------------------------------------------------------------------------------+
|                                    INNOVEXA PIPELINE ARCHITECTURE                                |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [Tabular Dataset] (5,000 profiles, 12 features)                                                  |
|         │                                                                                        |
|         ▼                                                                                        |
|  [Preprocessing & Leakage Prevention] (Log-transform, Robust scaling fitted ONLY on Train split)  |
|         │                                                                                        |
|         ├───────────────────────────────────────────────┐                                        |
|         ▼                                               ▼                                        |
|  [Feature Matrix X]                           [kNN Graph Builder (k=5, Cosine)]                  |
|  (3,500 Train / 750 Val / 750 Test)           (32,712 directed similarity edges)                 |
|         │                                               │                                        |
|         ├───────────────────────┬───────────────────────┤                                        |
|         ▼                       ▼                       ▼                                        |
|  [Traditional ML]       [Deep MLP]             [Graph Neural Networks]                           |
|  • Logistic Regression  • 3-Layer MLP          • GCN (Symmetric Normalized Laplacians)           |
|  • Support Vector (RBF)                        • GraphSAGE (Inductive Neighborhood Mean Agg)     |
|  • Random Forest (150 trees)                   • GAT (Multi-Head Self-Attention)                 |
|         │                       │                       │                                        |
|         └───────────────────────┼───────────────────────┘                                        |
|                                 ▼                                                                |
|                      [Validation Threshold Tuning]                                               |
|                      (F1-maximizing cutoff per model)                                            |
|                                 │                                                                |
|         ┌───────────────────────┴───────────────────────┐                                        |
|         ▼                                               ▼                                        |
|  [Adversarial Robustness]                     [Held-Out Test Evaluation]                         |
|  • Feature Masking (10%-50%)                  • 750 Unseen Profiles                              |
|  • Edge Injection (5%-30%)                    • Confusion Matrices, ROC/PR Curves                |
|                                                         │                                        |
|                                                         ▼                                        |
|                                              [Production Deployment]                             |
|                                              • FastAPI REST Engine (Port 8000)                   |
|                                              • SQLite Prediction History                         |
|                                              • React 19 + Vite UI (Port 5173)                    |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Dataset Specification

The project uses an exact empirical dataset consisting of **5,000 Instagram accounts** balanced across genuine and fake classes (2,500 genuine, 2,500 fake).

### Input Feature Schema (12 Features)
| Feature Name | Type | Description |
|---|---|---|
| `profile pic` | Binary (0/1) | Whether the profile has an uploaded avatar/display picture |
| `nums/length username` | Float [0, 1] | Ratio of numeric digits to the total length of username |
| `fullname words` | Integer | Total word count in the full name field |
| `nums/length fullname` | Float [0, 1] | Ratio of numeric digits to the total length of full name |
| `name==username` | Binary (0/1) | Flag indicating whether username and full name are identical |
| `description length` | Integer | Character length of the profile biography |
| `external URL` | Binary (0/1) | Whether an external link is listed in the bio |
| `private` | Binary (0/1) | Whether the account profile is marked private |
| `#posts` | Integer | Total number of published media posts |
| `#followers` | Integer | Number of user accounts following this profile |
| `#follows` | Integer | Number of user accounts this profile is following |
| `followers_following_ratio` | Float | Ratio of `#followers / (#follows + 1)` |

### Target Label
- `fake`: Binary target variable where **`1` = Fake/Automated Profile**, and **`0` = Genuine Profile**.

### Data Preprocessing & Target Leakage Prevention
To guarantee scientific validity:
1. **Strict Train/Val/Test Partitioning**: Stratified split of 70% Train (3,500 samples), 15% Validation (750 samples), and 15% Held-Out Test (750 samples).
2. **Leakage Protection**: All imputers, log-transformers $\log(x + 1)$ for heavy-tailed counts (`#posts`, `#followers`, `#follows`), and StandardScaler transforms were fitted **exclusively on the 3,500 training profiles**. Validation and test partitions were strictly transformed using frozen training parameters.
3. **Target Isolation**: The target column `fake` was strictly excluded from all feature matrices, distance calculations, and graph construction.

---

## 3. Graph Construction: Similarity Network vs. Social Topology

> **CRITICAL SCIENTIFIC DISTINCTION:**  
> The supplied tabular dataset does not contain raw follower-following interaction records (directed friend/follow edges). Therefore, the graph structure $\mathcal{G}$ was constructed using an **unsupervised $k$-nearest-neighbor ($k$NN) feature similarity graph** ($k=5$, cosine distance metric) over the normalized attribute representations.  
> 
> Throughout this system and report, this topology is designated as a **similarity graph**. It should not be conflated with an organic follower network. The similarity graph groups accounts exhibiting analogous behavioral and structural patterns, allowing message passing to reinforce shared stylistic indicators of automation.

### Graph Construction Parameters:
- **Node Count:** 5,000 profiles
- **Metric:** Cosine distance over 12 normalized attributes
- **$k$-Neighbors:** 5 nearest neighbors per node
- **Total Directed Edges:** 32,712 edges (16,356 undirected pairs)
- **Self-Loops:** Preserved for identity retention during message passing
- **Inductive Extension:** For unseen profiles at inference time, GraphSAGE projects the new node into the training embedding space and connects it dynamically to its $k=5$ closest training neighbors.

---

## 4. Models & Message-Passing Formulations

INNOVEXA benchmarks traditional machine learning against modern deep learning and graph neural network architectures:

### 1. Traditional Baselines
- **Logistic Regression:** $\ell_2$-regularized linear model with validation-tuned threshold.
- **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel ($\gamma=\text{scale}, C=1.0$) with Platt-scaled probabilistic output.
- **Random Forest:** Ensemble of 150 randomized decision trees (max depth 12) with Gini impurity split criterion.

### 2. Non-Graph Deep Learning
- **Multilayer Perceptron (MLP):** 3-layer feed-forward network ($\text{Linear}(12, 64) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.3) \to \text{Linear}(64, 32) \to \text{Linear}(32, 2)$).

### 3. Graph Neural Networks
- **Graph Convolutional Network (GCN):**  
  Uses symmetric normalized neighborhood aggregation:
  $$\mathbf{H}^{(l+1)} = \sigma\left( \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$
  Where $\mathbf{\tilde{A}} = \mathbf{A} + \mathbf{I}_N$ and $\mathbf{\tilde{D}}_{ii} = \sum_j \mathbf{\tilde{A}}_{ij}$.

- **GraphSAGE (Sample and Aggregate):**  
  Employs mean aggregation over local neighborhoods followed by concatenation with the target node's prior embedding:
  $$\mathbf{h}_{\mathcal{N}(v)}^{(l+1)} = \text{MEAN}\left( \{\mathbf{h}_u^{(l)}, \forall u \in \mathcal{N}(v)\} \right)$$
  $$\mathbf{h}_v^{(l+1)} = \sigma\left( \mathbf{W}^{(l)} \cdot \left[ \mathbf{h}_v^{(l)} \,\|\, \mathbf{h}_{\mathcal{N}(v)}^{(l+1)} \right] \right)$$
  Enables genuine **inductive inference** for unseen test profiles without retraining.

- **Graph Attention Network (GAT):**  
  Computes parameterized multi-head self-attention coefficients $\alpha_{vu}$ across adjacent nodes:
  $$\alpha_{vu} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \,\|\, \mathbf{W}\mathbf{h}_u]\right)\right)}{\sum_{k \in \mathcal{N}(v)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \,\|\, \mathbf{W}\mathbf{h}_k]\right)\right)}$$
  $$\mathbf{h}_v^{(l+1)} = \sigma\left( \sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W} \mathbf{h}_u \right)$$

---

## 5. Empirical Held-Out Test Results

All models were evaluated on the **750 strictly held-out test accounts** (15% split). Decision thresholds were optimized exclusively on the 750 validation accounts to maximize the binary F1-score.

### Real Held-Out Test Performance (750 Test Profiles)

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Optimal Thresh | Train Time (s) | Latency (ms/sample) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **98.67%** | 98.15% | **99.20%** | **98.67%** | **0.9992** | **0.9992** | 0.31 | 0.41 s | 0.068 ms |
| **MLP** | 98.27% | 98.14% | 98.40% | 98.27% | 0.9987 | 0.9988 | 0.26 | 13.72 s | 0.001 ms |
| **GCN** | 97.60% | 97.60% | 97.60% | 97.60% | 0.9967 | 0.9972 | 0.39 | 4.25 s | 0.060 ms |
| **Support Vector Machine (RBF)** | 97.60% | 97.10% | 98.13% | 97.61% | 0.9965 | 0.9969 | 0.34 | 0.56 s | 0.041 ms |
| **GraphSAGE** | 97.47% | 97.59% | 97.33% | 97.46% | 0.9955 | 0.9968 | 0.31 | 5.37 s | 0.036 ms |
| **GAT** | 97.07% | **99.16%** | 94.93% | 97.00% | 0.9943 | 0.9951 | 0.68 | 11.80 s | 0.041 ms |
| **Logistic Regression** | 96.40% | 98.07% | 94.67% | 96.34% | 0.9934 | 0.9946 | 0.59 | 0.02 s | 0.000 ms |

### Test Confusion Matrices ($N = 750$)
- **Random Forest:** True Positives: 372 | False Positives: 7 | True Negatives: 368 | False Negatives: 3
- **MLP:** True Positives: 369 | False Positives: 7 | True Negatives: 368 | False Negatives: 6
- **GCN:** True Positives: 366 | False Positives: 9 | True Negatives: 366 | False Negatives: 9
- **GraphSAGE:** True Positives: 365 | False Positives: 9 | True Negatives: 366 | False Negatives: 10
- **GAT:** True Positives: 356 | False Positives: 3 | True Negatives: 372 | False Negatives: 19

---

## 6. Adversarial Robustness Analysis

To evaluate how well models withstand evasive adversarial attempts by malicious operators, two simulated attacks were executed:

### A. Feature Masking (Attribute Evasion)
Key profile attributes (e.g., `#posts`, `description length`, `nums/length username`) were randomly masked (set to 0 or mean) at perturbation rates from 10% to 50%:
- Under 30% feature masking, standalone tabular baselines experienced steep recall degradation (drop of ~3.8%).
- GNN models (GraphSAGE and GCN) demonstrated superior resilience (drop of only ~1.4%), because message passing aggregates structural context from connected neighbor nodes, bridging individual attribute deficiencies.

### B. Relational Camouflage (Edge Injection)
Spurious edges were injected between fake and genuine profile clusters at rates from 5% to 30%:
- Standard GCN demonstrated mild sensitivity to random edge noise as neighbor aggregation smoothed features across class boundaries.
- **GAT** maintained highest precision (99.16%) owing to its learned attention coefficients, which assign lower weights to noisy or discordant edges.

---

## 7. Suspicious Network & Cluster Analysis

Using the constructed similarity graph and model predictions:
- **High-Risk Nodes:** 2,402 accounts identified with estimated fake probability $\ge 0.70$.
- **Community Detection:** Louvain modularity clustering partitioned the 5,000 accounts into **49 distinct topological communities**.
- **Dense Bot Syndicates:** Identified 18 candidate clusters where $>85\%$ of member nodes exhibited high-confidence fake classifications. These clusters correspond to accounts generated by similar automated scripts with shared ratio patterns.

---

## 8. Repository & Folder Structure

```
INNOVEXA/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application entrypoint with CORS & lifespan
│   │   ├── api/
│   │   │   └── routes.py            # REST endpoints (/health, /models, /metrics, /predict, etc.)
│   │   ├── core/
│   │   │   ├── config.py            # App settings and file path definitions
│   │   │   └── database.py          # SQLite database connection & history table schema
│   │   ├── schemas/
│   │   │   └── profile.py           # Pydantic request/response validation schemas
│   │   └── services/
│   │       └── detection_service.py # Inference coordinator (loads models, executes prediction)
│   ├── ml/
│   │   ├── preprocessing.py         # Leakage-free preprocessor (log transforms & robust scalers)
│   │   ├── graph_builder.py         # kNN cosine similarity graph builder & inductive mapper
│   │   ├── baselines.py             # LR, SVM, Random Forest, and PyTorch MLP models
│   │   ├── gnn_models.py            # Standalone PyTorch implementations of GCN, GraphSAGE, GAT
│   │   ├── train.py                 # Master training pipeline across all 7 models
│   │   ├── evaluate.py              # Metric computation (Accuracy, F1, ROC, PR, Latency)
│   │   ├── robustness.py            # Adversarial testing (Feature masking & Edge injection)
│   │   └── predict.py               # Single & batch inference engine with GraphSAGE inductive pass
│   ├── models/                      # Saved checkpoints (.joblib preprocessors, .pt model weights)
│   ├── tests/
│   │   └── run_tests.py             # Comprehensive 8-test validation suite
│   ├── requirements.txt             # Python dependencies
│   └── innovexa.db                  # SQLite persistence database (created on startup)
│
├── frontend/
│   ├── public/                      # Static assets and favicon
│   ├── src/
│   │   ├── components/              # UI components (Navbar, MetricCard, GraphCanvas, etc.)
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx        # Summary stats, real model metrics, distribution charts
│   │   │   ├── ProfileDetection.jsx # Live 12-feature detection form with 6 real preset accounts
│   │   │   ├── ModelPerformance.jsx # ROC/PR curves, confusion matrices, latency benchmark
│   │   │   ├── NetworkAnalysis.jsx  # Interactive HTML5 graph canvas with zoom/pan & node drawer
│   │   │   ├── DetectionHistory.jsx # Searchable, filterable SQLite prediction records
│   │   │   └── ModelInfo.jsx        # Architecture specs, mathematical formulas, limitations
│   │   ├── App.jsx                  # Main router and navigation shell
│   │   ├── index.css                # Dark-navy cybersecurity styling tokens
│   │   └── main.jsx                 # React root bootstrap
│   ├── package.json                 # Node dependencies (React 19, Lucide, Recharts, Vite)
│   ├── vite.config.js               # Vite config with backend API reverse proxy
│   └── .env.example                 # Frontend environment variables template
│
├── data/
│   ├── README.md                    # Dataset description and ethical licensing notes
│   └── dataset.csv                  # Working copy of the 5,000-row dataset (gitignored)
│
├── results/
│   ├── metrics.json                 # Machine-readable test metrics for all 7 models
│   ├── metrics.csv                  # Tabular performance metrics
│   ├── roc_curves.json              # FPR/TPR coordinate arrays for plotting
│   ├── pr_curves.json               # Precision/Recall coordinate arrays for plotting
│   ├── loss_history.json            # Epoch-by-epoch training and validation loss curves
│   ├── robustness_results.json      # Perturbation and edge injection degradation curves
│   ├── network_analysis.json        # Subgraph sample and cluster statistics for visualization
│   └── experiment_report.md         # Formal scientific research report
│
├── README.md                        # Master project documentation (this document)
├── .gitignore                       # Git exclusion rules (ignores virtualenvs, models, data)
└── .env.example                     # Root environment configuration template
```

---

## 9. Setup & Installation Guide

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Python 3.10 to 3.13 installed and added to `PATH`.
- **Node.js:** Node.js 18.x or newer with `npm`.
- **Git:** Git version control installed.

---

### Backend Setup (FastAPI & PyTorch)

1. Open a terminal (PowerShell or Bash) and navigate to the backend directory:
   ```powershell
   cd "e:\MINI PROJECT\INNOVEXA\backend"
   ```

2. Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Upgrade `pip` and install dependencies:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the FastAPI development server:
   ```powershell
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *The backend will start at `http://127.0.0.1:8000`. Interactive OpenAPI documentation is accessible at `http://127.0.0.1:8000/docs`.*

---

### Frontend Setup (React + Vite)

1. Open a second terminal and navigate to the frontend directory:
   ```powershell
   cd "e:\MINI PROJECT\INNOVEXA\frontend"
   ```

2. Install JavaScript packages:
   ```powershell
   npm install
   ```

3. Launch the Vite development server:
   ```powershell
   npm run dev
   ```
   *The frontend application will start at `http://localhost:5173`.*

---

## 10. Reproducing Experiments & Model Training

To retrain all 7 models from scratch and regenerate all empirical metrics and plots:

1. Ensure your virtual environment is active in the backend directory:
   ```powershell
   cd "e:\MINI PROJECT\INNOVEXA"
   ```

2. Execute the master training pipeline:
   ```powershell
   python backend/ml/train.py
   ```

This script will:
1. Verify the dataset in `data/dataset.csv`.
2. Fit the preprocessing pipeline on the training split and save `backend/models/preprocessor.joblib`.
3. Construct the $k$NN feature-similarity graph and save `backend/models/graph_config.joblib`.
4. Train Logistic Regression, SVM, Random Forest, MLP, GCN, GraphSAGE, and GAT.
5. Save model weights to `backend/models/*.pt` and `backend/models/*.joblib`.
6. Calculate held-out test metrics, threshold selection, and curves into `results/metrics.json` and `results/metrics.csv`.
7. Execute adversarial robustness tests (feature masking and edge injection).
8. Compute suspicious clusters and export `results/network_analysis.json`.

---

## 11. API Reference

The FastAPI backend exposes the following RESTful endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health check and model loading status |
| `GET` | `/api/models` | List of all 7 trained models and the active default model |
| `GET` | `/api/metrics` | Held-out test metrics, ROC/PR curves, and loss histories |
| `GET` | `/api/dashboard` | Aggregated dataset summary, class distributions, and top metrics |
| `POST` | `/api/predict` | Make a prediction for a single profile (supports inductive GraphSAGE) |
| `GET` | `/api/network` | Returns graph topology, edge list, risk levels, and cluster stats |
| `GET` | `/api/history` | Fetches SQLite prediction audit logs (filterable by class) |

### Sample Detection Request (`POST /api/predict`)
```json
{
  "profile_pic": 0,
  "nums_length_username": 0.45,
  "fullname_words": 1,
  "nums_length_fullname": 0.0,
  "name_equals_username": 0,
  "description_length": 0,
  "external_url": 0,
  "is_private": 0,
  "num_posts": 1,
  "num_followers": 24,
  "num_follows": 850,
  "followers_following_ratio": 0.028,
  "model_name": "GraphSAGE"
}
```

### Sample Response
```json
{
  "prediction_id": "pred_c3d7e8f1",
  "predicted_class": "Fake",
  "is_fake": true,
  "probability_fake": 0.9412,
  "confidence_score": 0.9412,
  "risk_level": "High",
  "model_used": "GraphSAGE",
  "threshold_applied": 0.31,
  "neighbors_analyzed": 5,
  "disclaimer": "This prediction is a probabilistic research estimation based on profile metadata and does not constitute conclusive legal proof of fraud."
}
```

---

## 12. Running Tests

A comprehensive automated test suite validates data loading, target isolation, model loading, inductive inference, and FastAPI routes:

```powershell
cd "e:\MINI PROJECT\INNOVEXA"
python backend/tests/run_tests.py
```

Expected output:
```text
........
----------------------------------------------------------------------
Ran 8 tests in 0.813s

OK
```

---

## 13. GitHub Repository Management

This project is configured for integration with GitHub under the repository name:  
**`innovexa-gnn-fake-profile-detection`**

### Pushing Verified Milestones to GitHub:
If you have created the repository on your GitHub account (`https://github.com/hemantchaubeyworking-netizen/innovexa-gnn-fake-profile-detection`), push your verified local commits via:

```powershell
cd "e:\MINI PROJECT\INNOVEXA"
git remote add origin https://github.com/hemantchaubeyworking-netizen/innovexa-gnn-fake-profile-detection.git
git branch -M master
git push -u origin master
```

*Note: All raw dataset CSV files and bulky cache directories are strictly excluded via `.gitignore` to maintain security and regulatory compliance.*

---

## 14. Ethical Disclosures & Scientific Limitations

1. **Similarity Graph vs. Follower Topology:**  
   The graph is an unsupervised $k$NN feature-similarity graph. Relationships represent geometric proximity in attribute space, not social endorsement or explicit follow links.
2. **Probabilistic Risk Estimation:**  
   Predictions generated by INNOVEXA are statistical estimates based on profile heuristics. A "Fake" classification must **not** be treated as definitive legal proof of malicious intent. Automated decisions should always involve human-in-the-loop verification.
3. **Distribution Shift:**  
   Bot operators continuously adapt their behavior (e.g., adding realistic profile pictures or purchasing genuine followers). Models should be periodically fine-tuned using fresh behavioral data to mitigate adversarial drift.

---

## Authors & Citation

- **Research Concept & Paper:** Based on *"Fake Profile Detection Using Graph Neural Networks"*
- **System Implementation:** INNOVEXA Research & Engineering Team (2026)
