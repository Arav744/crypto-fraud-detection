[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crypto-fraud-detection-xwc5e4mdbyjzk2kpyh59cp.streamlit.app/)
🚀 Crypto Fraud Detection System

A hybrid machine learning and deep learning system to detect fraudulent transactions in blockchain networks using Graph Neural Networks (GNN), XGBoost, and behavior-based anomaly detection.

📌 Overview

Cryptocurrency transactions are highly anonymous, making fraud detection challenging. Traditional models fail to capture relationships between wallets.

This project solves that by:

Modeling transactions as a graph
Combining graph learning + tabular ML
Adding behavior-based anomaly scoring
Providing real-time explainable predictions
🧠 Features
🔗 Graph-based learning (GraphSAGE GNN)
🌲 XGBoost for feature-based prediction
📊 Behavior modeling (volume, fan-out, fee patterns)
📈 Fraud propagation visualization
🔍 SHAP explainability (waterfall + feature impact)
⚡ Real-time prediction via Streamlit UI
🏗️ Architecture
Transaction Input
       ↓
Feature Extraction (165 features)
       ↓
 ┌───────────────┬───────────────┐
 │   GraphSAGE   │   XGBoost     │
 │   (GNN)       │   (Tabular)   │
 └───────────────┴───────────────┘
        ↓
 Behavior Modeling (Anomaly Score)
        ↓
   Ensemble Prediction
        ↓
 Explainability (SHAP) + Graph Visualization
📊 Dataset
Elliptic Dataset
~200K+ transactions (nodes)
~230K+ transaction flows (edges)
165 features per transaction
Classes: Fraud, Legit, Unknown
📈 Model Performance
Model	Accuracy	F1-Score
Logistic Regression	~58%	~0.58
MLP	~65%	~0.63
XGBoost	~78%	~0.75
GAT	~81%	~0.78
GraphSAGE	~83%	~0.80
Hybrid Model	~86%	~0.82–0.85

F1-score is prioritized due to class imbalance in fraud detection.

🖥️ Demo Features
Input custom transaction data
Predict fraud probability
Visualize transaction graph
Animate fraud propagation
Explain predictions using SHAP
⚙️ Tech Stack

Languages

Python

Libraries & Frameworks

PyTorch
PyTorch Geometric
XGBoost
Scikit-learn
SHAP
NetworkX
Streamlit
🚀 Installation
git clone https://github.com/Arav744/crypto-fraud-detection.git
cd crypto-fraud-detection

pip install -r requirements.txt
streamlit run app.py
📂 Project Structure
├── app.py
├── fraud_model.pkl
├── gnn_model_sage.pth
├── model_features.pkl
├── transactions.db
├── requirements.txt
└── README.md
⚠️ Deployment Notes
Streamlit Cloud may not fully support torch-geometric
For deployment:
Use CPU-compatible setup OR
Run without GNN (fallback to XGBoost)
🔍 Explainability
Uses SHAP to explain predictions
Waterfall plots show feature contributions
Helps understand why a transaction is flagged as fraud
⚠️ Limitations
Cold start problem for new wallets
GNN depends on graph size and connectivity
Dataset is static (real blockchain is dynamic)
🔮 Future Enhancements
Real-time blockchain integration
Temporal fraud detection
Cloud deployment with scalable backend
Advanced graph visualization
📎 GitHub

👉 https://github.com/Arav744/crypto-fraud-detection

👨‍💻 Author

Arav Goel
23BAI1269

⭐ If you like this project, give it a star!