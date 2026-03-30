import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pickle
import shap
import matplotlib.pyplot as plt
import networkx as nx

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================
class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(2, 32)   # FIXED: now using REAL features (2 features)
        self.conv2 = SAGEConv(32, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


@st.cache_resource
def load_models():
    models = {}

    # Load GNN
    try:
        gnn = FraudGNN()
        gnn.load_state_dict(torch.load("gnn_model_sage.pth", map_location="cpu"))
        gnn.eval()
        models["gnn"] = gnn
    except:
        models["gnn"] = None

    # Load XGBoost
    try:
        with open("fraud_model.pkl", "rb") as f:
            models["xgb"] = pickle.load(f)
    except:
        models["xgb"] = None

    return models


models = load_models()

# ==========================================
# 2. PERSISTENT GRAPH
# ==========================================
if "graph" not in st.session_state:
    st.session_state.graph = nx.DiGraph()

G = st.session_state.graph

# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
def extract_features(G):
    pagerank = nx.pagerank(G) if len(G.nodes) > 0 else {}
    features = []

    for node in G.nodes:
        features.append([
            G.degree(node),                 # degree
            pagerank.get(node, 0.0)         # pagerank
        ])

    return torch.tensor(features, dtype=torch.float)


# ==========================================
# 4. BUILD GRAPH DATA FOR GNN
# ==========================================
def build_pyg_data(G):
    nodes = list(G.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges]

    if len(edges) == 0:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        edge_index = torch.tensor(edges).t().contiguous()

    x = extract_features(G)

    return Data(x=x, edge_index=edge_index), node_to_idx


# ==========================================
# 5. UI
# ==========================================
st.set_page_config(page_title="Crypto Fraud Sentinel", layout="wide")
st.title("🛡️ Real-Time Crypto Fraud Detection System")

with st.sidebar:
    st.header("Transaction Input")

    sender = st.text_input("Sender Wallet", "A")
    receiver = st.text_input("Receiver Wallet", "B")
    amount = st.number_input("Transaction Amount", value=100.0)

    run_btn = st.button("Analyze Transaction", type="primary")


# ==========================================
# 6. MAIN PIPELINE
# ==========================================
if run_btn:

    # Add transaction to graph
    G.add_edge(sender, receiver, weight=amount)

    # Build graph data
    data, node_to_idx = build_pyg_data(G)

    # Get target node index
    target_idx = node_to_idx[receiver]

    # ---------------- GNN ----------------
    p_gnn = 0.0
    if models["gnn"]:
        with torch.no_grad():
            logits = models["gnn"](data)
            probs = torch.exp(logits)
            p_gnn = probs[target_idx, 1].item()

    # ---------------- XGBoost ----------------
    p_xgb = 0.0
    if models["xgb"]:
        embedding = logits.detach().numpy()[target_idx]

        # combine embedding + simple feature
        xgb_input = np.concatenate([
            embedding,
            [G.degree(receiver)]
        ]).reshape(1, -1)

        try:
            p_xgb = models["xgb"].predict_proba(xgb_input)[0, 1]
        except:
            p_xgb = 0.0

    # ---------------- Final Score ----------------
    final_score = (p_gnn + p_xgb) / 2

    # ==========================================
    # 7. DISPLAY
    # ==========================================
    c1, c2, c3 = st.columns(3)

    c1.metric("GNN Risk", f"{p_gnn*100:.2f}%")
    c2.metric("XGBoost Risk", f"{p_xgb*100:.2f}%")
    c3.metric("Final Risk", f"{final_score*100:.2f}%")

    st.divider()

    # Graph visualization
    st.subheader("📊 Transaction Network")
    st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

    # ==========================================
    # 8. SHAP EXPLAINABILITY
    # ==========================================
    if models["xgb"]:
        try:
            explainer = shap.TreeExplainer(models["xgb"])

            feature_names = ["emb_1", "emb_2", "degree"]
            df = pd.DataFrame(xgb_input, columns=feature_names)

            shap_values = explainer(df)

            plt.clf()
            shap.plots.bar(shap_values[0], show=False)

            st.pyplot(plt.gcf())

            st.info("""
            🔴 Red → increases fraud risk  
            🔵 Blue → decreases fraud risk  
            """)

        except Exception as e:
            st.warning(f"SHAP error: {e}")