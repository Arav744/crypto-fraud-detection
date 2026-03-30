import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import pickle
import networkx as nx

# ==========================================
# 1. MODEL
# ==========================================
class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(165, 128)
        self.conv2 = SAGEConv(128, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


@st.cache_resource
def load_models():
    models = {}

    try:
        gnn = FraudGNN()
        gnn.load_state_dict(torch.load("gnn_model_sage.pth", map_location="cpu"))
        gnn.eval()
        models["gnn"] = gnn
    except:
        models["gnn"] = None

    try:
        with open("fraud_model.pkl", "rb") as f:
            models["xgb"] = pickle.load(f)
    except:
        models["xgb"] = None

    return models


models = load_models()

# ==========================================
# 2. INITIAL GRAPH (IMPORTANT)
# ==========================================
if "data" not in st.session_state:
    # Initialize empty graph
    st.session_state.data = Data(
        x=torch.zeros((1, 165), dtype=torch.float),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long)
    )

data = st.session_state.data

# ==========================================
# 3. UI
# ==========================================
st.set_page_config(page_title="Crypto Fraud Sentinel", layout="wide")
st.title("🛡️ Crypto Fraud Detection (Real GNN + 165 Features)")

with st.sidebar:
    st.header("Transaction Input")

    sender = st.text_input("Sender Wallet", "A")
    receiver = st.text_input("Receiver Wallet", "B")
    amount = st.number_input("Transaction Amount", value=100.0)
    fee = st.number_input("Gas Fee", value=0.001)

    run_btn = st.button("Analyze Transaction", type="primary")

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
if run_btn:

# -------- Add new node (transaction) --------

# ==========================================
# PERSISTENT STORAGE (history + graph)
# ==========================================
    if "history" not in st.session_state:
        st.session_state.history = {}

    if "nx_graph" not in st.session_state:
        st.session_state.nx_graph = nx.DiGraph()

    history = st.session_state.history
    G_nx = st.session_state.nx_graph

    # ==========================================
    # UPDATE GRAPH
    # ==========================================
    new_node_idx = data.x.shape[0]

    # Add node to NetworkX graph
    G_nx.add_node(new_node_idx)

    # Connect to previous node (simple realistic structure)
    if new_node_idx > 0:
        G_nx.add_edge(new_node_idx - 1, new_node_idx)
        G_nx.add_edge(new_node_idx, new_node_idx - 1)

    # ==========================================
    # UPDATE HISTORY
    # ==========================================
    if receiver not in history:
        history[receiver] = {
            "total_sent": 0,
            "total_received": 0,
            "count": 0
        }

    history[receiver]["total_received"] += amount
    history[receiver]["count"] += 1

    # ==========================================
    # COMPUTE GRAPH METRICS
    # ==========================================
    pagerank = nx.pagerank(G_nx) if len(G_nx.nodes) > 1 else {new_node_idx: 0}
    degree = G_nx.degree(new_node_idx)

    # ==========================================
    # BUILD 165 FEATURE VECTOR
    # ==========================================
    new_features = np.zeros(165)

    # 🔹 1. BASIC FEATURES
    new_features[0] = amount / 1000
    new_features[1] = fee / 0.01

    # 🔹 2. HISTORY FEATURES
    new_features[2] = history[receiver]["total_received"] / 10000
    new_features[3] = history[receiver]["count"]

    # 🔹 3. GRAPH FEATURES
    new_features[4] = degree
    new_features[5] = pagerank.get(new_node_idx, 0)

    # 🔹 4. DERIVED FEATURES
    if history[receiver]["count"] > 0:
        new_features[6] = history[receiver]["total_received"] / history[receiver]["count"]

    # (Remaining features stay 0 — unknown in live system)

    # Convert to tensor
    new_tensor = torch.tensor(new_features, dtype=torch.float).unsqueeze(0)

    # Append node to PyG graph
    data.x = torch.cat([data.x, new_tensor], dim=0)

    # ==========================================
    # EDGE INDEX UPDATE (PyTorch Geometric)
    # ==========================================
    if new_node_idx > 0:
        new_edges = torch.tensor([
            [new_node_idx - 1, new_node_idx],
            [new_node_idx, new_node_idx - 1]
        ], dtype=torch.long)

    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    new_tensor = torch.tensor(new_features, dtype=torch.float).unsqueeze(0)

    # Append node
    data.x = torch.cat([data.x, new_tensor], dim=0)

    new_node_idx = data.x.shape[0] - 1

    # -------- Add edges --------
    # connect to previous node (simple realistic structure)
    prev_node = new_node_idx - 1 if new_node_idx > 0 else 0

    new_edges = torch.tensor([
        [prev_node, new_node_idx],
        [new_node_idx, prev_node]
    ], dtype=torch.long)

    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    # -------- GNN Prediction --------
    p_gnn = 0.0
    logits = None

    if models["gnn"]:
        with torch.no_grad():
            logits = models["gnn"](data)
            probs = torch.exp(logits)
            p_gnn = probs[new_node_idx, 1].item()

    # -------- XGBoost Prediction --------
    p_xgb = 0.0

    if models["xgb"]:
        try:
            xgb_input = new_features.reshape(1, -1)
            p_xgb = models["xgb"].predict_proba(xgb_input)[0, 1]
        except:
            p_xgb = 0.0

    # -------- Final Score --------
    final_score = (p_gnn + p_xgb) / 2

    # ==========================================
    # 5. DISPLAY
    # ==========================================
    c1, c2, c3 = st.columns(3)

    c1.metric("GNN Risk", f"{p_gnn*100:.2f}%")
    c2.metric("XGBoost Risk", f"{p_xgb*100:.2f}%")
    c3.metric("Final Risk", f"{final_score*100:.2f}%")

    st.divider()

    # Show graph size
    st.write(f"Total Nodes: {data.x.shape[0]}")
    st.write(f"Total Edges: {data.edge_index.shape[1]}")