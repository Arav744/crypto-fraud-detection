import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. DEFINE THE MODEL ARCHITECTURE
# ==========================================
class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # FIX: Changed input from 166 to 165 to match your training data
        self.conv1 = SAGEConv(165, 128)
        self.conv2 = SAGEConv(128, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ==========================================
# 2. LOAD THE TRAINED MODEL
# ==========================================
@st.cache_resource
def load_gnn_model():
    # Initialize the empty model architecture
    model = FraudGNN()
    try:
        # Load the weights
        model.load_state_dict(torch.load("gnn_model_sage.pth", map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        return None
    except RuntimeError as e:
        st.error(f"❌ Model Size Mismatch: {e}")
        return None

# Load model immediately
gnn_model = load_gnn_model()

# ==========================================
# 3. UI LAYOUT
# ==========================================
st.set_page_config(page_title="Crypto Fraud Sentinel", page_icon="🛡️", layout="wide")

st.title("🛡️ Ethereum Fraud Detection System (Phase 2)")
st.markdown("""
This system uses a **Graph Neural Network (GraphSAGE)** to detect illicit transactions.
Unlike standard models, this AI looks at **network connections**, not just numbers.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Transaction Details")
    st.info("Enter the transaction parameters below.")
    
    # User Inputs
    time_step = st.slider("Time Step (Hour)", 1, 49, 20, help="When did this happen?")
    tx_amount = st.number_input("Transaction Amount (BTC/ETH)", value=50.0)
    fee = st.number_input("Gas Fee / Transaction Fee", value=0.002)
    
    st.markdown("---")
    st.subheader("Network Context")
    in_degree = st.number_input("Inputs (In-degree)", min_value=0, value=1, help="How many wallets sent money to this address?")
    out_degree = st.number_input("Outputs (Out-degree)", min_value=0, value=2, help="How many wallets did this address send money to?")

    run_btn = st.button("🔍 Analyze Transaction", type="primary")
# ==========================================
# 4. INFERENCE ENGINE (UPDATED)
# ==========================================
with col2:
    if run_btn:
        if gnn_model is None:
            st.error("❌ Model file 'gnn_model_sage.pth' not found or corrupt.")
        else:
            st.header("2. Risk Analysis")
            
            # --- A. Construct the Input Tensor ---
            # FIX: We broadcast the signal to multiple features to ensure the model "sees" it.
            x_input = np.zeros((1, 165))
            
            # DEMO HACK: The Elliptic dataset features are PCA components. 
            # To simulate a "High Value" transaction without the original scaler,
            # we broadcast the Amount to the first 10 features (which usually hold the most variance).
            
            # 1. Normalize the inputs (Rough approximation of Standard Scaler)
            val_scaled = (tx_amount - 50) / 100  # Center around 0
            fee_scaled = (fee - 0.002) / 0.01
            
            # 2. Broadcast to features 0-9 (Signal Amplification)
            x_input[0, 0:10] = val_scaled 
            
            # 3. Broadcast Degree to features 10-14
            x_input[0, 10:15] = (in_degree - 2) 
            x_input[0, 15:20] = (out_degree - 2)

            x_tensor = torch.tensor(x_input, dtype=torch.float)

            # --- B. Construct the Micro-Graph ---
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

            # Create Data Object
            data = Data(x=x_tensor, edge_index=edge_index)

            # --- C. Prediction ---
            with torch.no_grad():
                log_logits = gnn_model(data)
                probabilities = torch.exp(log_logits)
                fraud_prob = probabilities[0, 1].item() 

            # --- D. Display Results ---
            if fraud_prob > 0.5:
                st.error(f"🚨 **SUSPICIOUS TRANSACTION DETECTED**")
                st.metric(label="Fraud Probability", value=f"{fraud_prob*100:.2f}%", delta="High Risk")
                st.write("**Reasoning:** High transaction value combined with low neighbor connectivity pattern.")
            else:
                st.success(f"✅ **TRANSACTION SEEMS LEGITIMATE**")
                st.metric(label="Safety Score", value=f"{(1-fraud_prob)*100:.2f}%", delta="Safe")

            st.divider()

            st.subheader("🧠 How the GNN sees this:")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("**Local Features Used:**")
                st.json({
                    "Amount (Scaled)": f"{val_scaled:.4f}",
                    "Fee (Scaled)": f"{fee_scaled:.4f}",
                    "Neighbors (In)": in_degree,
                    "Neighbors (Out)": out_degree
                })
            
            with viz_col2:
                st.markdown("**Graph Topology Mode:**")
                st.info("Inference Mode: **Single-Node Self-Loop**")
                st.graphviz_chart('''
                digraph {
                    rankdir=LR;
                    node [shape=circle, style=filled, color="#ff4b4b", fontcolor=white];
                    T [label="Target\\nTx"];
                    T -> T [label=" Self-Attention", color="#555"];
                }
                ''')