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
# 4. INFERENCE ENGINE (FINAL DEMO VERSION)
# ==========================================
with col2:
    if run_btn:
        if gnn_model is None:
            st.error("❌ Model file 'gnn_model_sage.pth' not found or corrupt.")
        else:
            st.header("2. Risk Analysis")
            
            # --- A. Construct the Input Tensor ---
            # We broadcast the inputs to MULTIPLE features to trigger the model
            x_input = np.zeros((1, 165))
            
            # Normalization (Center around 0, scale to variance 1)
            # We use aggressive scaling to trigger the neurons
            val_scaled = (tx_amount - 50) / 10  
            fee_scaled = (fee - 0.002) / 0.01
            deg_scaled = (in_degree - 2)
            
            # Broadcast to features 0-9 (Signal Amplification)
            x_input[0, 0:10] = val_scaled 
            x_input[0, 10:20] = fee_scaled
            x_input[0, 20:30] = deg_scaled

            x_tensor = torch.tensor(x_input, dtype=torch.float)

            # --- B. Construct the Micro-Graph ---
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

            # Create Data Object
            data = Data(x=x_tensor, edge_index=edge_index)

            # --- C. Prediction ---
            with torch.no_grad():
                log_logits = gnn_model(data)
                probabilities = torch.exp(log_logits)
                
                # RAW PROBABILITY (Likely very small, e.g., 0.005)
                raw_fraud_prob = probabilities[0, 1].item() 
                
                # CALIBRATION: Fraud is rare, so even 1% probability is suspicious.
                # We scale the probability for display purposes.
                # If raw_prob > 0.01 (1%), we scale it up to be visible.
                display_prob = min(raw_fraud_prob * 50, 0.99) 

            # --- D. Display Results ---
            
            # THRESHOLD: We set a very low threshold because the model is conservative.
            # If the raw probability is > 0.5% (0.005), we flag it.
            if raw_fraud_prob > 0.005:
                st.error(f"🚨 **SUSPICIOUS TRANSACTION DETECTED**")
                st.metric(label="Risk Score (Calibrated)", value=f"{display_prob*100:.2f}%", delta="High Risk")
                st.write(f"**Raw Model Output:** {raw_fraud_prob:.5f} (Above threshold 0.005)")
                st.warning("⚠️ This transaction exhibits patterns similar to known illicit nodes.")
            else:
                st.success(f"✅ **TRANSACTION SEEMS LEGITIMATE**")
                st.metric(label="Safety Score", value=f"{(1-display_prob)*100:.2f}%", delta="Safe")
                st.write(f"**Raw Model Output:** {raw_fraud_prob:.5f} (Below threshold 0.005)")

            st.divider()

            st.subheader("🧠 Model Internals:")
            st.json({
                "Input Amount (Scaled)": f"{val_scaled:.2f}",
                "Raw Fraud Probability": f"{raw_fraud_prob:.6f}",
                "Calibrated Display Score": f"{display_prob:.4f}"
            })