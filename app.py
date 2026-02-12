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

# ==========================================
# 1. CORE DEFINITIONS
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
def load_all_models():
    m = {}
    try:
        gnn = FraudGNN()
        gnn.load_state_dict(torch.load("gnn_model_sage.pth", map_location=torch.device('cpu')))
        gnn.eval()
        m['gnn'] = gnn
    except: m['gnn'] = None
    
    try:
        with open("fraud_model.pkl", "rb") as f:
            m['xgb'] = pickle.load(f)
    except: m['xgb'] = None
    return m

models = load_all_models()

# ==========================================
# 2. UI HEADER & INPUTS
# ==========================================
st.set_page_config(page_title="Crypto Fraud Sentinel", layout="wide")
st.title("🛡️ Advanced Hybrid Fraud Intelligence")

with st.sidebar:
    st.header("Transaction Parameters")
    tx_val = st.number_input("Transaction Value (ETH)", value=500.0)
    gas_fee = st.number_input("Gas Fee", value=0.005, format="%.4f")
    st.markdown("---")
    st.header("Network Topology")
    in_deg = st.number_input("In-Degree (Senders)", value=0)
    out_deg = st.number_input("Out-Degree (Receivers)", value=0)
    run_btn = st.button("Execute Deep Analysis", type="primary")

# ==========================================
# 3. DYNAMIC ANALYSIS ENGINE
# ==========================================
if run_btn:
    # 1. Initialize variables
    p_gnn, p_xgb = 0.0, 0.0
    # Determine the exact feature count your XGBoost model expects
    xgb_count = models['xgb'].n_features_in_ if models['xgb'] else 39
    
    # 2. Build Dynamic Graph (Reactive to Degrees)
    num_nodes = 1 + in_deg + out_deg
    gnn_features = np.zeros((num_nodes, 165))
    
    # Map Target Node (Node 0)
    gnn_features[0, 0] = tx_val / 5000.0 
    gnn_features[0, 1] = gas_fee / 0.1
    
    # Construct Edges dynamically
    sources, targets = [], []
    for i in range(1, in_deg + 1): # Senders
        sources.append(i); targets.append(0)
    for i in range(in_deg + 1, num_nodes): # Receivers
        sources.append(0); targets.append(i)

    edge_index = torch.tensor([sources, targets], dtype=torch.long) if sources else torch.tensor([[0],[0]], dtype=torch.long)

    # 3. GNN Inference
    if models['gnn']:
        with torch.no_grad():
            logits = models['gnn'](Data(x=torch.tensor(gnn_features, dtype=torch.float), edge_index=edge_index))
            p_gnn = torch.exp(logits)[0, 1].item()
    
    # 4. XGBoost Inference - DEFINING xgb_input HERE
    if models['xgb']:
        # We extract only the target node's features for the XGBoost expert
        xgb_input = gnn_features[0:1, :xgb_count]
        p_xgb = models['xgb'].predict_proba(xgb_input)[0, 1]

    blended_risk = (p_gnn * 0.5) + (p_xgb * 0.5)

    # ==========================================
    # 4. ENHANCED DASHBOARD DISPLAY
    # ==========================================
    m1, m2, m3 = st.columns(3)
    m1.metric("GNN Topology Risk", f"{p_gnn*100:.1f}%")
    m2.metric("XGBoost Local Risk", f"{p_xgb*100:.1f}%")
    # Using 'blended_risk' now defined above
    m3.metric("Aggregated Security Score", f"{(1-blended_risk)*100:.1f}%")

    st.divider()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Dynamic Network Topology")
        dot_code = 'digraph { rankdir=LR; node [style=filled, fontcolor=white, shape=circle];'
        dot_code += f'Target [color="#ff4b4b", label="Target\\nTx"];'
        
        # Dynamically add Senders
        for i in range(1, in_deg + 1):
            dot_code += f'S{i} [label="Sender {i}", color="#555"]; S{i} -> Target;'
            
        # Dynamically add Receivers
        for i in range(in_deg + 1, num_nodes):
            dot_code += f'R{i} [label="Receiver {i}", color="#555"]; Target -> R{i};'
            
        dot_code += '}'
        st.graphviz_chart(dot_code)

    with c2:
        st.subheader("📝 Explainable AI (Why?)")
        if models['xgb']:
            try:
                # Professional Labels
                names = [f"Other Proprietary_{i}" for i in range(xgb_count)]
                names[0], names[1], names[2] = "Transaction Value", "Gas Fee Density", "Network Centrality"
                
                df = pd.DataFrame(xgb_input, columns=names)
                explainer = shap.TreeExplainer(models['xgb'])
                shap_values = explainer(df) # Use the newer Explainer API for better plots

                plt.clf()
                # Create a waterfall-style bar plot to match your reference image
                # We show the top 10 features for clarity
                shap.plots.bar(shap_values[0], max_display=10, show=False)
                
                fig = plt.gcf()
                fig.set_size_inches(8, 5) # Match the aspect ratio of your image
                st.pyplot(fig, bbox_inches='tight')
                
                # Help box like your reference image
                st.info("""
                **How to read this graph:**
                * **Red Bars (+)**: These features increase the fraud risk score.
                * **Blue Bars (-)**: These features make the transaction look safer.
                * The length represents the strength of the feature's influence.
                """)
            except Exception as e:
                st.warning(f"SHAP Visualization error: {e}")