"""
Crypto Fraud Detection — Persistent Transaction Graph
=====================================================
Architecture:
  - SQLite stores every transaction submitted via the UI
  - NetworkX builds a live wallet-to-wallet graph from stored transactions
  - On each new submission the new node is added to the graph
  - Node features are engineered from the graph topology + transaction history
  - GraphSAGE GNN runs on the full graph and returns the new node's fraud score
  - XGBoost model is used as a second opinion / fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import hashlib
import time as _time
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib

# ---------------------------------------------------------------------------
# 0. CONSTANTS
# ---------------------------------------------------------------------------
# Use an absolute path so the DB persists regardless of working directory.
# On Streamlit Cloud, use /tmp (survives restarts but not redeploys).
# Locally, store next to app.py so it always survives app restarts.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(_APP_DIR, "transactions.db")
GNN_FEATURES = 165   # must match training
GNN_HIDDEN   = 128
GNN_CLASSES  = 2

# ---------------------------------------------------------------------------
# 1. MODEL DEFINITION  (must match training exactly)
# ---------------------------------------------------------------------------
class FraudGNN(torch.nn.Module):
    def __init__(self, in_feats=GNN_FEATURES, hidden=GNN_HIDDEN, out=GNN_CLASSES):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden)
        self.conv2 = SAGEConv(hidden, out)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# 2. SQLITE BACKEND
# ---------------------------------------------------------------------------
def get_db():
    """Return a thread-local SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables on first run."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_hash     TEXT UNIQUE,
                sender      TEXT NOT NULL,
                receiver    TEXT NOT NULL,
                amount      REAL NOT NULL,
                fee         REAL NOT NULL,
                time_step   INTEGER NOT NULL,
                in_degree   INTEGER NOT NULL,
                out_degree  INTEGER NOT NULL,
                submitted_at REAL NOT NULL,
                fraud_prob  REAL,
                xgb_prob    REAL,
                label       TEXT DEFAULT 'unknown'
            )
        """)
        conn.commit()


def insert_transaction(tx: dict) -> str:
    """Insert a transaction and return its hash."""
    raw = f"{tx['sender']}{tx['receiver']}{tx['amount']}{_time.time()}"
    tx_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    with get_db() as conn:
        try:
            conn.execute("""
                INSERT INTO transactions
                  (tx_hash, sender, receiver, amount, fee, time_step,
                   in_degree, out_degree, submitted_at, fraud_prob, xgb_prob, label)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                tx_hash, tx["sender"], tx["receiver"],
                tx["amount"], tx["fee"], tx["time_step"],
                tx["in_degree"], tx["out_degree"], _time.time(),
                tx.get("fraud_prob"), tx.get("xgb_prob"),
                tx.get("label", "unknown")
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            pass   # duplicate hash — very unlikely
    return tx_hash


def load_all_transactions() -> pd.DataFrame:
    with get_db() as conn:
        return pd.read_sql("SELECT * FROM transactions ORDER BY submitted_at", conn)


def delete_transaction(tx_hash: str):
    with get_db() as conn:
        conn.execute("DELETE FROM transactions WHERE tx_hash = ?", (tx_hash,))
        conn.commit()


def update_fraud_prob(tx_hash: str, fraud_prob: float, xgb_prob: float):
    with get_db() as conn:
        conn.execute(
            "UPDATE transactions SET fraud_prob=?, xgb_prob=? WHERE tx_hash=?",
            (fraud_prob, xgb_prob, tx_hash)
        )
        conn.commit()


# ---------------------------------------------------------------------------
# 3. GRAPH BUILDER
# ---------------------------------------------------------------------------
def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed wallet graph from stored transactions.
    Nodes = wallet addresses
    Edges = transactions (with attributes: amount, fee, time_step)
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_node(row["sender"])
        G.add_node(row["receiver"])
        G.add_edge(
            row["sender"], row["receiver"],
            amount=row["amount"],
            fee=row["fee"],
            time_step=row["time_step"]
        )
    return G


def wallet_node_features(wallet: str, G: nx.DiGraph, df: pd.DataFrame) -> np.ndarray:
    """
    Engineer a 165-dimensional feature vector for a wallet node.

    Feature layout (matching Elliptic dataset spirit):
      [0]   = time_step (normalized)
      [1]   = in_degree
      [2]   = out_degree
      [3]   = total amount sent
      [4]   = total amount received
      [5]   = avg amount sent
      [6]   = avg amount received
      [7]   = max amount sent
      [8]   = max amount received
      [9]   = total fees paid
      [10]  = avg fee
      [11]  = degree centrality (in)
      [12]  = degree centrality (out)
      [13]  = self-loop flag (always 0 — no self-transfers expected)
      [14]  = log(total volume + 1)
      [15]  = ratio out/in degree (money fan-out ratio)
      [16..164] = zero-padded (edge-level aggregates from Elliptic
                  that we cannot reconstruct without raw data)
    """
    feat = np.zeros(GNN_FEATURES, dtype=np.float32)

    sent_rows = df[df["sender"] == wallet]
    recv_rows = df[df["receiver"] == wallet]

    in_deg  = G.in_degree(wallet)
    out_deg = G.out_degree(wallet)

    total_sent = sent_rows["amount"].sum()
    total_recv = recv_rows["amount"].sum()
    avg_sent   = sent_rows["amount"].mean() if len(sent_rows) else 0.0
    avg_recv   = recv_rows["amount"].mean() if len(recv_rows) else 0.0
    max_sent   = sent_rows["amount"].max()  if len(sent_rows) else 0.0
    max_recv   = recv_rows["amount"].max()  if len(recv_rows) else 0.0
    total_fees = sent_rows["fee"].sum()
    avg_fee    = sent_rows["fee"].mean() if len(sent_rows) else 0.0

    n_nodes = G.number_of_nodes() if G.number_of_nodes() > 1 else 2
    in_cent  = in_deg  / (n_nodes - 1)
    out_cent = out_deg / (n_nodes - 1)

    total_vol = total_sent + total_recv
    fanout    = out_deg / (in_deg + 1e-6)   # classic money-mule signal

    # Grab the latest time_step for this wallet
    all_steps = pd.concat([sent_rows["time_step"], recv_rows["time_step"]])
    last_ts = int(all_steps.max()) if len(all_steps) else 1

    # Normalize (values observed in Elliptic dataset range)
    feat[0]  = last_ts / 49.0
    feat[1]  = min(in_deg  / 10.0, 1.0)
    feat[2]  = min(out_deg / 10.0, 1.0)
    feat[3]  = np.log1p(total_sent) / 10.0
    feat[4]  = np.log1p(total_recv) / 10.0
    feat[5]  = np.log1p(avg_sent)   / 10.0
    feat[6]  = np.log1p(avg_recv)   / 10.0
    feat[7]  = np.log1p(max_sent)   / 10.0
    feat[8]  = np.log1p(max_recv)   / 10.0
    feat[9]  = np.log1p(total_fees) / 10.0
    feat[10] = np.log1p(avg_fee)    / 10.0
    feat[11] = float(in_cent)
    feat[12] = float(out_cent)
    feat[13] = 0.0
    feat[14] = np.log1p(total_vol)  / 10.0
    feat[15] = min(fanout / 5.0, 1.0)   # cap at 5x fan-out

    return feat


def graph_to_pyg(G: nx.DiGraph, df: pd.DataFrame,
                 target_wallet: str) -> tuple[Data, int]:
    """
    Convert a NetworkX graph to a PyTorch Geometric Data object.
    Returns (Data, node_index_of_target_wallet).
    Adds a dummy self-loop edge for isolated nodes so SAGEConv
    always has at least one edge to aggregate.
    """
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # Node feature matrix
    x_list = [wallet_node_features(wallet, G, df) for wallet in nodes]
    x = torch.tensor(np.stack(x_list), dtype=torch.float)

    # Edge list — also add self-loops for isolated nodes
    edges_src, edges_dst = [], []
    for (u, v) in G.edges():
        edges_src.append(node_idx[u])
        edges_dst.append(node_idx[v])

    # Guarantee every node has at least one edge (self-loop)
    present = set(edges_src + edges_dst)
    for i in range(n):
        if i not in present:
            edges_src.append(i)
            edges_dst.append(i)

    edge_index = torch.tensor(
        [edges_src, edges_dst], dtype=torch.long
    )

    data = Data(x=x, edge_index=edge_index)
    target_idx = node_idx.get(target_wallet, 0)
    return data, target_idx


# ---------------------------------------------------------------------------
# 4. MODEL LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_gnn() -> FraudGNN | None:
    model = FraudGNN()
    try:
        model.load_state_dict(
            torch.load("gnn_model_sage.pth", map_location="cpu")
        )
        model.eval()
        return model
    except FileNotFoundError:
        st.warning("⚠️ gnn_model_sage.pth not found — GNN predictions disabled.")
        return None
    except RuntimeError as e:
        st.warning(f"⚠️ GNN weight mismatch: {e}")
        return None


@st.cache_resource
def load_xgb():
    try:
        model = joblib.load("fraud_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None


# ---------------------------------------------------------------------------
# 5. XGBoost INFERENCE  (uses model_features.pkl to align columns)
# ---------------------------------------------------------------------------
def xgb_predict(xgb_model, feature_names: list, feat_vec: np.ndarray,
                G: nx.DiGraph, df: pd.DataFrame, wallet: str) -> tuple:
    """
    Build a feature row matching fraud_model.pkl's exact column names.
    Uses a list of (value, exact_name) pairs — NOT a dict — to avoid
    float-key collisions when multiple features are 0.0 simultaneously.
    Returns (probability, debug_dict) or (None, debug_dict).
    """
    sent_rows = df[df["sender"]   == wallet]
    recv_rows = df[df["receiver"] == wallet]

    def safe(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else 0.0
        except Exception:
            return 0.0

    # Exact column names from model_features.pkl — order matches ALL_model_features
    mappings = [
        # ── ETH transaction time features ──────────────────────────────
        (safe((df["time_step"].max() - df["time_step"].min()) * 60),
            "Time Diff between first and last (Mins)"),
        (safe(sent_rows["time_step"].diff().mean() * 60) if len(sent_rows) > 1 else 0.0,
            "Avg min between sent tnx"),
        (safe(recv_rows["time_step"].diff().mean() * 60) if len(recv_rows) > 1 else 0.0,
            "Avg min between received tnx"),
        # ── ETH transaction counts ──────────────────────────────────────
        (safe(len(sent_rows)),                          "Sent tnx"),
        (safe(len(recv_rows)),                          "Received Tnx"),
        (0.0,                                           "Number of Created Contracts"),
        (safe(recv_rows["sender"].nunique()),           "Unique Received From Addresses"),
        (safe(sent_rows["receiver"].nunique()),         "Unique Sent To Addresses"),
        # ── ETH received amounts ────────────────────────────────────────
        (safe(recv_rows["amount"].min()) if len(recv_rows) else 0.0,  "min value received"),
        (safe(recv_rows["amount"].max()) if len(recv_rows) else 0.0,  "max value received "),  # trailing space — exact
        (safe(recv_rows["amount"].mean()) if len(recv_rows) else 0.0, "avg val received"),
        # ── ETH sent amounts ────────────────────────────────────────────
        (safe(sent_rows["amount"].min()) if len(sent_rows) else 0.0,  "min val sent"),
        (safe(sent_rows["amount"].max()) if len(sent_rows) else 0.0,  "max val sent"),
        (safe(sent_rows["amount"].mean()) if len(sent_rows) else 0.0, "avg val sent"),
        # ── ETH sent-to-contract (no contract data — always 0) ──────────
        (0.0,  "min value sent to contract"),
        (0.0,  "max val sent to contract"),
        (0.0,  "avg value sent to contract"),
        # ── ETH totals ──────────────────────────────────────────────────
        (safe(len(sent_rows) + len(recv_rows)),
            "total transactions (including tnx to create contract"),   # exact — no closing paren
        (safe(sent_rows["amount"].sum()),               "total Ether sent"),
        (safe(recv_rows["amount"].sum()),               "total ether received"),
        (0.0,                                           "total ether sent contracts"),
        (safe(recv_rows["amount"].sum() - sent_rows["amount"].sum()), "total ether balance"),
        # ── ERC20 features — all have a leading space (exact from pkl) ──
        (0.0,  " Total ERC20 tnxs"),
        (0.0,  " ERC20 total Ether received"),
        (0.0,  " ERC20 total ether sent"),
        (0.0,  " ERC20 total Ether sent contract"),
        (0.0,  " ERC20 uniq sent addr"),
        (0.0,  " ERC20 uniq rec addr"),
        (0.0,  " ERC20 uniq sent addr.1"),
        (0.0,  " ERC20 uniq rec contract addr"),
        (0.0,  " ERC20 min val rec"),
        (0.0,  " ERC20 max val rec"),
        (0.0,  " ERC20 avg val rec"),
        (0.0,  " ERC20 min val sent"),
        (0.0,  " ERC20 max val sent"),
        (0.0,  " ERC20 avg val sent"),
        (0.0,  " ERC20 uniq sent token name"),
        (0.0,  " ERC20 uniq rec token name"),
    ]

    row = pd.DataFrame(
        np.zeros((1, len(feature_names)), dtype=np.float32),
        columns=feature_names
    )

    matched_cols = []
    for value, col_name in mappings:
        if col_name in row.columns:
            row[col_name] = value
            matched_cols.append(col_name)

    debug = {
        "total_model_features":  len(feature_names),
        "matched_columns":       len(matched_cols),
        "matched_names":         matched_cols,
        "ALL_model_features":    list(feature_names),   # full list for debugging
    }

    if len(matched_cols) == 0:
        return None, debug

    prob = xgb_model.predict_proba(row)[0, 1]
    return float(prob), debug


# ---------------------------------------------------------------------------
# 6. GRAPH VISUALISATION
# ---------------------------------------------------------------------------
def draw_graph(G: nx.DiGraph, df: pd.DataFrame,
               highlight_wallet: str | None = None,
               fraud_probs: dict | None = None):
    """
    Draw the transaction graph.
    - Node color encodes fraud probability (if available)
    - The highlighted wallet (newest) is outlined in orange
    - Edge thickness encodes transaction amount (log-scaled)
    """
    if G.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Node colors
    node_colors = []
    for node in G.nodes():
        p = (fraud_probs or {}).get(node, None)
        if p is None:
            node_colors.append("#4A90D9")
        elif p > 0.5:
            node_colors.append("#E05252")
        elif p > 0.25:
            node_colors.append("#E0A052")
        else:
            node_colors.append("#52C672")

    # Edge widths
    edge_amounts = []
    for (u, v, data) in G.edges(data=True):
        edge_amounts.append(np.log1p(data.get("amount", 1)))
    max_a = max(edge_amounts) if edge_amounts else 1
    edge_widths = [max(0.5, 2.0 * a / max_a) for a in edge_amounts]

    # Highlight ring for newest node
    if highlight_wallet and highlight_wallet in G.nodes():
        highlight_nodes = [highlight_wallet]
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, ax=ax,
                               node_color="orange", node_size=700, alpha=0.4)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: n[:6] for n in G.nodes()},
                            font_size=7, font_color="white")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#AAAAAA",
                           width=edge_widths, arrows=True,
                           arrowstyle="->", arrowsize=12,
                           connectionstyle="arc3,rad=0.1")

    # Legend
    patches = [
        mpatches.Patch(color="#52C672", label="Safe (< 25%)"),
        mpatches.Patch(color="#E0A052", label="Suspicious (25-50%)"),
        mpatches.Patch(color="#E05252", label="Fraud (> 50%)"),
        mpatches.Patch(color="#4A90D9", label="Unscored"),
        mpatches.Patch(color="orange",  label="New transaction"),
    ]
    ax.legend(handles=patches, loc="lower left",
              facecolor="#1E2130", edgecolor="#555", fontsize=7,
              labelcolor="white")
    ax.axis("off")
    ax.set_title("Transaction Graph", color="white", fontsize=11, pad=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Fraud Sentinel",
    page_icon="🛡️",
    layout="wide"
)
init_db()

gnn_model = load_gnn()
xgb_model, xgb_features = load_xgb()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Crypto Fraud Sentinel")
    st.caption("Persistent Transaction Graph + GNN + XGBoost")
    st.divider()

    df_all = load_all_transactions()
    n_tx   = len(df_all)
    n_nodes = df_all[["sender","receiver"]].stack().nunique() if n_tx else 0
    n_fraud = (df_all["fraud_prob"] > 0.5).sum() if n_tx else 0

    col_a, col_b = st.columns(2)
    col_a.metric("Transactions", n_tx)
    col_b.metric("Wallets", n_nodes)
    col_a.metric("Flagged", int(n_fraud))
    col_b.metric("GNN ready", "✅" if gnn_model else "❌")

    st.divider()
    if st.button("🗑️ Clear all transactions", type="secondary"):
        with get_db() as c:
            c.execute("DELETE FROM transactions")
            c.commit()
        st.cache_resource.clear()
        st.rerun()

# ── Main tabs ────────────────────────────────────────────────────────────────
tab_submit, tab_graph, tab_history = st.tabs([
    "➕  Submit Transaction",
    "🕸️  Transaction Graph",
    "📋  History"
])

# ============================================================
# TAB 1: Submit + Predict
# ============================================================
with tab_submit:
    st.subheader("New Transaction")
    st.caption(
        "Each submission is stored permanently. The GNN runs on the **entire "
        "accumulated graph**, so predictions improve as the network grows."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        sender   = st.text_input("Sender wallet address",   value="wallet_A",
                                 help="Use any identifier — address, alias, etc.")
        receiver = st.text_input("Receiver wallet address", value="wallet_B")
        amount   = st.number_input("Amount (ETH / BTC)", value=1.0, min_value=0.0,
                                   format="%.4f")
        fee      = st.number_input("Gas / transaction fee", value=0.002,
                                   min_value=0.0, format="%.6f")

    with col2:
        time_step  = st.slider("Time step (1–49)", 1, 49, 20,
                               help="Hour bucket from the Elliptic dataset scale")
        in_degree  = st.number_input("Sender in-degree (how many wallets sent TO sender)",
                                     min_value=0, value=1)
        out_degree = st.number_input("Sender out-degree (how many wallets sender sent TO)",
                                     min_value=0, value=2)

        st.caption("ℹ️ In/out-degree are optional hints. The graph derives them "
                   "automatically from accumulated transactions.")

    analyze = st.button("🔍 Submit & Analyze", type="primary", use_container_width=True)

    if analyze:
        if not sender.strip() or not receiver.strip():
            st.error("Please enter both sender and receiver wallet addresses.")
            st.stop()

        if sender.strip() == receiver.strip():
            st.error("Sender and receiver cannot be the same address "
                     "(self-transfers are a common fraud signal — flagging manually).")
            st.stop()

        # ── Build current graph including new transaction ──────────────────
        df_all = load_all_transactions()
        new_row = pd.DataFrame([{
            "sender": sender.strip(), "receiver": receiver.strip(),
            "amount": amount, "fee": fee,
            "time_step": time_step,
            "in_degree": in_degree, "out_degree": out_degree
        }])
        df_with_new = pd.concat([df_all, new_row], ignore_index=True)

        G = build_graph(df_with_new)

        st.divider()
        st.subheader("📊 Risk Analysis")

        # ── GNN prediction ────────────────────────────────────────────────
        # IMPORTANT: GNN is only meaningful when the graph has enough nodes
        # to aggregate real neighbourhood information.  With < 10 nodes the
        # feature distribution is so far from the Elliptic training set that
        # outputs are unreliable → we skip it and rely on XGBoost instead.
        GNN_MIN_NODES = 10
        gnn_prob      = None
        gnn_skipped   = False

        if gnn_model is not None:
            if G.number_of_nodes() < GNN_MIN_NODES:
                gnn_skipped = True
            else:
                with st.spinner("Running GNN on the transaction graph…"):
                    try:
                        data_pyg, tgt_idx = graph_to_pyg(G, df_with_new, sender.strip())
                        with torch.no_grad():
                            log_logits = gnn_model(data_pyg)
                            probs      = torch.exp(log_logits)
                            gnn_prob   = probs[tgt_idx, 1].item()
                    except Exception as e:
                        st.warning(f"GNN inference failed: {e}")

        # ── XGBoost prediction ────────────────────────────────────────────
        # XGBoost is the PRIMARY model — it works well even on small graphs
        # because it operates on per-node features, not graph structure.
        xgb_prob  = None
        xgb_debug = {}
        if xgb_model is not None:
            try:
                xgb_prob, xgb_debug = xgb_predict(
                    xgb_model, xgb_features,
                    wallet_node_features(sender.strip(), G, df_with_new),
                    G, df_with_new, sender.strip()
                )
                if xgb_prob is None:
                    st.warning(
                        "⚠️ XGBoost skipped — none of the model's feature "
                        "column names matched our computed features. "
                        "Check the debug expander below to see the actual names."
                    )
            except Exception as e:
                st.warning(f"XGBoost inference failed: {e}")

        # ── Ensemble: weight XGBoost 70 %, GNN 30 % once graph is big enough
        # When GNN is skipped, use XGBoost only.
        if gnn_prob is not None and xgb_prob is not None:
            ensemble = 0.3 * gnn_prob + 0.7 * xgb_prob
        elif xgb_prob is not None:
            ensemble = xgb_prob
        elif gnn_prob is not None:
            ensemble = gnn_prob
        else:
            ensemble = None

        # ── Display results ───────────────────────────────────────────────
        res_cols = st.columns(3)
        with res_cols[0]:
            if gnn_prob is not None:
                st.metric("GNN Score", f"{gnn_prob*100:.1f}%",
                          delta="🚨 High" if gnn_prob > 0.5 else "✅ Low",
                          delta_color="inverse")
            else:
                st.metric("GNN Score", "N/A")

        with res_cols[1]:
            if xgb_prob is not None:
                st.metric("XGBoost Score", f"{xgb_prob*100:.1f}%",
                          delta="🚨 High" if xgb_prob > 0.5 else "✅ Low",
                          delta_color="inverse")
            else:
                st.metric("XGBoost Score", "N/A")

        with res_cols[2]:
            if ensemble is not None:
                st.metric("Ensemble Score", f"{ensemble*100:.1f}%",
                          delta="🚨 High" if ensemble > 0.5 else "✅ Low",
                          delta_color="inverse")
            else:
                st.metric("Ensemble Score", "N/A")

        if gnn_skipped:
            st.info(
                f"ℹ️ GNN skipped — graph has only {G.number_of_nodes()} nodes "
                f"(needs ≥ {GNN_MIN_NODES}). Add more transactions to activate it. "
                "Using XGBoost only for now."
            )

        if ensemble is not None:
            if ensemble > 0.5:
                st.error("🚨 **SUSPICIOUS TRANSACTION DETECTED**  "
                         "— Storing in graph for ongoing monitoring.")
            elif ensemble > 0.25:
                st.warning("⚠️ **ELEVATED RISK** — Monitor this wallet.")
            else:
                st.success("✅ **LOW RISK** — Transaction appears legitimate.")

        # Graph stats
        with st.expander("🔍 Graph context for this prediction"):
            sender_in_deg  = G.in_degree(sender.strip())
            sender_out_deg = G.out_degree(sender.strip())
            st.json({
                "graph_nodes":             G.number_of_nodes(),
                "graph_edges":             G.number_of_edges(),
                "sender_in_degree":        sender_in_deg,
                "sender_out_degree":       sender_out_deg,
                "sender_total_sent":       float(df_with_new[df_with_new["sender"] == sender.strip()]["amount"].sum()),
                "sender_total_received":   float(df_with_new[df_with_new["receiver"] == sender.strip()]["amount"].sum()),
                "gnn_fraud_probability":   round(gnn_prob, 4) if gnn_prob is not None else None,
                "xgboost_fraud_probability": round(xgb_prob, 4) if xgb_prob is not None else None,
                "ensemble_fraud_probability": round(ensemble, 4) if ensemble is not None else None,
            })

        # ── CRITICAL DEBUG: show actual XGBoost feature names ─────────────
        if xgb_debug:
            with st.expander(
                "🛠️ XGBoost feature debug "
                f"({xgb_debug.get('matched_columns', 0)}"
                f"/{xgb_debug.get('total_model_features', '?')} columns matched) "
                "— paste ALL_model_features here if < 38"
            ):
                st.caption(
                    "If matched_columns = 0, copy the feature names below "
                    "and tell me — I'll fix the mapping immediately."
                )
                st.json(xgb_debug)

        # ── Persist to database ────────────────────────────────────────────
        tx_hash = insert_transaction({
            "sender": sender.strip(), "receiver": receiver.strip(),
            "amount": amount, "fee": fee,
            "time_step": time_step,
            "in_degree": in_degree, "out_degree": out_degree,
            "fraud_prob": ensemble,
            "xgb_prob":   xgb_prob,
            "label": "fraud" if (ensemble or 0) > 0.5 else "legit"
        })
        st.caption(f"Transaction stored → hash `{tx_hash}`")

# ============================================================
# TAB 2: Graph Visualisation
# ============================================================
with tab_graph:
    df_all = load_all_transactions()

    if df_all.empty:
        st.info("No transactions yet. Submit one from the first tab to see the graph.")
    else:
        G = build_graph(df_all)

        # Build fraud_probs dict: wallet → max fraud_prob across its transactions
        fraud_probs = {}
        for _, row in df_all.iterrows():
            p = row["fraud_prob"]
            if p is not None:
                for wallet in [row["sender"], row["receiver"]]:
                    fraud_probs[wallet] = max(fraud_probs.get(wallet, 0.0), p)

        # Most recent transaction's sender
        newest = df_all.iloc[-1]["sender"] if len(df_all) else None

        fig = draw_graph(G, df_all, highlight_wallet=newest, fraud_probs=fraud_probs)
        if fig:
            st.pyplot(fig, use_container_width=True)

        st.divider()
        gcol1, gcol2, gcol3, gcol4 = st.columns(4)
        gcol1.metric("Nodes (wallets)", G.number_of_nodes())
        gcol2.metric("Edges (transactions)", G.number_of_edges())
        gcol3.metric("Avg degree", f"{np.mean([d for _, d in G.degree()]):.1f}")
        gcol4.metric(
            "Suspicious wallets",
            sum(1 for p in fraud_probs.values() if p > 0.5)
        )

        with st.expander("Top 10 wallets by out-degree (highest fan-out — money mule signal)"):
            out_degs = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]
            df_deg   = pd.DataFrame(out_degs, columns=["wallet", "out_degree"])
            df_deg["fraud_prob"] = df_deg["wallet"].map(
                lambda w: round(fraud_probs.get(w, float("nan")), 3)
            )
            st.dataframe(df_deg, use_container_width=True)

# ============================================================
# TAB 3: History
# ============================================================
with tab_history:
    df_all = load_all_transactions()

    if df_all.empty:
        st.info("No transactions recorded yet.")
    else:
        # Color code risk level
        def risk_label(p):
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return "—"
            if p > 0.5:   return "🚨 Fraud"
            if p > 0.25:  return "⚠️ Suspicious"
            return "✅ Legit"

        display_df = df_all[[
            "tx_hash","sender","receiver","amount","fee",
            "time_step","fraud_prob","xgb_prob","label","submitted_at"
        ]].copy()
        display_df["risk"]         = display_df["fraud_prob"].apply(risk_label)
        display_df["fraud_prob"]   = display_df["fraud_prob"].apply(
            lambda x: f"{x*100:.1f}%" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"
        )
        display_df["xgb_prob"]     = display_df["xgb_prob"].apply(
            lambda x: f"{x*100:.1f}%" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"
        )
        display_df["submitted_at"] = pd.to_datetime(
            display_df["submitted_at"], unit="s"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            display_df.rename(columns={
                "tx_hash": "Hash", "sender": "Sender", "receiver": "Receiver",
                "amount": "Amount", "fee": "Fee", "time_step": "Step",
                "fraud_prob": "GNN Fraud%", "xgb_prob": "XGB Fraud%",
                "label": "Label", "submitted_at": "Submitted", "risk": "Risk"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.subheader("Delete a transaction")
        del_hash = st.text_input("Enter tx hash to delete")
        if st.button("Delete", type="secondary") and del_hash:
            delete_transaction(del_hash.strip())
            st.success(f"Deleted {del_hash}")
            st.rerun()