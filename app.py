"""
Crypto Fraud Detection — Persistent Transaction Graph
=====================================================
Architecture:
  - SQLite stores every transaction submitted via the UI
  - NetworkX builds a live wallet-to-wallet graph from stored transactions
  - On each new submission the new node is added to the graph
  - Node features are engineered from the graph topology + transaction history
  - GraphSAGE GNN runs on the full graph and returns the SENDER node's fraud score
  - XGBoost model is used as a second opinion / fallback
  - Ensemble = weighted average (GNN weight grows with graph size)
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
_APP_DIR     = os.path.dirname(os.path.abspath(__file__))
DB_PATH      = os.path.join(_APP_DIR, "transactions.db")
GNN_FEATURES = 165
GNN_HIDDEN   = 128
GNN_CLASSES  = 2
GNN_MIN_NODES = 5   # minimum graph nodes before GNN activates

# ---------------------------------------------------------------------------
# 1. MODEL DEFINITION
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                tx_hash      TEXT UNIQUE,
                sender       TEXT NOT NULL,
                receiver     TEXT NOT NULL,
                amount       REAL NOT NULL,
                fee          REAL NOT NULL,
                time_step    INTEGER NOT NULL,
                in_degree    INTEGER NOT NULL,
                out_degree   INTEGER NOT NULL,
                submitted_at REAL NOT NULL,
                fraud_prob   REAL,
                xgb_prob     REAL,
                label        TEXT DEFAULT 'unknown'
            )
        """)
        conn.commit()


def insert_transaction(tx: dict) -> str:
    raw     = f"{tx['sender']}{tx['receiver']}{tx['amount']}{_time.time()}"
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
            pass
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
# 3. SYNTHETIC DATA SEEDER
# ---------------------------------------------------------------------------
def load_synthetic_data(xgb_model, xgb_features, gnn_model):
    """
    Insert pre-labelled synthetic transactions so the graph is pre-populated
    and fraud patterns exist before the user submits anything.
    Scores each transaction at insert time so the graph visualization is colored.
    """
    synthetic = [
        # (sender, receiver, amount, fee, time_step, in_deg, out_deg, known_label)
        ("wallet_A",   "wallet_B",  2.5,   0.0021, 12, 3, 2,  "legit"),
        ("wallet_C",   "wallet_D",  3.0,   0.0025, 13, 2, 2,  "legit"),
        ("wallet_E",   "wallet_F",  1.8,   0.0018, 14, 4, 1,  "legit"),
        ("wallet_K",   "wallet_L",  15.0,  0.005,  20, 1, 5,  "unknown"),
        ("wallet_L",   "wallet_M",  14.5,  0.0045, 21, 1, 6,  "unknown"),
        ("wallet_X",   "wallet_Y",  120.0, 0.0005, 30, 0, 10, "fraud"),
        ("wallet_Y",   "wallet_Z",  115.0, 0.0004, 31, 0, 9,  "fraud"),
        ("wallet_Z",   "wallet_AA", 110.0, 0.0003, 32, 0, 8,  "fraud"),
        ("wallet_bot", "wallet_1",  1.0,   0.0001, 15, 0, 8,  "fraud"),
        ("wallet_bot", "wallet_2",  1.1,   0.0001, 15, 0, 9,  "fraud"),
        ("wallet_bot", "wallet_3",  0.9,   0.0001, 15, 0, 7,  "fraud"),
    ]

    # Build the full graph from synthetic data first so scoring uses graph context
    df_temp = pd.DataFrame([{
        "sender": s, "receiver": r, "amount": amt, "fee": fee,
        "time_step": ts, "in_degree": ind, "out_degree": outd
    } for s, r, amt, fee, ts, ind, outd, _ in synthetic])

    G_temp = build_graph(df_temp)

    for s, r, amt, fee, ts, ind, outd, known_label in synthetic:
        xgb_prob  = None
        gnn_prob  = None
        ensemble  = None

        # Score with XGBoost
        if xgb_model is not None and xgb_features is not None:
            try:
                xgb_prob, _ = xgb_predict(
                    xgb_model, xgb_features,
                    wallet_node_features(s, G_temp, df_temp),
                    G_temp, df_temp, s
                )
            except Exception:
                pass

        # Score with GNN if graph big enough
        if gnn_model is not None and G_temp.number_of_nodes() >= GNN_MIN_NODES:
            try:
                data_pyg, tgt_idx = graph_to_pyg(G_temp, df_temp, s)
                with torch.no_grad():
                    log_logits = gnn_model(data_pyg)
                    probs      = torch.exp(log_logits)
                    gnn_prob   = probs[tgt_idx, 1].item()
            except Exception:
                pass

        ensemble = compute_ensemble(gnn_prob, xgb_prob, G_temp.number_of_nodes())

        # Use known label to override ensemble where we're certain
        if known_label == "fraud":
            final_label = "fraud"
        elif known_label == "legit":
            final_label = "legit"
        else:
            final_label = "fraud" if (ensemble or 0) > 0.5 else "legit"

        insert_transaction({
            "sender": s, "receiver": r, "amount": amt, "fee": fee,
            "time_step": ts, "in_degree": ind, "out_degree": outd,
            "fraud_prob": ensemble,
            "xgb_prob":   xgb_prob,
            "label":      final_label,
        })


# ---------------------------------------------------------------------------
# 4. GRAPH BUILDER
# ---------------------------------------------------------------------------
def build_graph(df: pd.DataFrame) -> nx.DiGraph:
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
    165-dimensional feature vector for a wallet node.
    Features 0–15: graph-derived, meaningful.
    Features 16–164: zeros (not random noise — deterministic and neutral).

    BUG FIX: removed np.random.normal noise injection — noise was seeded
    by hash(wallet) making predictions wallet-name-dependent and non-reproducible.
    """
    feat = np.zeros(GNN_FEATURES, dtype=np.float32)

    sent_rows = df[df["sender"]   == wallet]
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
    avg_fee    = sent_rows["fee"].mean()    if len(sent_rows) else 0.0

    n_nodes  = max(G.number_of_nodes(), 2)
    in_cent  = in_deg  / (n_nodes - 1)
    out_cent = out_deg / (n_nodes - 1)
    total_vol = total_sent + total_recv
    fanout    = out_deg / (in_deg + 1e-6)

    all_steps = pd.concat([sent_rows["time_step"], recv_rows["time_step"]])
    last_ts   = int(all_steps.max()) if len(all_steps) else 1

    feat[0]  = last_ts / 49.0
    feat[1]  = np.log1p(in_deg)  / 3.0
    feat[2]  = np.log1p(out_deg) / 3.0
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
    feat[14] = np.log1p(total_vol) / 10.0
    feat[15] = min(fanout / 5.0, 1.0)
    # feat[16:164] remain 0.0 — deterministic, model-neutral

    return feat


def graph_to_pyg(G: nx.DiGraph, df: pd.DataFrame,
                 target_wallet: str) -> tuple:
    nodes     = list(G.nodes())
    node_idx  = {n: i for i, n in enumerate(nodes)}
    n         = len(nodes)

    x_list = [wallet_node_features(w, G, df) for w in nodes]
    x      = torch.tensor(np.stack(x_list), dtype=torch.float)

    edges_src, edges_dst = [], []
    for (u, v) in G.edges():
        edges_src.append(node_idx[u])
        edges_dst.append(node_idx[v])

    # Self-loops for isolated nodes
    present = set(edges_src + edges_dst)
    for i in range(n):
        if i not in present:
            edges_src.append(i)
            edges_dst.append(i)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    data       = Data(x=x, edge_index=edge_index)
    target_idx = node_idx.get(target_wallet, 0)
    return data, target_idx


# ---------------------------------------------------------------------------
# 5. MODEL LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_gnn():
    model = FraudGNN()
    try:
        model.load_state_dict(torch.load("gnn_model_sage.pth", map_location="cpu"))
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
        model    = joblib.load("fraud_model.pkl")
        features = joblib.load("model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None


# ---------------------------------------------------------------------------
# 6. XGBoost INFERENCE
# ---------------------------------------------------------------------------
def xgb_predict(xgb_model, feature_names: list, feat_vec: np.ndarray,
                G: nx.DiGraph, df: pd.DataFrame, wallet: str) -> tuple:
    """
    Map computed features onto the exact column names in model_features.pkl.
    Uses list-of-tuples to avoid dict key collisions on 0.0 values.

    BUG FIX: time diff now computed from WALLET's own tx history, not global df.
    """
    sent_rows = df[df["sender"]   == wallet]
    recv_rows = df[df["receiver"] == wallet]

    def safe(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else 0.0
        except Exception:
            return 0.0

    # BUG FIX: use wallet-specific time range, not global df range
    wallet_steps = pd.concat([sent_rows["time_step"], recv_rows["time_step"]])
    wallet_time_diff = safe(
        (wallet_steps.max() - wallet_steps.min()) * 60
    ) if len(wallet_steps) > 1 else 0.0

    mappings = [
        # ── Time (wallet-scoped) ────────────────────────────────────────
        (wallet_time_diff,                                "Time Diff between first and last (Mins)"),
        (safe(sent_rows["time_step"].diff().mean() * 60) if len(sent_rows) > 1 else 0.0,
                                                          "Avg min between sent tnx"),
        (safe(recv_rows["time_step"].diff().mean() * 60) if len(recv_rows) > 1 else 0.0,
                                                          "Avg min between received tnx"),
        # ── Counts ─────────────────────────────────────────────────────
        (safe(len(sent_rows)),                            "Sent tnx"),
        (safe(len(recv_rows)),                            "Received Tnx"),
        (0.0,                                             "Number of Created Contracts"),
        (safe(recv_rows["sender"].nunique()),              "Unique Received From Addresses"),
        (safe(sent_rows["receiver"].nunique()),            "Unique Sent To Addresses"),
        # ── Received amounts ────────────────────────────────────────────
        (safe(recv_rows["amount"].min())  if len(recv_rows) else 0.0, "min value received"),
        (safe(recv_rows["amount"].max())  if len(recv_rows) else 0.0, "max value received "),  # trailing space exact
        (safe(recv_rows["amount"].mean()) if len(recv_rows) else 0.0, "avg val received"),
        # ── Sent amounts ────────────────────────────────────────────────
        (safe(sent_rows["amount"].min())  if len(sent_rows) else 0.0, "min val sent"),
        (safe(sent_rows["amount"].max())  if len(sent_rows) else 0.0, "max val sent"),
        (safe(sent_rows["amount"].mean()) if len(sent_rows) else 0.0, "avg val sent"),
        # ── Contract sent (no data) ─────────────────────────────────────
        (0.0, "min value sent to contract"),
        (0.0, "max val sent to contract"),
        (0.0, "avg value sent to contract"),
        # ── Totals ──────────────────────────────────────────────────────
        (safe(len(sent_rows) + len(recv_rows)),           "total transactions (including tnx to create contract"),
        (safe(sent_rows["amount"].sum()),                  "total Ether sent"),
        (safe(recv_rows["amount"].sum()),                  "total ether received"),
        (0.0,                                              "total ether sent contracts"),
        (safe(recv_rows["amount"].sum() - sent_rows["amount"].sum()), "total ether balance"),
        # ── ERC20 — all have leading space (exact from pkl) ─────────────
        (0.0, " Total ERC20 tnxs"),
        (0.0, " ERC20 total Ether received"),
        (0.0, " ERC20 total ether sent"),
        (0.0, " ERC20 total Ether sent contract"),
        (0.0, " ERC20 uniq sent addr"),
        (0.0, " ERC20 uniq rec addr"),
        (0.0, " ERC20 uniq sent addr.1"),
        (0.0, " ERC20 uniq rec contract addr"),
        (0.0, " ERC20 min val rec"),
        (0.0, " ERC20 max val rec"),
        (0.0, " ERC20 avg val rec"),
        (0.0, " ERC20 min val sent"),
        (0.0, " ERC20 max val sent"),
        (0.0, " ERC20 avg val sent"),
        (0.0, " ERC20 uniq sent token name"),
        (0.0, " ERC20 uniq rec token name"),
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
        "total_model_features": len(feature_names),
        "matched_columns":      len(matched_cols),
        "matched_names":        matched_cols,
        "ALL_model_features":   list(feature_names),
    }

    if len(matched_cols) == 0:
        return None, debug

    prob = xgb_model.predict_proba(row)[0, 1]
    return float(prob), debug


# ---------------------------------------------------------------------------
# 7. ENSEMBLE  (fixed — was broken if/elif chain)
# ---------------------------------------------------------------------------
def compute_ensemble(gnn_prob, xgb_prob, n_nodes: int):
    """
    BUG FIX: The original code had a bare 'if' (not 'elif') after the first
    block, causing the second block to ALWAYS run and overwrite the ensemble
    with xgb_prob alone — GNN contribution was silently discarded every time.

    Correct logic:
      - Both available → weighted average, GNN weight grows with graph size
      - Only one available → use it directly
      - Neither → None
    """
    if gnn_prob is not None and xgb_prob is not None:
        # GNN weight grows as the graph becomes more informative
        if n_nodes < 10:
            w_gnn = 0.15
        elif n_nodes < 20:
            w_gnn = 0.25
        else:
            w_gnn = 0.40
        return float(w_gnn * gnn_prob + (1.0 - w_gnn) * xgb_prob)
    elif xgb_prob is not None:
        return float(xgb_prob)
    elif gnn_prob is not None:
        return float(gnn_prob)
    return None


# ---------------------------------------------------------------------------
# 8. GRAPH VISUALISATION
# ---------------------------------------------------------------------------
def draw_graph(G: nx.DiGraph, df: pd.DataFrame,
               highlight_wallet=None, fraud_probs=None):
    if G.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    pos = nx.spring_layout(G, seed=42, k=2.5)

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

    edge_amounts = [np.log1p(d.get("amount", 1)) for _, _, d in G.edges(data=True)]
    max_a        = max(edge_amounts) if edge_amounts else 1
    edge_widths  = [max(0.5, 2.0 * a / max_a) for a in edge_amounts]

    if highlight_wallet and highlight_wallet in G.nodes():
        nx.draw_networkx_nodes(G, pos, nodelist=[highlight_wallet], ax=ax,
                               node_color="orange", node_size=700, alpha=0.4)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: n[:8] for n in G.nodes()},
                            font_size=7, font_color="white")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#AAAAAA",
                           width=edge_widths, arrows=True,
                           arrowstyle="->", arrowsize=12,
                           connectionstyle="arc3,rad=0.1")

    patches = [
        mpatches.Patch(color="#52C672", label="Safe (< 25%)"),
        mpatches.Patch(color="#E0A052", label="Suspicious (25–50%)"),
        mpatches.Patch(color="#E05252", label="Fraud (> 50%)"),
        mpatches.Patch(color="#4A90D9", label="Unscored"),
        mpatches.Patch(color="orange",  label="Newest tx"),
    ]
    ax.legend(handles=patches, loc="lower left",
              facecolor="#1E2130", edgecolor="#555",
              fontsize=7, labelcolor="white")
    ax.axis("off")
    ax.set_title("Transaction Graph", color="white", fontsize=11, pad=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 9. STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Fraud Sentinel",
    page_icon="🛡️",
    layout="wide"
)
init_db()

gnn_model          = load_gnn()
xgb_model, xgb_features = load_xgb()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Crypto Fraud Sentinel")
    st.caption("Persistent Transaction Graph + GNN + XGBoost")
    st.divider()

    df_all  = load_all_transactions()
    n_tx    = len(df_all)
    n_nodes = df_all[["sender", "receiver"]].stack().nunique() if n_tx else 0
    n_fraud = int((df_all["fraud_prob"] > 0.5).sum()) if n_tx else 0

    col_a, col_b = st.columns(2)
    col_a.metric("Transactions", n_tx)
    col_b.metric("Wallets",      n_nodes)
    col_a.metric("Flagged",      n_fraud)
    col_b.metric("GNN ready",    "✅" if gnn_model else "❌")

    st.divider()

    if st.button("⚡ Load Synthetic Dataset", use_container_width=True):
        load_synthetic_data(xgb_model, xgb_features, gnn_model)
        st.success("Synthetic data loaded and scored!")
        st.rerun()

    if st.button("🗑️ Clear all transactions", type="secondary", use_container_width=True):
        with get_db() as c:
            c.execute("DELETE FROM transactions")
            c.commit()
        st.cache_resource.clear()
        st.rerun()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_submit, tab_graph, tab_history = st.tabs([
    "➕  Submit Transaction",
    "🕸️  Transaction Graph",
    "📋  History",
])

# ============================================================
# TAB 1 — Submit & Predict
# ============================================================
with tab_submit:
    st.subheader("New Transaction")
    st.caption(
        "Each submission is stored permanently. "
        "The GNN runs on the **entire accumulated graph** — "
        "predictions improve as the network grows."
    )

    col1, col2 = st.columns(2)
    with col1:
        sender   = st.text_input("Sender wallet address",   value="wallet_A")
        receiver = st.text_input("Receiver wallet address", value="wallet_B")
        amount   = st.number_input("Amount (ETH / BTC)", value=1.0,
                                   min_value=0.0, format="%.4f")
        fee      = st.number_input("Gas / transaction fee", value=0.002,
                                   min_value=0.0, format="%.6f")
    with col2:
        time_step  = st.slider("Time step (1–49)", 1, 49, 20)
        in_degree  = st.number_input(
            "Sender in-degree (wallets that sent TO sender)", min_value=0, value=1)
        out_degree = st.number_input(
            "Sender out-degree (wallets sender sent TO)",     min_value=0, value=2)
        st.caption("ℹ️ In/out-degree are hints. "
                   "The graph auto-derives them from stored transactions.")

    analyze = st.button("🔍 Submit & Analyze", type="primary", use_container_width=True)

    if analyze:
        sender   = sender.strip()
        receiver = receiver.strip()

        if not sender or not receiver:
            st.error("Enter both sender and receiver wallet addresses.")
            st.stop()
        if sender == receiver:
            st.error("Self-transfers are flagged — sender and receiver must differ.")
            st.stop()

        # Build graph with new transaction included
        df_all      = load_all_transactions()
        new_row     = pd.DataFrame([{
            "sender": sender, "receiver": receiver,
            "amount": amount, "fee": fee, "time_step": time_step,
            "in_degree": in_degree, "out_degree": out_degree
        }])
        df_with_new = pd.concat([df_all, new_row], ignore_index=True)
        G           = build_graph(df_with_new)

        st.divider()
        st.subheader("📊 Risk Analysis")

        # ── GNN (scores the SENDER — the initiator of the transaction)  ───
        # BUG FIX: was scoring receiver.strip() — changed to sender
        gnn_prob    = None
        gnn_skipped = False

        if gnn_model is not None:
            if G.number_of_nodes() < GNN_MIN_NODES:
                gnn_skipped = True
            else:
                with st.spinner("Running GNN on transaction graph…"):
                    try:
                        data_pyg, tgt_idx = graph_to_pyg(G, df_with_new, sender)
                        with torch.no_grad():
                            log_logits = gnn_model(data_pyg)
                            probs      = torch.exp(log_logits)
                            gnn_prob   = probs[tgt_idx, 1].item()
                    except Exception as e:
                        st.warning(f"GNN inference failed: {e}")

        # ── XGBoost (primary model) ────────────────────────────────────────
        xgb_prob  = None
        xgb_debug = {}
        if xgb_model is not None:
            try:
                xgb_prob, xgb_debug = xgb_predict(
                    xgb_model, xgb_features,
                    wallet_node_features(sender, G, df_with_new),
                    G, df_with_new, sender
                )
                if xgb_prob is None:
                    st.warning(
                        "⚠️ XGBoost skipped — no column names matched. "
                        "Check the debug expander below."
                    )
            except Exception as e:
                st.warning(f"XGBoost inference failed: {e}")

        # ── Ensemble (fixed if/elif chain) ─────────────────────────────────
        ensemble = compute_ensemble(gnn_prob, xgb_prob, G.number_of_nodes())

        # ── Display scores ─────────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        with r1:
            if gnn_prob is not None:
                st.metric("GNN Score", f"{gnn_prob*100:.1f}%",
                          delta="🚨 High" if gnn_prob > 0.5 else "✅ Low",
                          delta_color="inverse")
            else:
                st.metric("GNN Score", "N/A")
        with r2:
            if xgb_prob is not None:
                st.metric("XGBoost Score", f"{xgb_prob*100:.1f}%",
                          delta="🚨 High" if xgb_prob > 0.5 else "✅ Low",
                          delta_color="inverse")
            else:
                st.metric("XGBoost Score", "N/A")
        with r3:
            if ensemble is not None:
                st.metric("Ensemble Score", f"{ensemble*100:.1f}%",
                          delta="🚨 High" if ensemble > 0.5 else "✅ Low",
                          delta_color="inverse")
            else:
                st.metric("Ensemble Score", "N/A")

        if gnn_skipped:
            st.info(
                f"ℹ️ GNN skipped — graph has {G.number_of_nodes()} nodes "
                f"(needs ≥ {GNN_MIN_NODES}). Using XGBoost only."
            )

        # ── Risk verdict (BUG FIX: restored the 🚨 fraud banner) ──────────
        if ensemble is not None:
            if ensemble > 0.5:
                st.error(
                    "🚨 **SUSPICIOUS TRANSACTION DETECTED** "
                    "— storing in graph for ongoing monitoring."
                )
            elif ensemble > 0.25:
                st.warning("⚠️ **ELEVATED RISK** — monitor this wallet.")
            else:
                st.success("✅ **LOW RISK** — transaction appears legitimate.")

        # ── Graph context ──────────────────────────────────────────────────
        with st.expander("🔍 Graph context for this prediction"):
            st.json({
                "graph_nodes":               G.number_of_nodes(),
                "graph_edges":               G.number_of_edges(),
                "sender_in_degree":          G.in_degree(sender),
                "sender_out_degree":         G.out_degree(sender),
                "sender_total_sent":         float(df_with_new[df_with_new["sender"]   == sender]["amount"].sum()),
                "sender_total_received":     float(df_with_new[df_with_new["receiver"] == sender]["amount"].sum()),
                "gnn_fraud_probability":     round(gnn_prob,  4) if gnn_prob  is not None else None,
                "xgboost_fraud_probability": round(xgb_prob,  4) if xgb_prob  is not None else None,
                "ensemble_fraud_probability":round(ensemble,  4) if ensemble  is not None else None,
            })

        # ── XGBoost debug expander ─────────────────────────────────────────
        if xgb_debug:
            n_matched = xgb_debug.get("matched_columns", 0)
            n_total   = xgb_debug.get("total_model_features", "?")
            with st.expander(f"🛠️ XGBoost feature debug ({n_matched}/{n_total} matched)"):
                st.json(xgb_debug)

        # ── Persist ────────────────────────────────────────────────────────
        tx_hash = insert_transaction({
            "sender": sender, "receiver": receiver,
            "amount": amount, "fee": fee,
            "time_step": time_step,
            "in_degree": in_degree, "out_degree": out_degree,
            "fraud_prob": ensemble,
            "xgb_prob":   xgb_prob,
            "label": "fraud" if (ensemble or 0) > 0.5 else "legit",
        })
        st.caption(f"Stored → hash `{tx_hash}`")

# ============================================================
# TAB 2 — Graph
# ============================================================
with tab_graph:
    df_all = load_all_transactions()

    if df_all.empty:
        st.info("No transactions yet. Submit one or load the synthetic dataset.")
    else:
        G = build_graph(df_all)

        fraud_probs = {}
        for _, row in df_all.iterrows():
            p = row["fraud_prob"]
            if p is not None:
                for w in [row["sender"], row["receiver"]]:
                    fraud_probs[w] = max(fraud_probs.get(w, 0.0), float(p))

        newest = df_all.iloc[-1]["sender"] if len(df_all) else None
        fig    = draw_graph(G, df_all, highlight_wallet=newest, fraud_probs=fraud_probs)
        if fig:
            st.pyplot(fig, use_container_width=True)

        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes (wallets)",    G.number_of_nodes())
        c2.metric("Edges (txns)",       G.number_of_edges())
        c3.metric("Avg degree",         f"{np.mean([d for _, d in G.degree()]):.1f}")
        c4.metric("Suspicious wallets", sum(1 for p in fraud_probs.values() if p > 0.5))

        with st.expander("Top 10 wallets by out-degree"):
            out_degs = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]
            df_deg   = pd.DataFrame(out_degs, columns=["wallet", "out_degree"])
            df_deg["fraud_prob"] = df_deg["wallet"].map(
                lambda w: round(fraud_probs.get(w, float("nan")), 3)
            )
            st.dataframe(df_deg, use_container_width=True)

# ============================================================
# TAB 3 — History
# ============================================================
with tab_history:
    df_all = load_all_transactions()

    if df_all.empty:
        st.info("No transactions recorded yet.")
    else:
        def risk_label(p):
            if p is None or (isinstance(p, float) and np.isnan(p)):
                return "—"
            if p > 0.5:  return "🚨 Fraud"
            if p > 0.25: return "⚠️ Suspicious"
            return "✅ Legit"

        disp = df_all[[
            "tx_hash", "sender", "receiver", "amount", "fee",
            "time_step", "fraud_prob", "xgb_prob", "label", "submitted_at"
        ]].copy()

        disp["risk"] = disp["fraud_prob"].apply(risk_label)
        disp["fraud_prob"] = disp["fraud_prob"].apply(
            lambda x: f"{x*100:.1f}%" if x is not None
                      and not (isinstance(x, float) and np.isnan(x)) else "—"
        )
        disp["xgb_prob"] = disp["xgb_prob"].apply(
            lambda x: f"{x*100:.1f}%" if x is not None
                      and not (isinstance(x, float) and np.isnan(x)) else "—"
        )
        disp["submitted_at"] = pd.to_datetime(
            disp["submitted_at"], unit="s"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            disp.rename(columns={
                "tx_hash": "Hash", "sender": "Sender", "receiver": "Receiver",
                "amount": "Amount", "fee": "Fee", "time_step": "Step",
                "fraud_prob": "Ensemble%", "xgb_prob": "XGB%",
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