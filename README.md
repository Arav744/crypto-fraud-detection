[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://crypto-fraud-detection-xwc5e4mdbyjzk2kpyh59cp.streamlit.app/)
# 🕵️‍♂️ Ethereum Fraud Detection with Explainable AI

## 🚀 Project Overview
A machine learning system that detects fraudulent Ethereum transactions with **96% Accuracy**. Unlike standard "black box" models, this project integrates **SHAP (Explainable AI)** to provide transparency, showing exactly *why* a transaction was flagged (e.g., "Account created 5 mins ago").

## 🛠️ Tech Stack
* **Model:** XGBoost Classifier (Optimized with `scale_pos_weight`)
* **Validation:** 5-Fold Cross-Validation & ROC-AUC Analysis
* **Interpretability:** SHAP (SHapley Additive exPlanations)
* **App Interface:** Streamlit (Python)

## 📊 Key Results
* **Accuracy:** 95.94% (Test Set)
* **F1-Score:** 0.91
* **Top Indicator:** Time difference between first and last transaction.

## 💻 How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ethereum-fraud-detection.git](https://github.com/YOUR_USERNAME/ethereum-fraud-detection.git)

