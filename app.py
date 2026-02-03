import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# 1. Load the Model and Features
model = joblib.load('fraud_model.pkl')
feature_names = joblib.load('model_features.pkl')

# 2. App Title & Description
st.title("🕵️‍♂️ Ethereum Fraud Detector System")
st.markdown("""
This system uses **XGBoost** and **Explainable AI (SHAP)** to detect fraudulent patterns in Ethereum transactions.
Adjust the transaction parameters below to test the model.
""")

# 3. Sidebar Inputs (Simulating a Transaction)
st.sidebar.header("Transaction Features")

def user_input_features():
    # We create inputs for the TOP 5 most important features
    # You can add more, but these 5 drive most decisions
    time_diff = st.sidebar.slider('Time Diff between first & last (Mins)', 0.0, 10000.0, 50.0)
    total_balance = st.sidebar.number_input('Total Ether Balance', min_value=0.0, value=0.0)
    min_val_rx = st.sidebar.number_input('Min Value Received', min_value=0.0, value=10.0)
    avg_val_rx = st.sidebar.number_input('Avg Value Received', min_value=0.0, value=5.0)
    total_erc20 = st.sidebar.number_input('Total ERC20 Tnxs', min_value=0, value=1)
    
    # Initialize a dictionary with all 0s for all features
    data = {col: 0 for col in feature_names}
    
    # Update the specific values we collected
    data['Time Diff between first and last (Mins)'] = time_diff
    data['total ether balance'] = total_balance
    data['min value received'] = min_val_rx
    data['avg val received'] = avg_val_rx
    # NEW (Correct)
    data[' Total ERC20 tnxs'] = total_erc20
    
    return pd.DataFrame([data])

input_df = user_input_features()

# 4. Display Input
st.subheader("Transaction Parameters")
st.write(input_df)

# 5. Prediction
if st.button("Analyze Transaction"):
    # Get Probability
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Display Result
    st.subheader("Risk Analysis Result")
    is_fraud = prediction[0] == 1
    
    if is_fraud:
        st.error(f"🚨 FRAUD DETECTED (Confidence: {probability[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ LEGITIMATE TRANSACTION (Confidence: {probability[0][0]*100:.2f}%)")

   # 6. xAI Explanation (SHAP)
    st.subheader("📝 Explainable AI (Why?)")
    with st.spinner('Calculating SHAP values...'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        # Create a specialized "Explanation" object for the waterfall plot
        # This handles the data formatting automatically
        explanation = shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=input_df.iloc[0], 
            feature_names=input_df.columns
        )
        
        # Create a clean figure
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Draw the Waterfall Plot
        shap.plots.waterfall(explanation, show=False)
        
        # Display in Streamlit
        st.pyplot(fig, bbox_inches='tight')
        
        st.info("⬇️ How to read this graph:\n"
                "* **E[f(x)]** is the average risk score.\n"
                "* **Red Bars (+)** mean this feature increases fraud risk.\n"
                "* **Blue Bars (-)** mean this feature makes it look safe.\n"
                "* **f(x)** is the final calculated risk score.")