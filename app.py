# ✅ Streamlit app.py for Real-Time Fraud Detection (fixed version)

import streamlit as st
import joblib
import numpy as np

# Load saved model and both scalers
model = joblib.load(r'C:\Users\ADMIN\Desktop\fraud-detection\fraud_rf_model.pkl')
scaler_amount = joblib.load(r'C:\Users\ADMIN\Desktop\fraud-detection\scaler_amount.pkl')
scaler_time = joblib.load(r'C:\Users\ADMIN\Desktop\fraud-detection\scaler_time.pkl')

st.title("💳 Real-Time Fraud Detector")

# User inputs
amount = st.number_input("Transaction Amount:", value=0.0)
time = st.number_input("Transaction Time:", value=0.0)

# For demonstration, use dummy PCA features as zeros
other_features = [0] * 28  # V1, V2, ..., V28

if st.button("Predict"):
    # Scale time and amount separately
    time_scaled = scaler_time.transform([[time]])[0][0]
    amount_scaled = scaler_amount.transform([[amount]])[0][0]

    # Combine scaled features + dummy features
    input_array = np.array([[time_scaled] + other_features + [amount_scaled]])

    # Predict
    pred = model.predict(input_array)
    if pred[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Normal")
