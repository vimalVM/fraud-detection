# ✅ Streamlit app.py for Real-Time Fraud Detection (Mixed Stream with Final Count)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load saved model and both scalers
model = joblib.load('fraud_rf_model.pkl')
scaler_amount = joblib.load('scaler_amount.pkl')
scaler_time = joblib.load('scaler_time.pkl')

# Load dataset
df = pd.read_csv(r'D:\creditcard.csv')

st.markdown(
    """
    <h1 style='white-space: nowrap;'>💳 Real-Time Fraud Detector (Live Mixed Stream)</h1>
    """,
    unsafe_allow_html=True
)

# Button to start streaming
if st.button("▶️ Start Real-Time Transaction Stream"):
    st.info("Streaming 10 transactions (Fraud + Normal) ...")
    progress = st.empty()
    result = st.empty()

    # ✅ Pick 5 fraud + 5 normal rows → shuffle → realistic test
    fraud_rows = df[df['Class'] == 1].sample(5, random_state=42)
    normal_rows = df[df['Class'] == 0].sample(5, random_state=42)

    stream_rows = pd.concat([fraud_rows, normal_rows]).sample(frac=1).reset_index(drop=True)

    # Counters
    fraud_correct = 0
    normal_correct = 0

    for i, row in stream_rows.iterrows():
        time_val = row['Time']
        amount_val = row['Amount']

        # Scale time & amount
        time_scaled = scaler_time.transform([[time_val]])[0][0]
        amount_scaled = scaler_amount.transform([[amount_val]])[0][0]

        # Get other features (V1..V28)
        other_features = row.drop(['Time', 'Amount', 'Class']).values

        # Combine all: Time_scaled, V1..V28, Amount_scaled
        input_array = np.array([[time_scaled] + list(other_features) + [amount_scaled]])

        # Predict
        pred = model.predict(input_array)

        # Actual label
        true_label = int(row['Class'])

        # Update counters
        if pred[0] == true_label:
            if true_label == 1:
                fraud_correct += 1
            else:
                normal_correct += 1

        # Show result with ground truth (good for demo)
        if pred[0] == 1:
            result.error(f"🚨 Transaction {i+1}: FRAUD DETECTED! | True Label: {true_label}")
        else:
            result.success(f"✅ Transaction {i+1}: Normal | True Label: {true_label}")

        progress.text(f"Processed {i+1}/10 transactions...")
        time.sleep(2)

    st.success("✅ Done streaming mixed transactions!")

    # ✅ Final summary
    st.write("---")
    st.subheader("📊 Final Prediction Summary")
    st.write(f"✔️ Correctly predicted FRAUD: {fraud_correct} / 5")
    st.write(f"✔️ Correctly predicted NORMAL: {normal_correct} / 5")
    st.write(f"🏆 Overall Accuracy: {(fraud_correct + normal_correct)}/10")
