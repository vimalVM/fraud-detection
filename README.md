# 🚀 Credit Card Fraud Detection

A simple machine learning project to detect fraudulent transactions using Random Forest, SMOTE, and an interactive Streamlit app.

✅ Detects fraudulent transactions using Machine Learning  
✅ Random Forest with SMOTE to handle imbalanced data  
✅ ROC AUC: ~0.90 → Good performance  
✅ Real-time prediction app using Streamlit

## How to run

1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Run notebook to see training: `FraudDetection.ipynb`
4. Launch Streamlit app: `streamlit run app.py`

## Files
- `FraudDetection.ipynb` → Model training
- `app.py` → Streamlit real-time predictor
- `*.pkl` → Saved model & scalers
