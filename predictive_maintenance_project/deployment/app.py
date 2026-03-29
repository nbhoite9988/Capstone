import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="nbhoite9988/predictive-maintenance", filename="best_predictive_maintenance_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for user input and prediction
st.title("Predictive Maintenance App")
st.write("""
This application predicts the likelihood of a vehicle engine failure(or Need for Maintenance) based on its parameters.
Please enter the data below to get a prediction.
""")

# User input
input_data = {
    'Engine rpm': st.number_input("Engine rpm", min_value=0, max_value=10000, value=790),
    'Lub oil pressure': st.number_input("Lub oil pressure", min_value=0, max_value=30, value=3),
    'Fuel pressure': st.number_input("Fuel pressure", min_value=0, max_value=30, value=7),
    'Coolant pressure': st.number_input("Coolant pressure", min_value=0, max_value=30, value=3),
    'Lub oil temp': st.number_input("Lub oil temp", min_value=0, max_value=150, value=75),
    'Coolant temp': st.number_input("Coolant temp", min_value=0, max_value=150, value=80),
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction
if st.button("Predict Engine Condition"):
    prediction = model.predict(input_df)[0]
    result = "Engine Failure or Need for Maintenance" if prediction == 1 else "Engine Healthy"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
