
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model_path = "heart_disease_model.pkl"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load dataset to get feature names
data_path = "heart_desease.csv"
df = pd.read_csv(data_path)
feature_names = df.columns[:-1]  # Assuming the last column is the target

# Streamlit UI
st.title("Heart Disease Prediction App")
st.write("Enter your health parameters to check the risk of heart disease.")

# User input fields
def user_input_features():
    inputs = []
    for feature in feature_names:
        value = st.number_input(f"Enter {feature}", min_value=0.0, step=0.1)
        inputs.append(value)
    return np.array([inputs])

input_data = user_input_features()

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "High Risk of Heart Disease" if prediction[0] == 1 else "Low Risk of Heart Disease"
    st.subheader("Prediction Result")
    st.write(result)