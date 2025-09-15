import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessing objects
model = joblib.load("svc_model.pkl")
scaler = joblib.load("scaler.pkl")

country_encoder = joblib.load("country_encoder.pkl")
used_app_encoder = joblib.load("used_app_encoder.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
jaundice_encoder = joblib.load("jaundice_encoder.pkl")
autism_encoder = joblib.load("autism_encoder.pkl")

st.title("Autism Screening Prediction App")

st.write("Enter the details below to predict the class")

# Example user inputs
A3_Score = st.number_input("A3 Score", min_value=0, max_value=1, step=1)
A4_Score = st.number_input("A4 Score", min_value=0, max_value=1, step=1)
A5_Score = st.number_input("A5 Score", min_value=0, max_value=1, step=1)
A6_Score = st.number_input("A6 Score", min_value=0, max_value=1, step=1)
A7_Score = st.number_input("A7 Score", min_value=0, max_value=1, step=1)
A8_Score = st.number_input("A8 Score", min_value=0, max_value=1, step=1)
A9_Score = st.number_input("A9 Score", min_value=0, max_value=1, step=1)
A10_Score = st.number_input("A10 Score", min_value=0, max_value=1, step=1)
age = st.number_input("Age", min_value=1, max_value=100, step=1)

gender = st.selectbox("Gender", ["male", "female"])
jaundice = st.selectbox("Jaundice (born with)", ["yes", "no"])
autism = st.selectbox("Family history of autism", ["yes", "no"])
used_app_before = st.selectbox("Used screening app before", ["yes", "no"])
country_of_res = st.text_input("Country of Residence")

# Make a dataframe from user input
input_data = pd.DataFrame({
    "A3_Score": [A3_Score],
    "A4_Score": [A4_Score],
    "A5_Score": [A5_Score],
    "A6_Score": [A6_Score],
    "A7_Score": [A7_Score],
    "A8_Score": [A8_Score],
    "A9_Score": [A9_Score],
    "A10_Score": [A10_Score],
    "age": [age],
    "gender": [gender],
    "jaundice": [jaundice],
    "autism": [autism],
    "used_app_before": [used_app_before],
    "country_of_res": [country_of_res]
})

# Apply encoders (must match training phase)
input_data["country_of_res"] = country_encoder.transform(input_data["country_of_res"])
input_data["used_app_before"] = used_app_encoder.transform(input_data["used_app_before"])
input_data["gender"] = gender_encoder.transform(input_data[["gender"]])
input_data["jaundice"] = jaundice_encoder.transform(input_data[["jaundice"]])
input_data["autism"] = autism_encoder.transform(input_data[["autism"]])

# Scale features
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted Class: {prediction}")
