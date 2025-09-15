import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model and preprocessing objects
# -----------------------------
model = joblib.load("svc_model.pkl")
scaler = joblib.load("scaler.pkl")

country_encoder = joblib.load("country_encoder.pkl")   # LabelEncoder
used_app_encoder = joblib.load("used_app_encoder.pkl") # LabelEncoder
gender_encoder = joblib.load("gender_encoder.pkl")     # OneHotEncoder
jaundice_encoder = joblib.load("jaundice_encoder.pkl") # OneHotEncoder
autism_encoder = joblib.load("autism_encoder.pkl")     # OneHotEncoder

st.title("Autism Screening Prediction App")
st.write("Select the details below to predict the autism screening class")

# -----------------------------
# User Inputs (A1 → A10 using selectbox)
# -----------------------------
options = [0, 1]  # Score options

A1_Score = st.selectbox("A1 Score", options)
A2_Score = st.selectbox("A2 Score", options)
A3_Score = st.selectbox("A3 Score", options)
A4_Score = st.selectbox("A4 Score", options)
A5_Score = st.selectbox("A5 Score", options)
A6_Score = st.selectbox("A6 Score", options)
A7_Score = st.selectbox("A7 Score", options)
A8_Score = st.selectbox("A8 Score", options)
A9_Score = st.selectbox("A9 Score", options)
A10_Score = st.selectbox("A10 Score", options)

age = st.selectbox("Age", list(range(1, 101)))
gender = st.selectbox("Gender", ["male", "female"])
jaundice = st.selectbox("Jaundice (born with)", ["yes", "no"])
autism = st.selectbox("Family history of autism", ["yes", "no"])
used_app_before = st.selectbox("Used screening app before", ["yes", "no"])
country_of_res = st.text_input("Country of Residence", value="India")

# -----------------------------
# Create input dataframe
# -----------------------------
input_data = pd.DataFrame({
    "A1_Score": [A1_Score],
    "A2_Score": [A2_Score],
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

# -----------------------------
# Safe transform for LabelEncoder
# -----------------------------
def safe_label_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        return 0  # default to 0 if unseen

input_data["country_of_res"] = input_data["country_of_res"].apply(lambda x: safe_label_transform(country_encoder, x))
input_data["used_app_before"] = input_data["used_app_before"].apply(lambda x: safe_label_transform(used_app_encoder, x))

# -----------------------------
# Safe transform for OneHotEncoder
# -----------------------------
def safe_ohe_transform(encoder, df, column):
    try:
        ohe = encoder.transform(df[[column]]).toarray()
    except ValueError:
        # if unseen category, create zeros
        ohe = np.zeros((df.shape[0], len(encoder.get_feature_names_out([column]))))
    return pd.DataFrame(ohe, columns=encoder.get_feature_names_out([column]))

gender_df = safe_ohe_transform(gender_encoder, input_data, "gender")
jaundice_df = safe_ohe_transform(jaundice_encoder, input_data, "jaundice")
autism_df = safe_ohe_transform(autism_encoder, input_data, "autism")

# Drop original categorical columns
input_data = input_data.drop(columns=["gender", "jaundice", "autism"])

# Concatenate numeric + encoded categorical features
input_data = pd.concat([input_data.reset_index(drop=True), gender_df, jaundice_df, autism_df], axis=1)

# -----------------------------
# Scale features
# -----------------------------
final_input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction button
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(final_input_scaled)[0]
    st.success(f"Predicted Class: {prediction}")
