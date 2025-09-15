# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load trained objects
# ---------------------------
best_svc = joblib.load("svc_model.pkl")
scaler = joblib.load("scaler.pkl")
le_country = joblib.load("country_encoder.pkl")
le_used_app = joblib.load("used_app_encoder.pkl")
ohe = joblib.load("onehot_encoder.pkl")
le_class = joblib.load("class_encoder.pkl")

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Autism Spectrum Disorder Screening Prediction 🧩")
st.write("Enter the details below to predict whether a child is likely to have autism or not:")

# Input features
A1 = st.selectbox("A1 Score", [0, 1])
A2 = st.selectbox("A2 Score", [0, 1])
A3 = st.selectbox("A3 Score", [0, 1])
A4 = st.selectbox("A4 Score", [0, 1])
A5 = st.selectbox("A5 Score", [0, 1])
A6 = st.selectbox("A6 Score", [0, 1])
A7 = st.selectbox("A7 Score", [0, 1])
A8 = st.selectbox("A8 Score", [0, 1])
A9 = st.selectbox("A9 Score", [0, 1])
A10 = st.selectbox("A10 Score", [0, 1])

age = st.number_input("Age", min_value=1, max_value=18, value=5)

gender = st.selectbox("Gender", ['male', 'female'])
jaundice = st.selectbox("Jaundice", ['yes', 'no'])
autism = st.selectbox("Autism", ['yes', 'no'])

country_of_res = st.selectbox("Country of Residence", le_country.classes_)
used_app_before = st.selectbox("Used App Before?", le_used_app.classes_)

# Predict button
if st.button("Predict"):
    # Create DataFrame
    user_input = {
        'A1_Score': A1, 'A2_Score': A2, 'A3_Score': A3, 'A4_Score': A4, 'A5_Score': A5,
        'A6_Score': A6, 'A7_Score': A7, 'A8_Score': A8, 'A9_Score': A9, 'A10_Score': A10,
        'age': age, 'gender': gender, 'jaundice': jaundice, 'autism': autism,
        'country_of_res': country_of_res, 'used_app_before': used_app_before
    }

    new_df = pd.DataFrame([user_input])

    # One-hot encode gender, jaundice, autism
    categorical_cols_ohe = ['gender', 'jaundice', 'autism']
    encoded_ohe = ohe.transform(new_df[categorical_cols_ohe])
    ohe_cols = ohe.get_feature_names_out(categorical_cols_ohe)
    encoded_df = pd.DataFrame(encoded_ohe, columns=ohe_cols, index=new_df.index)
    new_df = pd.concat([new_df, encoded_df], axis=1)
    new_df.drop(columns=categorical_cols_ohe, inplace=True)

    # Label encode country and used_app_before
    new_df['country_of_res'] = le_country.transform(new_df['country_of_res'])
    new_df['used_app_before'] = le_used_app.transform(new_df['used_app_before'])

    # Ensure column order matches training features
    try:
        from training_data import X  # If you saved your training X somewhere
        new_df = new_df[X.columns]
    except:
        st.warning("Training features (X) not available. Make sure column order matches training data.")

    # Scale features
    new_scaled = scaler.transform(new_df)

    # Predict
    prediction = best_svc.predict(new_scaled)[0]
    predicted_label = le_class.inverse_transform([prediction])[0]

    st.success(f"Predicted Class: {predicted_label}")

    # Predict probabilities
    if hasattr(best_svc, 'predict_proba'):
        probabilities = best_svc.predict_proba(new_scaled)[0]
        prob_df = pd.DataFrame({
            'Class': le_class.inverse_transform(best_svc.classes_),
            'Probability': probabilities
        })
        st.write("Prediction Probabilities:")
        st.dataframe(prob_df)
