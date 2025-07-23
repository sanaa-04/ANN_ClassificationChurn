import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model and scaler
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    ohe = pickle.load(file)

# Streamlit app
st.title("Bank Customer Churn Prediction")
st.write("Enter the customer details to predict churn probability.")

# Input fields for customer details
age = st.number_input("Age", min_value=0, max_value=100, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
geo = st.selectbox("Geography", options=["France", "Spain", "Germany"])
tenure = st.number_input("Tenure (in years)", min_value=0, value=1)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=5, value=1)
has_cr_card = st.selectbox("Has Credit Card", options=[1, 0])
is_active_member = st.selectbox("Is Active Member", options=[1, 0])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# Predict button
if st.button("Predict"):
    # Preprocess the input data
    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [label_encoder.transform([gender])[0]],
        "Geography": [geo],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Encode categorical features
    geo_encoded = ohe.transform(input_data[["Geography"]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(["Geography"]))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    

    # Scale the input data
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    st.write(f"Churn Probability: {prediction[0][0]:.2f}")

    if prediction[0][0] > 0.5:
        st.write("The customer is likely to leave the bank.")   
    else:
        st.write("The customer is likely to stay with the bank.")