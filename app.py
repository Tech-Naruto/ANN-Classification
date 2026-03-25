import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.keras')
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

with st.form('customer_details'):

    st.title('Customer Churn Prediction')

    creditScore = st.number_input('CreditScore')
    geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 18, 100)
    tenure = st.slider('Tenure', 0, 10)
    balance = st.number_input('Balance')
    noOfProducts = st.slider('NumOfProducts', 1, 4)
    hasCreditCard = st.selectbox('Has Credit Card', [0, 1])
    isActiveMember = st.selectbox('Is Active Member', [0, 1])
    estimatedSalary = st.number_input('EstimatedSalary')

    predict = st.form_submit_button('Predict')

if predict:

    customer = pd.DataFrame({
        'CreditScore': [creditScore],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [noOfProducts],
        'HasCrCard': [hasCreditCard],
        'IsActiveMember': [isActiveMember],
        'EstimatedSalary': [estimatedSalary]
    })

    customer = preprocessor.transform(customer)

    prediction = model.predict(customer)
    
    if prediction > 0.5:
        st.warning('The customer will leave the bank')
    else:
        st.success('The customer will not leave the bank')
    