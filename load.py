import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load('car_price_model.pkl')

# Streamlit app
st.title('Car Price Prediction')

# Sidebar inputs
st.sidebar.header('Input Features')
horsepower = st.sidebar.slider('Horsepower', 50, 300, 150)
weight = st.sidebar.slider('Weight', 1500, 5000, 3000)

# User input data
input_data = pd.DataFrame({'horsepower': [horsepower], 'weight': [weight]})

# Make a prediction
prediction = model.predict(input_data)[0]

# Display prediction
st.write(f'Predicted Price: ${prediction:.2f}')
