import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page title
st.set_page_config(page_title='Sales Prediction App')

# Load the trained model
try:
    model = joblib.load('linear_regression_model.sav')
    st.title('📊 Advertising Sales Predictor')
    st.markdown('Predict sales based on advertising expenditure in TV, Radio, and Newspaper.')

    # Input section
    st.sidebar.header('Input Advertising Budgets')
    tv = st.sidebar.number_input('TV Budget ($)', min_value=0.0, value=150.0)
    radio = st.sidebar.number_input('Radio Budget ($)', min_value=0.0, value=25.0)
    newspaper = st.sidebar.number_input('Newspaper Budget ($)', min_value=0.0, value=30.0)

    # Prediction logic
    if st.button('Predict Sales'):
        # Create input array (ensuring same shape as training)
        features = np.array([[tv, radio, newspaper]])
        prediction = model.predict(features)
        
        st.success(f'### Predicted Sales: {prediction[0]:.2f} units')
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("TV", f"${tv}")
        col2.metric("Radio", f"${radio}")
        col3.metric("Newspaper", f"${newspaper}")

except FileNotFoundError:
    st.error('Model file "linear_regression_model.sav" not found. Please run the training cells first.')
