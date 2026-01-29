import streamlit as st
import numpy as np
import pickle

# Load model
with open("model.pickle", "rb") as f:
    model = pickle.load(f)

st.title("Iris Species Prediction App")

# Input fields
sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.0)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 5.0)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 5.0)
petal_width = st.number_input("Petal Width", 0.0, 10.0, 5.0)

# Button
predict = st.button("Predict Species")

if predict:
    
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Prediction
    prediction = model.predict(input_data)
    
    # Show result
    st.success(f"Predicted species is: {prediction[0]}")
