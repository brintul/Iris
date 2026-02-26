import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

st.title("Iris Flower Prediction App 🌸")

st.write("Masukkan nilai fitur bunga:")

# Input user
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    classes = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Prediksi bunga: {classes[prediction[0]]}")
