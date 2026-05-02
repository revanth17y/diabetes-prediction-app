import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Prediction")

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("BloodPressure")
skin = st.number_input("SkinThickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
ped = st.number_input("Pedigree Function")
age = st.number_input("Age")

if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, ped, age]])
    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.write("Diabetic")
    else:
        st.write("Not Diabetic")