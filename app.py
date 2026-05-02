import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center;'>🩺 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter patient details below</p>", unsafe_allow_html=True)

st.write("---")

# Create 2 columns layout
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    ped = st.number_input("Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

st.write("---")

# Center button
if st.button("🔍 Predict", use_container_width=True):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, ped, age]])
    data = scaler.transform(data)
    result = model.predict(data)

    st.write("---")

    if result[0] == 1:
        st.error("⚠️ High Risk: Diabetic")
    else:
        st.success("✅ Low Risk: Not Diabetic")

# Footer
st.write("---")
st.caption("Developed using Logistic Regression | Streamlit App")
