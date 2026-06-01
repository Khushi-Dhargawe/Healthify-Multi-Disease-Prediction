import streamlit as st
import pickle
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import DIABETES_MODEL

st.set_page_config(page_title="Diabetes Detection", page_icon="🩸")
st.title("🩸 Early Diabetes Detection")
st.markdown("Enter patient lab report values to predict the likelihood of diabetes.")
st.markdown("---")

@st.cache_resource
def load_model():
    with open(DIABETES_MODEL, "rb") as f:
        return pickle.load(f)

model = load_model()

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, help="Number of pregnancies")
    glucose     = st.number_input("Glucose (mg/dL)", min_value=50, max_value=250, value=110, help="Plasma glucose concentration")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=30, max_value=140, value=72, help="Diastolic blood pressure")
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, help="Triceps skin fold thickness")

with col2:
    insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, value=80, help="2-hour serum insulin")
    bmi     = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=25.0, step=0.1, help="Body mass index")
    dpf     = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.01, help="Genetic predisposition score")
    age     = st.number_input("Age (years)", min_value=18, max_value=100, value=30, help="Patient age")

if st.button("🔍 Predict Diabetes Risk", type="primary"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    risk_pct    = probability[1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error(f"### ⚠️ High Diabetes Risk Detected")
        st.markdown(f"**Risk Score: {risk_pct:.1f}%**")
        st.markdown("""
        **Recommendations:**
        - Schedule an appointment with an Endocrinologist immediately
        - Monitor blood glucose levels daily
        - Reduce sugar and refined carbohydrate intake
        - Increase physical activity (30 min/day)
        - Consider HbA1c test for confirmation
        """)
    else:
        st.success(f"### ✅ Low Diabetes Risk")
        st.markdown(f"**Risk Score: {risk_pct:.1f}%**")
        st.markdown("""
        **Recommendations:**
        - Maintain healthy BMI and regular exercise
        - Annual blood glucose screening recommended
        - Continue balanced diet and healthy lifestyle
        """)

    # Risk gauge
    st.markdown("### 📊 Risk Level")
    st.progress(int(risk_pct), text=f"Diabetes Risk: {risk_pct:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Glucose Level", f"{glucose} mg/dL", delta="Normal: 70-99" if glucose <= 99 else "Above normal")
    with col_b:
        st.metric("BMI", f"{bmi:.1f} kg/m²", delta="Normal: 18.5-24.9" if 18.5 <= bmi <= 24.9 else "Outside normal")

st.markdown("---")
st.warning("⚠️ This tool is for informational purposes only. Consult an Endocrinologist for proper diagnosis.")
