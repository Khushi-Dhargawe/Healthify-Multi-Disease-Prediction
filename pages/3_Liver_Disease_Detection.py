import streamlit as st
import pickle
import numpy as np
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import LIVER_MODEL

st.set_page_config(page_title="Liver Disease Detection", page_icon="🫀")
st.title("🫀 Liver Disease Detection")
st.markdown("Enter patient lab report values to predict the likelihood of liver disease.")
st.markdown("---")

@st.cache_resource
def load_model():
    with open(LIVER_MODEL, "rb") as f:
        return pickle.load(f)

model = load_model()

col1, col2 = st.columns(2)
with col1:
    age          = st.number_input("Age (years)", 4, 90, 45)
    gender       = st.selectbox("Gender", ["Male", "Female"])
    gender_val   = 1 if gender == "Male" else 0
    tb           = st.number_input("Total Bilirubin (mg/dL)", 0.0, 75.0, 1.0, step=0.1)
    db           = st.number_input("Direct Bilirubin (mg/dL)", 0.0, 20.0, 0.3, step=0.1)
    alkphos      = st.number_input("Alkaline Phosphatase (IU/L)", 10, 2110, 200)

with col2:
    sgpt         = st.number_input("SGPT / ALT (IU/L)", 0, 2000, 35)
    sgot         = st.number_input("SGOT / AST (IU/L)", 0, 5000, 35, help="Aspartate aminotransferase")
    tp           = st.number_input("Total Proteins (g/dL)", 2.0, 9.0, 6.5, step=0.1)
    alb          = st.number_input("Albumin (g/dL)", 0.0, 6.0, 3.5, step=0.1)
    ag_ratio     = st.number_input("A/G Ratio", 0.0, 3.0, 1.0, step=0.1, help="Albumin/Globulin ratio")

if st.button("🔍 Predict Liver Disease Risk", type="primary"):
    input_data  = np.array([[age, gender_val, tb, db, alkphos, sgpt, sgot, tp, alb, ag_ratio]])
    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    risk_pct    = probability[1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error("### ⚠️ Liver Disease Risk Detected")
        st.markdown(f"**Risk Score: {risk_pct:.1f}%**")
        st.markdown("""
        **Recommendations:**
        - Consult a Hepatologist immediately
        - Avoid alcohol consumption completely
        - Follow a liver-friendly low-fat diet
        - Get ultrasound and liver function tests
        - Monitor bilirubin and enzyme levels regularly
        """)
    else:
        st.success("### ✅ Low Liver Disease Risk")
        st.markdown(f"**Risk Score: {risk_pct:.1f}%**")
        st.markdown("""
        **Recommendations:**
        - Maintain healthy lifestyle and diet
        - Limit alcohol consumption
        - Annual liver function test recommended
        """)

    st.progress(int(risk_pct), text=f"Liver Disease Risk: {risk_pct:.1f}%")

    # Key markers
    st.markdown("### 🧪 Key Biomarker Analysis")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Bilirubin", f"{tb} mg/dL", delta="Normal: 0.2-1.2" if tb <= 1.2 else "Elevated")
    with c2:
        st.metric("SGPT/ALT", f"{sgpt} IU/L", delta="Normal: 7-56" if sgpt <= 56 else "Elevated")
    with c3:
        st.metric("SGOT/AST", f"{sgot} IU/L", delta="Normal: 10-40" if sgot <= 40 else "Elevated")

st.markdown("---")
st.warning("⚠️ This tool is for informational purposes only. Consult a Hepatologist for proper diagnosis.")
