import streamlit as st
import pickle
import numpy as np
import json
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import symptoms, disease, TRAINED_MODEL_RF, DISEASE_DATA, PRECAUTIONS_DATA, doctor_search

st.set_page_config(page_title="Disease Diagnosis", page_icon="🩺")
st.title("🩺 Disease Diagnosis")
st.markdown("Select your symptoms to get a predicted diagnosis, precautions, and specialist recommendations.")
st.markdown("---")

# Load model and data
@st.cache_resource
def load_model():
    with open(TRAINED_MODEL_RF, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_disease_data():
    with open(DISEASE_DATA) as f:
        return json.load(f)

@st.cache_data
def load_precautions():
    return pd.read_csv(PRECAUTIONS_DATA)

model       = load_model()
disease_data = load_disease_data()
precautions  = load_precautions()

# Symptom selection
st.subheader("Select your symptoms")
symptom_display = [s.replace("_", " ").title() for s in symptoms]
selected_display = st.multiselect(
    "Choose all symptoms you are experiencing:",
    options=symptom_display,
    help="Select as many symptoms as apply"
)

if st.button("🔍 Diagnose", type="primary"):
    if len(selected_display) < 2:
        st.warning("Please select at least 2 symptoms for accurate diagnosis.")
    else:
        # Encode symptoms
        selected_raw = [symptoms[symptom_display.index(s)] for s in selected_display]
        input_vector = np.zeros(len(symptoms))
        for s in selected_raw:
            if s in symptoms:
                input_vector[symptoms.index(s)] = 1

        # Predict
        prediction = model.predict([input_vector])[0]
        probabilities = model.predict_proba([input_vector])[0]
        confidence = probabilities[prediction] * 100
        predicted_disease = disease[prediction]

        st.markdown("---")
        st.success(f"### 🏥 Predicted Condition: **{predicted_disease}**")
        st.info(f"**Confidence:** {confidence:.1f}%")

        # Description
        if predicted_disease in disease_data:
            info = disease_data[predicted_disease]
            st.markdown(f"**ℹ️ About:** {info['description']}")
            specialist = info['specialist']
        else:
            specialist = "General Physician"

        # Precautions
        st.markdown("### 🛡️ Precautions")
        prec_row = precautions[precautions['Disease'] == predicted_disease]
        if not prec_row.empty:
            row = prec_row.iloc[0]
            for i in range(1, 5):
                col = f"Precaution_{i}"
                if col in row and pd.notna(row[col]):
                    st.markdown(f"- {row[col]}")
        else:
            st.markdown("- Consult a doctor immediately\n- Rest and stay hydrated")

        # Specialist
        st.markdown(f"### 👨‍⚕️ Recommended Specialist: **{specialist}**")
        search_url = doctor_search(specialist)
        st.markdown(f"[🔍 Find a {specialist} near you]({search_url})")

        # Top 3 predictions
        st.markdown("### 📊 Top 3 Possible Conditions")
        top3_idx = np.argsort(probabilities)[::-1][:3]
        for idx in top3_idx:
            d_name = disease[idx] if idx < len(disease) else f"Disease {idx}"
            prob   = probabilities[idx] * 100
            st.progress(int(prob), text=f"{d_name}: {prob:.1f}%")

st.markdown("---")
st.warning("⚠️ This tool is for informational purposes only. Always consult a qualified medical professional.")
