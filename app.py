import streamlit as st

st.set_page_config(
    page_title="Healthify — Your AI Health Consultant",
    page_icon="👨‍⚕️",
    layout="wide",
)

st.sidebar.info(
    "**About Healthify**: Built using publicly available data and open-source ML models. "
    "We do not store any patient personal information."
)

# Title
st.markdown(
    "<h1 style='text-align:center; color:#2563EB;'>👨‍⚕️ Healthify</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:20px; color:#555;'>"
    "Improving Healthcare · Improving Lives · Bridging Technology and Health"
    "</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# About
st.markdown("<h2 style='text-align:center; color:#1D4ED8;'>About Healthify</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    💠 **AI-Powered Health Consultant** — State-of-the-art machine learning and deep learning models
    to provide personalised health assessments across 5 medical domains.
    """)
with col2:
    st.markdown("""
    💠 **Privacy First** — We do not store or share any patient data.
    All predictions are computed locally in real time.
    """)

st.markdown("---")

# Services
st.markdown("<h2 style='text-align:center; color:#1D4ED8;'>Our Services</h2>", unsafe_allow_html=True)
st.write("")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    ### 🩺 Disease Diagnosis
    Select from **132 symptoms** across 41 diseases.
    Get diagnosis, precautions, medication guidance, and specialist recommendations.
    - **Model:** Random Forest
    - **Diseases:** 41 conditions
    - **Symptoms:** 132 features
    """)

with c2:
    st.markdown("""
    ### 🩸 Diabetes Detection
    Enter lab report attributes to detect early-stage diabetes risk.
    - **Model:** Random Forest + StandardScaler
    - **Features:** 8 clinical attributes
    - **Output:** Risk score + recommendations
    """)

with c3:
    st.markdown("""
    ### 🫀 Liver Disease Detection
    Enter blood test values to assess liver disease risk.
    - **Model:** Random Forest + StandardScaler
    - **Features:** 10 biomarkers
    - **Output:** Risk score + biomarker analysis
    """)

st.write("")
c4, c5, _ = st.columns(3)
with c4:
    st.markdown("""
    ### 🦟 Malaria Detection
    Upload a **microscopic cell image** to detect malaria parasites.
    - **Model:** PyTorch CNN
    - **Dataset:** Kaggle Cell Images (27,558 images)
    - **Output:** Parasitised / Uninfected
    """)

with c5:
    st.markdown("""
    ### 🫁 Pneumonia Detection
    Upload a **chest X-ray** to detect pneumonia using ResNet50.
    - **Model:** ResNet50 (transfer learning)
    - **Dataset:** Kaggle Chest X-Ray (5,863 images)
    - **Output:** Normal / Pneumonia
    """)

st.markdown("---")

# Tech stack
st.markdown("<h2 style='text-align:center; color:#1D4ED8;'>Technology Stack</h2>", unsafe_allow_html=True)
t1, t2, t3, t4 = st.columns(4)
with t1:
    st.metric("Frontend", "Streamlit")
with t2:
    st.metric("ML Models", "Scikit-learn")
with t3:
    st.metric("Deep Learning", "PyTorch")
with t4:
    st.metric("CV Model", "ResNet50")

st.markdown("---")
st.warning(
    "⚠️ **Disclaimer:** The information on this site is not intended to be a substitute for professional "
    "medical advice, diagnosis or treatment. Always consult a qualified medical professional. "
    "NEVER DISREGARD PROFESSIONAL MEDICAL ADVICE BECAUSE OF SOMETHING READ ON THIS WEBSITE."
)
