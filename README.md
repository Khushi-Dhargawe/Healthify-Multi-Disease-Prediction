# 🏥 Healthify — Multi-Disease Prediction Web App

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)
![ResNet50](https://img.shields.io/badge/ResNet50-Transfer%20Learning-purple)
![Mumbai University](https://img.shields.io/badge/Mumbai%20University-BE%20AI%20%26%20ML-darkblue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Skills:** Python · Streamlit · Scikit-learn · PyTorch · ResNet50 · TensorFlow · Deep Learning  
> **Association:** A.P. Shah Institute of Technology, Thane — BE AI & ML, Mumbai University

---

## Why This Matters

Most people can't get immediate answers about their symptoms — 
Healthify bridges that gap with 5 AI modules covering diagnosis, 
lab-based detection and medical image analysis, all in one Streamlit app.

---

## 📌 Project Overview

**Healthify** is a 5-module health prediction web application built with Streamlit, using machine learning and deep learning to provide real-time health assessments across multiple medical domains.

**5 Prediction Modules:**

| Module | Input | Model | Output | Business Implication |
|---|---|---|---|---|
| 🩺 Disease Diagnosis | 132 symptoms | Random Forest | 41 diseases + precautions | Triage support — reduces unnecessary GP visits |
| 🩸 Diabetes Detection | 8 lab attributes | Random Forest | Risk score + recommendations | Early detection saves long-term treatment costs |
| 🫀 Liver Disease | 10 biomarkers | Random Forest | Risk score + biomarker analysis | Flags enzyme elevation before clinical symptoms appear |
| 🦟 Malaria Detection | Cell image | PyTorch CNN | Parasitised / Uninfected | Scalable screening in low-resource settings |
| 🫁 Pneumonia Detection | Chest X-ray | ResNet50 | Normal / Pneumonia | Radiologist support tool — reduces misdiagnosis risk |

---

## 🗂️ Repository Structure & How Files Connect

```
📁 Healthify-Multi-Disease-Prediction/
│
├── 🐍 app.py                               ← MAIN: Streamlit home page
│
├── 📁 pages/                               ← Streamlit multi-page routing
│   ├── 1_Disease_Diagnosis.py             ← Module 1: RF → 41 diseases
│   ├── 2_Diabetes_Detection.py            ← Module 2: RF → diabetes risk
│   ├── 3_Liver_Disease_Detection.py       ← Module 3: RF → liver risk
│   ├── 4_Malaria_Detection.py             ← Module 4: CNN → cell image
│   └── 5_Pneumonia_Detection.py           ← Module 5: ResNet50 → X-ray
│
├── 🐍 constants.py                         ← Shared: symptoms, diseases, paths
├── 🐍 train_models.py                      ← Train all 3 sklearn models
│
├── 📁 models/                              ← Trained model files (generated)
│   ├── randomforest_disease_pred.pkl
│   ├── diabetes_early_stage.pkl
│   ├── liver.pkl
│   ├── malaria_detection.pt               ← Download separately (see below)
│   └── pneumonia_resnet50_model.pth        ← Download separately (see below)
│
├── 📁 data/
│   ├── disease.json                        ← Disease descriptions + specialists
│   └── precautions.csv                     ← Precautions per disease
│
├── 📦 requirements.txt
└── 📜 LICENSE
```

### 🔗 Pipeline Flow

```
User Input (Streamlit UI)
        │
        ├── Symptoms selected ──────────► Random Forest ──► Disease + Precautions + Specialist
        ├── Lab values (diabetes) ──────► RF + Scaler ───► Risk Score + Recommendations
        ├── Lab values (liver) ─────────► RF + Scaler ───► Risk Score + Biomarker Analysis
        ├── Cell image uploaded ────────► PyTorch CNN ──► Parasitised / Uninfected
        └── Chest X-ray uploaded ───────► ResNet50 ─────► Normal / Pneumonia
```

---

## 🤖 Models

### Module 1 — Disease Diagnosis (Random Forest)
- **Features:** 132 binary symptom inputs
- **Output:** 41 disease classes + confidence scores
- **Extras:** Precautions, doctor search, top-3 predictions

### Module 2 — Diabetes Detection (Random Forest)
- **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, DPF, Age
- **Dataset:** Pima Indians Diabetes Dataset (768 records)
- **Output:** Binary risk score + metric analysis

### Module 3 — Liver Disease Detection (Random Forest)
- **Features:** Age, Gender, Total/Direct Bilirubin, Alkaline Phosphatase, SGPT, SGOT, TP, Albumin, A/G Ratio
- **Dataset:** Indian Liver Patient Dataset (583 records)
- **Output:** Binary risk score + biomarker highlights

### Module 4 — Malaria Detection (PyTorch CNN)
- **Dataset:** [Kaggle Malaria Cell Images](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (27,558 images)
- **Architecture:** Custom CNN (3 conv blocks, 100×100 RGB input)
- **Output:** Parasitised / Uninfected with confidence

### Module 5 — Pneumonia Detection (ResNet50)
- **Dataset:** [Kaggle Chest X-Ray](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (5,863 images)
- **Architecture:** ResNet50 fine-tuned (last FC layer replaced)
- **Output:** Normal / Pneumonia with severity level

---

## What I'd Do With More Time

With more time I'd deploy this on Streamlit Cloud with a public URL 
so recruiters can interact with it live. I'd also replace the synthetic 
training data in train_models.py with the actual Pima Indians and Indian 
Liver Patient datasets for production-grade model accuracy.

---

## 🚀 How to Run

### Step 1 — Clone and install
```bash
git clone https://github.com/Khushi-Dhargawe/Healthify-Multi-Disease-Prediction.git
cd Healthify-Multi-Disease-Prediction
pip install -r requirements.txt
```

### Step 2 — Train sklearn models
```bash
python train_models.py
```

### Step 3 — Launch the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 📁 Related Projects

| # | Project | Skills |
|---|---|---|
| 8 | [Signature Detection](../Signature-Detection-Verification) | PyTorch · CNN · OpenCV |
| **9** | **Healthify ← You are here** | **Streamlit · PyTorch · ResNet50** |
| 10 | [Prodigy ML Internship](../Prodigy-ML-Internship) | Python · Scikit-learn · OpenCV |

---

## 👩‍💻 Author

**Khushi Dhargawe**  
BE AI & ML (Hons. Cybersecurity) — A.P. Shah Institute of Technology, Mumbai University  
MSc Business Analytics — University College Cork (UCC)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/khushi-dhargawe/)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/Khushi-Dhargawe)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
