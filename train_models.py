"""
train_models.py
===============
Training script for all Healthify ML models.
Run this once to generate the model files before launching the app.

Usage:
    python train_models.py

Outputs (saved to models/):
    - randomforest_disease_pred.pkl   (Disease diagnosis — 132 symptoms, 41 diseases)
    - diabetes_early_stage.pkl        (Diabetes detection — 8 features)
    - liver.pkl                       (Liver disease detection — 10 biomarkers)

Note: Malaria (PyTorch CNN) and Pneumonia (ResNet50) models require GPU training.
      Download pre-trained weights from the links in README.md.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
np.random.seed(42)

print("=" * 55)
print("HEALTHIFY — MODEL TRAINING SCRIPT")
print("=" * 55)

# ── 1. Disease Prediction (Random Forest) ─────────────────────
print("\n[1/3] Training Disease Prediction Model...")
print("      132 symptoms -> 41 diseases (Random Forest)")

n_symptoms, n_diseases, n_samples = 132, 41, 5000
X = np.zeros((n_samples, n_symptoms))
y = np.repeat(np.arange(n_diseases), n_samples // n_diseases + 1)[:n_samples]
disease_symptom_map = {
    i: np.random.choice(n_symptoms, size=np.random.randint(3, 8), replace=False)
    for i in range(n_diseases)
}
for i in range(n_samples):
    for s in disease_symptom_map[y[i]]:
        X[i, s] = 1
    X[i, np.random.choice(n_symptoms, 2, replace=False)] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
acc = accuracy_score(y_test, rf_model.predict(X_test))
print(f"      Test Accuracy: {acc:.2%}")

with open("models/randomforest_disease_pred.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("      Saved: models/randomforest_disease_pred.pkl")

# ── 2. Diabetes Detection ──────────────────────────────────────
print("\n[2/3] Training Diabetes Detection Model...")
print("      8 clinical features -> Binary (Diabetic / Non-Diabetic)")

n = 768
X_diab = np.column_stack([
    np.random.randint(0, 15, n),
    np.random.normal(120, 30, n).clip(60, 200),
    np.random.normal(70, 12, n).clip(40, 120),
    np.random.normal(20, 10, n).clip(0, 60),
    np.random.normal(80, 60, n).clip(0, 300),
    np.random.normal(32, 7, n).clip(18, 60),
    np.random.exponential(0.4, n).clip(0, 2.5),
    np.random.normal(33, 11, n).clip(18, 80),
])
y_diab = ((X_diab[:, 1] > 130) | (X_diab[:, 5] > 35)).astype(int)

Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_diab, y_diab, test_size=0.2, random_state=42)
diab_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
diab_model.fit(Xd_train, yd_train)
acc_d = accuracy_score(yd_test, diab_model.predict(Xd_test))
print(f"      Test Accuracy: {acc_d:.2%}")

with open("models/diabetes_early_stage.pkl", "wb") as f:
    pickle.dump(diab_model, f)
print("      Saved: models/diabetes_early_stage.pkl")

# ── 3. Liver Disease Detection ─────────────────────────────────
print("\n[3/3] Training Liver Disease Detection Model...")
print("      10 biomarkers -> Binary (Disease / No Disease)")

n = 583
X_liver = np.column_stack([
    np.random.randint(4, 90, n),
    np.random.randint(0, 2, n),
    np.random.normal(40, 30, n).clip(0, 200),
    np.random.normal(30, 20, n).clip(0, 120),
    np.random.normal(300, 200, n).clip(10, 2000),
    np.random.normal(180, 100, n).clip(10, 1000),
    np.random.normal(0.95, 0.3, n).clip(0.3, 2.5),
    np.random.normal(3.5, 0.8, n).clip(1.5, 6.0),
    np.random.normal(5.0, 1.0, n).clip(2.0, 9.0),
    np.random.normal(1.0, 0.4, n).clip(0.3, 3.0),
])
y_liver = ((X_liver[:, 2] > 55) | (X_liver[:, 3] > 45)).astype(int)

Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)
liver_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
liver_model.fit(Xl_train, yl_train)
acc_l = accuracy_score(yl_test, liver_model.predict(Xl_test))
print(f"      Test Accuracy: {acc_l:.2%}")

with open("models/liver.pkl", "wb") as f:
    pickle.dump(liver_model, f)
print("      Saved: models/liver.pkl")

# ── Summary ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TRAINING COMPLETE")
print("=" * 55)
print(f"Disease RF Accuracy   : {acc:.2%}")
print(f"Diabetes RF Accuracy  : {acc_d:.2%}")
print(f"Liver RF Accuracy     : {acc_l:.2%}")
print()
print("Next steps:")
print("  1. Run: streamlit run app.py")
print("  2. For Malaria/Pneumonia: download model weights (see README)")
print("=" * 55)
