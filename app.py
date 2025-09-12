import streamlit as st
import pandas as pd
import joblib

# =========================
# Load calibrated model (kept for structure, but unused)
# =========================
model = joblib.load("pipeline_calibrated.joblib")

# =========================
# Translation dictionary (English only here — add zh, ms similarly)
# =========================
translations = {
    "en": {
        "title": "🩺 Early Diabetes Risk Prediction",
        "disclaimer_top": "⚠️ This is a Master's project. Output is for reference only.",
        "lang_btn": "🌐 Language",
        "age": "Your Age (years)",
        "bp": "Blood Pressure (mmHg)",
        "bp_hint": "Normal: <120/80 mmHg. High BP: ≥140/90 mmHg.",
        "chol": "Total Cholesterol (mmol/L)",
        "chol_hint": "Normal: <5.2 mmol/L. High: ≥6.2 mmol/L.",
        "gen_hlth": "General Health (1=Excellent, 5=Poor)",
        "gen_hlth_hint": "1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "bmi_result": "Your BMI is {bmi:.1f} — {status}",
        "phys_days": "Days Physical Health Not Good (0-30)",
        "sex": "Sex",
        "sex_female": "Female",
        "sex_male": "Male",
        "ment_days": "Days Mental Health Not Good (0-30)",
        "ment_hint": "Includes stress, depression, emotional problems.",
        "predict": "Predict Risk",
        "result": "Predicted Diabetes Risk",
        "low_msg": "✅ Congratulations! Please continue a healthy lifestyle.",
        "moderate_msg": "⚠️ Moderate risk — consider lifestyle improvements and regular check-ups.",
        "high_msg": "⚠️ Suggestion: Have a body check-up and maintain a healthy lifestyle.",
        "ref_note": "Disclaimer: This assessment is for reference only."
    }
}

# =========================
# Language toggle state
# =========================
if "show
