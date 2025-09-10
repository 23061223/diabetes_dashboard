import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model
# =========================
model = joblib.load("pipeline.joblib")

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
        "high_msg": "⚠️ Suggestion: Have a body check-up and maintain a healthy lifestyle.",
        "ref_note": "Disclaimer: This assessment is for reference only."
    }
}

# =========================
# Language toggle state
# =========================
if "show_lang" not in st.session_state:
    st.session_state.show_lang = False
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# =========================
# Language toggle button
# =========================
if st.button(translations[st.session_state.lang]["lang_btn"]):
    st.session_state.show_lang = not st.session_state.show_lang
if st.session_state.show_lang:
    chosen_lang = st.radio(
        "Choose language / 选择语言 / Pilih bahasa",
        ["en"],  # add zh, ms here
        horizontal=True
    )
    st.session_state.lang = chosen_lang

t = translations[st.session_state.lang]

# =========================
# Top disclaimer & title
# =========================
st.markdown(f"**{t['disclaimer_top']}**")
st.title(t["title"])

# =========================
# Inputs
# =========================
age = st.number_input(t["age"], min_value=18, max_value=100, value=30)

bp_input = st.text_input(t["bp"], placeholder="e.g., 120/80")
st.caption(t["bp_hint"])

chol_input = st.number_input(t["chol"], min_value=2.0, max_value=10.0, step=0.1)
st.caption(t["chol_hint"])

gen_hlth = st.selectbox(t["gen_hlth"], [1, 2, 3, 4, 5])
st.caption(t["gen_hlth_hint"])

height = st.number_input(t["height"], min_value=100, max_value=220, value=170)
weight = st.number_input(t["weight"], min_value=30, max_value=200, value=65)

# BMI calculation
bmi = weight / ((height / 100) ** 2)
bmi_status = "Healthy range" if 18.5 <= bmi <= 24.9 else "Outside healthy range"
st.info(t["bmi_result"].format(bmi=bmi, status=bmi_status))

phys_days = st.slider(t["phys_days"], 0, 30, 0)

sex_choice = st.radio(t["sex"], [t["sex_female"], t["sex_male"]])
sex = 0 if sex_choice == t["sex_female"] else 1

ment_days = st.slider(t["ment_days"], 0, 30, 0)
st.caption(t["ment_hint"])

# =========================
# Backend conversions
# =========================
def age_to_group(age):
    bins = [24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 200]
    for i, upper in enumerate(bins, start=1):
        if age <= upper:
            return i
age_group = age_to_group(age)

def bp_to_flag(bp_str):
    try:
        sys, dia = map(int, bp_str.split("/"))
        return 1 if sys >= 140 or dia >= 90 else 0
    except:
        return 0
high_bp = bp_to_flag(bp_input)

high_chol = 1 if chol_input >= 6.2 else 0

# =========================
# Prediction
# =========================
if st.button(t["predict"]):
    input_df = pd.DataFrame([[
        high_bp, high_chol, gen_hlth, bmi, phys_days,
        sex, age_group, ment_days
    ]], columns=[
        "HighBP", "HighChol", "GenHlth", "BMI", "PhysActivity",
        "Sex", "AgeGroup", "MentHlth"
    ])

    # Engineered features
    input_df["BMI_PhysAct"] = input_df["BMI"] * input_df["PhysActivity"]
    input_df["AgeGroup_Sq"] = input_df["AgeGroup"] ** 2

    # Prediction
    proba = model.predict_proba(input_df)[0, 1]

    # Risk display
    color = "green" if proba < 0.5 else "red"
    st.markdown(f"<h2 style='color:{color};'>{t['result']}: {proba:.1%}</h2>", unsafe_allow_html=True)
    if proba < 0.5:
        st.success(t["low_msg"])
    else:
        st.error(t["high_msg"])
    st.caption(t["ref_note"])
