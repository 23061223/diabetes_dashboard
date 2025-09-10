import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# Load model & SHAP explainer
model = joblib.load("pipeline.joblib")
explainer = joblib.load("shap_explainer.pkl")

# Translation dictionary (shortened here for brevity — keep your full version)
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
        "shap_title": "🔍 Feature Contributions to Prediction",
        "low_msg": "✅ Congratulations! Please continue a healthy lifestyle.",
        "high_msg": "⚠️ Suggestion: Have a body check-up and maintain a healthy lifestyle.",
        "ref_note": "Disclaimer: This assessment is for reference only."
    }
    # zh, ms translations omitted here for brevity — keep your full set
}

# Language toggle state
if "show_lang" not in st.session_state:
    st.session_state.show_lang = False
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Language toggle button
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

# Top disclaimer
st.markdown(f"**{t['disclaimer_top']}**")

st.title(t["title"])

# Inputs
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

# Backend conversions
# Age → AgeGroup
def age_to_group(age):
    bins = [24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 200]
    for i, upper in enumerate(bins, start=1):
        if age <= upper:
            return i
age_group = age_to_group(age)

# BP → HighBP flag
def bp_to_flag(bp_str):
    try:
        sys, dia = map(int, bp_str.split("/"))
        return 1 if sys >= 140 or dia >= 90 else 0
    except:
        return 0
high_bp = bp_to_flag(bp_input)

# Cholesterol → HighChol flag
high_chol = 1 if chol_input >= 6.2 else 0

# Predict
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

    # SHAP aggregation (same Option 2 logic as before)
    X_transformed = model.named_steps["prep"].transform(input_df)
    shap_values = explainer.shap_values(X_transformed)[0]
    feature_names = model.named_steps["prep"].get_feature_names_out()
    mapping = {
        "HighBP": t["bp"],
        "HighChol": t["chol"],
        "GenHlth": t["gen_hlth"],
        "BMI": t["bmi"],
        "PhysActivity": t["phys_days"],
        "Sex": t["sex"],
        "AgeGroup": t["age"],
        "MentHlth": t["ment_days"],
        "BMI_PhysAct": "BMI × " + t["phys_days"],
        "AgeGroup_Sq": t["age"] + "²"
    }
    agg_shap = {}
    for orig in mapping:
        mask = [orig in fname for fname in feature_names]
        agg_shap[mapping[orig]] = np.sum(shap_values[mask])
    shap_df = pd.DataFrame({
        "Feature": list(agg_shap.keys()),
        "SHAP Value": list(agg_shap.values())
    }).sort_values("SHAP Value", key=abs, ascending=True)
    st.subheader(t["shap_title"])
    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"],
            color=["#FF4B4B" if v > 0 else "#4BFF4B" for v in shap_df["SHAP Value"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on Risk")
    st.pyplot(fig)
