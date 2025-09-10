import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap

# Load model and explainer
model = joblib.load("pipeline.joblib")
explainer = joblib.load("shap_explainer.pkl")

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Early Diabetes Risk Prediction")

# Input fields
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
age_group = st.selectbox("Age Group", options=list(range(1, 14)), format_func=lambda x: f"Group {x}")
phys_activity = st.radio("Physically Active?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

# Feature engineering
bmi_physact = bmi * phys_activity
agegroup_sq = age_group ** 2

# Predict
if st.button("Predict Risk"):
    input_data = pd.DataFrame([[bmi, age_group, bmi_physact, agegroup_sq]],
                              columns=["BMI", "AgeGroup", "BMI_PhysActivity", "AgeGroup_sq"])
    proba = model.predict_proba(input_data)[0, 1]
    st.metric(label="Predicted Diabetes Risk", value=f"{proba:.2%}")

    # SHAP explanation
    scaled_input = model.named_steps["prep"].transform(input_data)
    shap_values = explainer.shap_values(scaled_input)[0]
    st.subheader("🔍 Feature Contributions")
    for feature, value in zip(input_data.columns, shap_values):
        st.write(f"**{feature}**: {value:+.4f}")
