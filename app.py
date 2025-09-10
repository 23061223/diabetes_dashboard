import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# Load model & SHAP explainer
# =========================
model = joblib.load("pipeline.joblib")
explainer = joblib.load("shap_explainer.pkl")

# =========================
# Translation dictionary
# =========================
translations = {
    "en": {
        "title": "🩺 Early Diabetes Risk Prediction",
        "lang_btn": "🌐 Language",
        "age_group": "Age Group",
        "high_bp": "High Blood Pressure",
        "high_chol": "High Cholesterol",
        "gen_hlth": "General Health (1=Excellent, 5=Poor)",
        "bmi": "Body Mass Index (BMI)",
        "phys_activity": "Physically Active?",
        "income": "Income Level (1=Lowest, 8=Highest)",
        "education": "Education Level (1=Lowest, 6=Highest)",
        "sex": "Sex (0=Female, 1=Male)",
        "ment_hlth": "Days Mental Health Not Good (0-30)",
        "predict": "Predict Risk",
        "result": "Predicted Diabetes Risk",
        "shap_title": "🔍 Feature Contributions to Prediction"
    },
    "zh": {
        "title": "🩺 早期糖尿病风险预测",
        "lang_btn": "🌐 语言",
        "age_group": "年龄组",
        "high_bp": "高血压",
        "high_chol": "高胆固醇",
        "gen_hlth": "总体健康状况 (1=优秀, 5=差)",
        "bmi": "身体质量指数 (BMI)",
        "phys_activity": "是否有体育活动？",
        "income": "收入水平 (1=最低, 8=最高)",
        "education": "教育水平 (1=最低, 6=最高)",
        "sex": "性别 (0=女, 1=男)",
        "ment_hlth": "心理健康不佳天数 (0-30)",
        "predict": "预测风险",
        "result": "预测糖尿病风险",
        "shap_title": "🔍 特征对预测的贡献"
    },
    "ms": {
        "title": "🩺 Ramalan Risiko Diabetes Awal",
        "lang_btn": "🌐 Bahasa",
        "age_group": "Kumpulan Umur",
        "high_bp": "Tekanan Darah Tinggi",
        "high_chol": "Kolesterol Tinggi",
        "gen_hlth": "Kesihatan Umum (1=Cemerlang, 5=Teruk)",
        "bmi": "Indeks Jisim Badan (BMI)",
        "phys_activity": "Aktif Secara Fizikal?",
        "income": "Tahap Pendapatan (1=Terendah, 8=Tertinggi)",
        "education": "Tahap Pendidikan (1=Terendah, 6=Tertinggi)",
        "sex": "Jantina (0=Perempuan, 1=Lelaki)",
        "ment_hlth": "Hari Kesihatan Mental Tidak Baik (0-30)",
        "predict": "Ramalkan Risiko",
        "result": "Risiko Diabetes Diramal",
        "shap_title": "🔍 Sumbangan Ciri kepada Ramalan"
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
        ["en", "zh", "ms"],
        format_func=lambda x: {"en": "English", "zh": "中文", "ms": "Bahasa Melayu"}[x],
        horizontal=True
    )
    st.session_state.lang = chosen_lang

# =========================
# Active translation
# =========================
t = translations[st.session_state.lang]

# =========================
# App title
# =========================
st.title(t["title"])

# =========================
# Input fields
# =========================
age_group = st.selectbox(t["age_group"], list(range(1, 14)))
high_bp = st.selectbox(t["high_bp"], [0, 1])
high_chol = st.selectbox(t["high_chol"], [0, 1])
gen_hlth = st.selectbox(t["gen_hlth"], [1, 2, 3, 4, 5])
bmi = st.number_input(t["bmi"], min_value=10.0, max_value=60.0, value=25.0)
phys_activity = st.selectbox(t["phys_activity"], [0, 1])
income = st.selectbox(t["income"], list(range(1, 9)))
education = st.selectbox(t["education"], list(range(1, 7)))
sex = st.selectbox(t["sex"], [0, 1])
ment_hlth = st.slider(t["ment_hlth"], 0, 30, 0)

# =========================
# Prediction + SHAP
# =========================
if st.button(t["predict"]):
    input_df = pd.DataFrame([[
        high_bp, high_chol, gen_hlth, bmi, phys_activity,
        income, education, sex, age_group, ment_hlth
    ]], columns=[
        "HighBP", "HighChol", "GenHlth", "BMI", "PhysActivity",
        "Income", "Education", "Sex", "AgeGroup", "MentHlth"
    ])
    
    # Prediction
    proba = model.predict_proba(input_df)[0, 1]
    st.metric(t["result"], f"{proba:.2%}")
    
    # SHAP values
    X_transformed = model.named_steps["prep"].transform(input_df)
    shap_values = explainer.shap_values(X_transformed)[0]
    
    # Map feature names to selected language
    feature_labels = [
        t["high_bp"], t["high_chol"], t["gen_hlth"], t["bmi"], t["phys_activity"],
        t["income"], t["education"], t["sex"], t["age_group"], t["ment_hlth"]
    ]
    
    # Plot SHAP values
    st.subheader(t["shap_title"])
    shap_df = pd.DataFrame({
        "Feature": feature_labels,
        "SHAP Value": shap_values
    }).sort_values("SHAP Value", key=abs, ascending=True)
    
    fig, ax = plt.subplots()
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=["#FF4B4B" if v > 0 else "#4BFF4B" for v in shap_df["SHAP Value"]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on Risk")
    st.pyplot(fig)
