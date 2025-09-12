import streamlit as st
import pandas as pd
import joblib

# =========================
# Load calibrated model
# =========================
model = joblib.load("pipeline_calibrated.joblib")

# =========================
# Translation dictionary
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
    },
    "zh": {
        "title": "🩺 早期糖尿病风险预测",
        "disclaimer_top": "⚠️ 这是硕士项目，结果仅供参考。",
        "lang_btn": "🌐 语言",
        "age": "您的年龄（岁）",
        "bp": "血压 (mmHg)",
        "bp_hint": "正常：<120/80 mmHg。高血压：≥140/90 mmHg。",
        "chol": "总胆固醇 (mmol/L)",
        "chol_hint": "正常：<5.2 mmol/L。偏高：≥6.2 mmol/L。",
        "gen_hlth": "总体健康状况 (1=优秀, 5=差)",
        "gen_hlth_hint": "1=优秀, 2=很好, 3=好, 4=一般, 5=差",
        "height": "身高 (cm)",
        "weight": "体重 (kg)",
        "bmi_result": "您的BMI为 {bmi:.1f} — {status}",
        "phys_days": "身体不适天数 (0-30)",
        "sex": "性别",
        "sex_female": "女",
        "sex_male": "男",
        "ment_days": "心理不适天数 (0-30)",
        "ment_hint": "包括压力、抑郁、情绪问题。",
        "predict": "预测风险",
        "result": "预测糖尿病风险",
        "low_msg": "✅ 恭喜！请继续保持健康的生活方式。",
        "moderate_msg": "⚠️ 中等风险 — 建议改善生活方式并定期检查。",
        "high_msg": "⚠️ 建议体检并保持健康的生活方式。",
        "ref_note": "免责声明：本评估仅供参考。"
    },
    "ms": {
        "title": "🩺 Ramalan Awal Risiko Diabetes",
        "disclaimer_top": "⚠️ Ini adalah projek Sarjana. Keputusan hanya untuk rujukan.",
        "lang_btn": "🌐 Bahasa",
        "age": "Umur Anda (tahun)",
        "bp": "Tekanan Darah (mmHg)",
        "bp_hint": "Normal: <120/80 mmHg. Tekanan tinggi: ≥140/90 mmHg.",
        "chol": "Kolesterol Jumlah (mmol/L)",
        "chol_hint": "Normal: <5.2 mmol/L. Tinggi: ≥6.2 mmol/L.",
        "gen_hlth": "Kesihatan Umum (1=Cemerlang, 5=Teruk)",
        "gen_hlth_hint": "1=Cemerlang, 2=Sangat Baik, 3=Baik, 4=Sederhana, 5=Teruk",
        "height": "Tinggi (cm)",
        "weight": "Berat (kg)",
        "bmi_result": "BMI anda ialah {bmi:.1f} — {status}",
        "phys_days": "Hari Kesihatan Fizikal Tidak Baik (0-30)",
        "sex": "Jantina",
        "sex_female": "Perempuan",
        "sex_male": "Lelaki",
        "ment_days": "Hari Kesihatan Mental Tidak Baik (0-30)",
        "ment_hint": "Termasuk tekanan, kemurungan, masalah emosi.",
        "predict": "Ramalkan Risiko",
        "result": "Risiko Diabetes Diramalkan",
        "low_msg": "✅ Tahniah! Teruskan gaya hidup sihat.",
        "moderate_msg": "⚠️ Risiko sederhana — pertimbangkan penambahbaikan gaya hidup dan pemeriksaan berkala.",
        "high_msg": "⚠️ Cadangan: Lakukan pemeriksaan kesihatan dan kekalkan gaya hidup sihat.",
        "ref_note": "Penafian: Penilaian ini hanya untuk rujukan."
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
if 18.5 <= bmi <= 24.9:
    bmi_status = "Healthy range (18.5–24.9 kg/m²)"
else:
    bmi_status = "Outside healthy range (Healthy BMI: 18.5–24.9 kg/m²)"
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

def bp_to_flag
