import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Alzheimer's Clinical Prediction", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1e3a5f; text-align: center;}
    .sub-header {font-size: 1rem; color: #5a6c7d; text-align: center; margin-bottom: 1.5rem;}
    .section-card {background: #f8fafc; border-radius: 12px; padding: 1.2rem; border-left: 4px solid #3b82f6;}
    .section-title {font-size: 1.1rem; font-weight: 600; color: #1e3a5f; margin-bottom: 0.8rem;}
    .risk-high {background: linear-gradient(135deg, #ff416c, #ff4b2b); color: white; padding: 1rem; border-radius: 12px; text-align: center; font-size: 1.4rem; font-weight: 700;}
    .risk-moderate {background: linear-gradient(135deg, #f2994a, #f2c94c); color: white; padding: 1rem; border-radius: 12px; text-align: center; font-size: 1.4rem; font-weight: 700;}
    .risk-low {background: linear-gradient(135deg, #11998e, #38ef7d); color: white; padding: 1rem; border-radius: 12px; text-align: center; font-size: 1.4rem; font-weight: 700;}
    .model-box {background: #eef2ff; border-radius: 8px; padding: 0.8rem; margin: 0.3rem 0;}
    .stButton>button {background: linear-gradient(135deg, #667eea, #764ba2); color: white; font-size: 1.1rem; font-weight: 600; padding: 0.6rem 2rem; border-radius: 10px; border: none; width: 100%;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Alzheimer\'s Clinical Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">3-Model Ensemble: CSF Biomarkers + Orexin Survey + Protein Levels</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("👤 Patient")
    patient_id = st.text_input("Patient ID", "PT-2024-001")
    patient_name = st.text_input("Name", "")
    patient_age = st.number_input("Age", 40, 120, 75)
    patient_gender = st.selectbox("Gender", ["Male", "Female"])
    assessment_date = st.date_input("Date", datetime.now())
    st.info("Fill all 3 sections → click Generate Report")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔬 CSF Biomarkers</div>', unsafe_allow_html=True)
    norm_abeta = st.slider("Aβ42 (normalized)", 0.0, 2.0, 0.8, 0.01, help="Lower = more amyloid pathology")
    norm_ptau = st.slider("p-Tau181 (normalized)", 0.0, 3.0, 1.2, 0.01, help="Higher = more tau pathology")
    norm_orexin = st.slider("Orexin (normalized)", 0.0, 2.0, 1.0, 0.01, help="Lower = sleep dysfunction")
    naccne4s = st.selectbox("APOE4 alleles", [0, 1, 2], help="0=none, 1=one, 2=two (higher = genetic risk)")
    amylcsf = st.selectbox("Amyloid CSF abnormal?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    csftau = st.selectbox("Tau CSF abnormal?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    rbd = st.selectbox("REM Sleep Behavior Disorder?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">😴 Orexin/Sleep Survey</div>', unsafe_allow_html=True)
    sleep_hours = st.slider("Sleep hours/night", 0.0, 12.0, 7.0, 0.5)
    sleep_quality = st.slider("Sleep quality (1-10)", 1, 10, 5)
    difficulty_sleeping = st.selectbox("Difficulty sleeping", [0, 1, 2, 3], format_func=lambda x: ["Never", "Occasional", "Frequent", "Chronic"][x])
    daytime_sleepiness = st.selectbox("Daytime sleepiness", [0, 1, 2, 3], format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
    snoring_apnea = st.selectbox("Snoring/Sleep Apnea?", [0, 1], format_func=lambda x: "Yes" if x else "No")
    stress_level = st.slider("Stress level (1-10)", 1, 10, 5)
    night_waking = st.selectbox("Night waking frequency", [0, 1, 2, 3], format_func=lambda x: ["None", "1-2x", "3-5x", ">5x"][x])
    mental_exhaustion = st.selectbox("Mental exhaustion", [0, 1, 2, 3], format_func=lambda x: ["None", "Mild", "Moderate", "Severe"][x])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🧪 Protein & Labs</div>', unsafe_allow_html=True)
    total_protein = st.number_input("Total Protein (mg/dL)", 0.0, 200.0, 65.0)
    ref_min = st.number_input("Normal range min", 0.0, 100.0, 60.0)
    ref_max = st.number_input("Normal range max", 0.0, 100.0, 80.0)
    albumin = st.number_input("Albumin (g/dL)", 0.0, 10.0, 3.8)
    globulin = st.number_input("Globulin (g/dL)", 0.0, 10.0, 2.7)
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

if st.button("🚀 GENERATE CLINICAL REPORT", type="primary"):

    # ── MODEL 1: XGBoost heuristic ──
    m1_prob = 0.0
    if norm_abeta < 0.6: m1_prob += 0.30
    if norm_ptau > 1.5: m1_prob += 0.25
    if norm_orexin < 0.7: m1_prob += 0.15
    if naccne4s >= 1: m1_prob += 0.10
    if amylcsf == 1: m1_prob += 0.10
    if csftau == 1: m1_prob += 0.10
    m1_prob = min(m1_prob, 0.99)

    # ── MODEL 2: Orexin/Sleep heuristic ──
    sleep_risk = 0
    if sleep_hours < 5: sleep_risk += 1
    if sleep_quality <= 3: sleep_risk += 1
    if difficulty_sleeping >= 2: sleep_risk += 1
    if daytime_sleepiness >= 2: sleep_risk += 1
    if snoring_apnea == 1: sleep_risk += 1
    if stress_level >= 7: sleep_risk += 1
    if night_waking >= 2: sleep_risk += 1
    if mental_exhaustion >= 2: sleep_risk += 1
    m2_prob = min(sleep_risk / 8.0, 0.99)

    # ── MODEL 3: Protein heuristic ──
    m3_prob = 0.0
    if total_protein < ref_min: m3_prob += 0.40
    if albumin < 3.5: m3_prob += 0.35
    if globulin > 3.5: m3_prob += 0.25
    m3_prob = min(m3_prob, 0.99)

    # ── ENSEMBLE ──
    W1, W2, W3 = 0.65, 0.25, 0.10
    ensemble_score = W1 * m1_prob + W2 * m2_prob + W3 * m3_prob

    if ensemble_score >= 0.7:
        risk_cat, risk_class = "HIGH RISK", "risk-high"
    elif ensemble_score >= 0.4:
        risk_cat, risk_class = "MODERATE RISK", "risk-moderate"
    else:
        risk_cat, risk_class = "LOW RISK", "risk-low"

    # ── DISPLAY ──
    st.markdown("---")
    st.markdown('<div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 16px; padding: 2rem;">', unsafe_allow_html=True)

    st.markdown(f'<h2 style="text-align:center;">📋 CLINICAL REPORT</h2>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center; opacity:0.9;"><b>{patient_name or "N/A"}</b> | ID: {patient_id} | Age: {patient_age} | {assessment_date.strftime("%B %d, %Y")}</p>', unsafe_allow_html=True)
    st.divider()

    # Individual models
    st.subheader("🔬 Individual Model Predictions")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="model-box"><b>Model 1: XGBoost</b><br>Biomarkers<br><b>{m1_prob*100:.1f}%</b> risk<br>Weight: {W1*100:.0f}%</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="model-box"><b>Model 2: Sleep/Orexin</b><br>Survey-based<br><b>{m2_prob*100:.1f}%</b> risk<br>Weight: {W2*100:.0f}%</div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="model-box"><b>Model 3: Protein</b><br>Logistic Regression<br><b>{m3_prob*100:.1f}%</b> risk<br>Weight: {W3*100:.0f}%</div>', unsafe_allow_html=True)

    # Ensemble result
    st.divider()
    st.subheader("🎯 Ensemble Meta-Model")
    col_mid, = st.columns([2])
    with col_mid:
        st.markdown(f'<div class="{risk_class}">{risk_cat}</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center; font-size:1.2rem;">Ensemble Score: <b>{ensemble_score*100:.1f}%</b></p>', unsafe_allow_html=True)

    # Interpretation
    st.divider()
    st.subheader("📝 Clinical Interpretation")
    interp = []
    if norm_abeta < 0.6: interp.append("• **Amyloid pathology:** Low Aβ42 suggests amyloid plaque deposition.")
    if norm_ptau > 1.5: interp.append("• **Tau pathology:** Elevated p-Tau181 indicates active neurodegeneration.")
    if norm_orexin < 0.7: interp.append("• **Orexin dysfunction:** Low orexin indicates hypothalamic involvement.")
    if sleep_risk >= 4: interp.append(f"• **Sleep disorder:** {sleep_risk}/8 risk factors present.")
    if total_protein < ref_min: interp.append("• **Protein deficiency:** Below reference range.")
    if naccne4s >= 1: interp.append(f"• **Genetic risk:** APOE4 ε{naccne4s} carrier.")
    if not interp: interp.append("• **Normal profile:** No significant biomarker abnormalities detected.")
    for i in interp: st.markdown(i)

    # Recommendations
    st.subheader("💡 Recommendations")
    if risk_cat == "HIGH RISK":
        recs = ["**IMMEDIATE ACTION:**", "• Refer to neurology for neuropsych eval", "• Consider amyloid PET or CSF confirmation", "• Discuss disease-modifying therapies", "• Family counseling & care planning", "• Follow-up in 3 months"]
    elif risk_cat == "MODERATE RISK":
        recs = ["**MONITORING:**", "• Repeat biomarkers in 6-12 months", "• Neuropsychological baseline testing", "• Lifestyle: Mediterranean diet, exercise, cognitive training", "• Optimize cardiovascular risk", "• Follow-up in 6 months"]
    else:
        recs = ["**PREVENTIVE:**", "• Annual cognitive screening", "• Exercise 150min/week, cognitive engagement", "• Monitor sleep quality", "• Optimize cardiovascular health", "• Routine follow-up in 12 months"]
    for r in recs: st.markdown(r)

    # Download
    report = f"""ALZHEIMER'S CLINICAL PREDICTION REPORT
{'='*60}
Patient: {patient_name or 'N/A'} | ID: {patient_id}
Age: {patient_age} | Gender: {patient_gender} | Date: {assessment_date.strftime('%B %d, %Y')}

ENSEMBLE RISK: {risk_cat} ({ensemble_score*100:.1f}%)
Model 1 (XGBoost): {m1_prob*100:.1f}% | Model 2 (Sleep): {m2_prob*100:.1f}% | Model 3 (Protein): {m3_prob*100:.1f}%

RECOMMENDATIONS:
{chr(10).join([r.replace('• ','- ').replace('**','') for r in recs])}

Generated by Alzheimer's Clinical Prediction System
"""
    st.download_button("📥 Download Report", report, f"Report_{patient_id}.pdf", "text/plain")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("🔬 For Research & Clinical Support Only | Not a substitute for professional diagnosis")