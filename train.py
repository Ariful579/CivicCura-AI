import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import datetime
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="CivicCura - Offline Retinal Diagnostic Network",
    page_icon="👁️",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0px; }
    .sub-title { font-size: 1.0rem; color: #64748B; margin-bottom: 20px; }
    .risk-banner { padding: 18px 24px; border-radius: 10px; color: white; font-weight: bold; text-align: center; margin-top: 15px; margin-bottom: 20px; }
    .risk-low { background-color: #10B981; }
    .risk-moderate { background-color: #F59E0B; }
    .risk-high { background-color: #EF4444; }
    @keyframes pulse-red {
        0% { background-color: #DC2626; box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }
        70% { background-color: #991B1B; box-shadow: 0 0 0 12px rgba(220, 38, 38, 0); }
        100% { background-color: #DC2626; box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
    }
    .risk-critical { animation: pulse-red 1.5s infinite; font-size: 1.3rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">CivicCura 👁️</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Offline AI Primary Retinal Diagnostic Network</p>', unsafe_allow_html=True)

@st.cache_resource
def load_diagnostic_model():
    return tf.keras.models.load_model('civiccura_mobilenet.h5')

try:
    model = load_diagnostic_model()
    st.success("⚡ Offline Diagnostic Engine Loaded & Ready")
except Exception:
    st.error("Model file 'civiccura_mobilenet.h5' not found. Please run train.py first.")

# Patient Information Inputs
with st.expander("👤 Patient Information (Optional)", expanded=False):
    patient_id = st.text_input("Patient ID / Ref Number", value="PAT-001")
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=45)

uploaded_file = st.file_uploader("Upload Retinal Scan (Fundus Image)", type=["png", "jpg", "jpeg"])

def generate_pdf(pid, age, diagnosis, severity, confidence, action):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 750, "CivicCura Primary Diagnostic Report")
    c.setFont("Helvetica", 10)
    c.drawString(50, 735, f"Generated Offline: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(50, 725, 550, 725)
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 690, f"Patient ID: {pid}")
    c.drawString(50, 670, f"Patient Age: {age}")
    
    c.drawString(50, 630, f"Diagnostic Result: {diagnosis}")
    c.drawString(50, 610, f"Risk Severity: {severity}")
    c.drawString(50, 590, f"AI Confidence Score: {confidence:.2f}%")
    
    c.setFont("Helvetica-Bold", 11)
    c.drawString(50, 550, "Recommended Next Steps / Clinical Guidance:")
    c.setFont("Helvetica", 10)
    c.drawString(50, 530, action)
    
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, "CivicCura AI Screening Tool - For Primary Triaging Only. Consult an Ophthalmologist.")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retinal Scan", use_container_width=True)
    
    if st.button("Run AI Diagnostics", type="primary"):
        with st.spinner("Analyzing retinal microvasculature offline..."):
            size = (224, 224)
            image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            img_array = np.asarray(image_resized) / 255.0
            img_reshape = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_reshape)
            class_idx = int(np.argmax(predictions))
            confidence = float(np.max(predictions) * 100)
            
            diagnosis_info = {
                0: {"class": "Class 0: No DR", "severity": "NORMAL / NO RISK", "level": 5, "css": "risk-low", "icon": "🟢", "action": "Routine annual eye exam recommended."},
                1: {"class": "Class 1: Mild DR", "severity": "LOW RISK", "level": 25, "css": "risk-low", "icon": "🟢", "action": "Follow up with an ophthalmologist in 6-12 months."},
                2: {"class": "Class 2: Moderate DR", "severity": "MODERATE RISK", "level": 50, "css": "risk-moderate", "icon": "⚠️", "action": "Schedule specialist evaluation within 3-6 months."},
                3: {"class": "Class 3: Severe DR", "severity": "HIGH RISK", "level": 75, "css": "risk-high", "icon": "🚨", "action": "Urgent referral to an ophthalmologist required."},
                4: {"class": "Class 4: Proliferative DR", "severity": "CRITICAL RISK", "level": 100, "css": "risk-critical", "icon": "⚡", "action": "Immediate medical intervention required to prevent vision loss."}
            }
            
            result = diagnosis_info[class_idx]
            
            st.divider()
            st.markdown(f'<div class="risk-banner {result["css"]}">{result["icon"]} DIAGNOSTIC ALERT: {result["severity"]}</div>', unsafe_allow_html=True)
            
            st.write("**Severity Level Index:**")
            st.progress(result["level"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Classification", result["class"])
            with col2:
                st.metric("Model Confidence", f"{confidence:.2f}%")
                
            st.info(f"**Clinical Guidance:** {result['action']}")
            
            # PDF Report Download Feature
            pdf_bytes = generate_pdf(patient_id, patient_age, result["class"], result["severity"], confidence, result["action"])
            st.download_button(
                label="📄 Download Offline Medical PDF Report",
                data=pdf_bytes,
                file_name=f"CivicCura_Report_{patient_id}.pdf",
                mime="application/pdf"
            )