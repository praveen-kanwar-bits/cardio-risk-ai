import streamlit as st
import pandas as pd
from utils import predict_heart_disease

st.set_page_config(page_title="CardioRisk-AI: Heart Disease Predictor", layout="centered")

st.title("CardioRisk-AI 🫀")

st.markdown("""
This system is an AI-powered clinical decision-support tool. It predicts the likelihood of heart disease based on common demographic and clinical parameters.
""")

st.warning("⚠️ **Disclaimer:** This system is for educational decision-support demonstration only and is not a replacement for medical diagnosis. Please consult a qualified medical professional for actual diagnosis.")

with st.expander("ℹ️ Click here for Clinical Feature Descriptions"):
    st.markdown("""
    - **Age**: Age of the patient in years
    - **Sex**: 1 = Male; 0 = Female
    - **Chest Pain Type (cp)**: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-anginal Pain, 3 = Asymptomatic
    - **Resting Blood Pressure (trestbps)**: Resting blood pressure (in mm Hg on admission to the hospital)
    - **Cholesterol (chol)**: Serum cholestoral in mg/dl
    - **Fasting Blood Sugar (fbs)**: 1 = if > 120 mg/dl; 0 = otherwise
    - **Rest ECG (restecg)**: 0 = Normal, 1 = Having ST-T wave abnormality, 2 = Showing probable or definite left ventricular hypertrophy
    - **Max Heart Rate (thalach)**: Maximum heart rate achieved during exercise
    - **Exercise Angina (exang)**: Exercise induced angina (1 = yes; 0 = no)
    - **Oldpeak**: ST depression induced by exercise relative to rest
    - **Slope**: The slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
    - **CA**: Number of major vessels (0-4) colored by flourosopy
    - **Thal**: 0 = normal; 1 = fixed defect; 2 = reversable defect
    """)

st.header("Patient Data Entry")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 100, value=55)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=130)
    chol = st.number_input("Cholesterol (mg/dl)", value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [0, 1])
    restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Angina (1=Yes, 0=No)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", value=1.0)
    slope = st.selectbox("Slope (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal (0-2)", [0, 1, 2], index=2)

st.markdown("---")

if st.button("Predict Risk", type="primary"):
    # Note: Column names must match the training set exactly. 
    # utils.py validate_input will ensure the correct order based on feature_columns.pkl
    data_dict = {
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
        'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
        'slope': [slope], 'ca': [ca], 'thal': [thal]
    }
    data = pd.DataFrame(data_dict)

    try:
        result, probability = predict_heart_disease(data)

        if result == 1:
            st.error("🚨 **Prediction: High Risk of Heart Disease**")
            if probability is not None:
                st.write(f"**Confidence / Probability:** {probability * 100:.1f}%")
            st.write("Interpretation: The model indicates a high likelihood of heart disease based on the provided clinical parameters. Further medical evaluation is strongly recommended.")
        else:
            st.success("✅ **Prediction: Low Risk of Heart Disease**")
            if probability is not None:
                st.write(f"**Confidence / Probability:** {probability * 100:.1f}%")
            st.write("Interpretation: The model indicates a low likelihood of heart disease. Maintain a healthy lifestyle.")
            
        st.info("🔒 Note: The data you entered was processed in memory and has not been saved or stored in any database.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")