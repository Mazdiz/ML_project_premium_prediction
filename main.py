import streamlit as st
import pandas as pd
from prediction_helper import predict

st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")
st.title("🩺 Health Insurance Cost Predictor")
st.write("Fill in the details below to proceed.")

# ... all your columns UI ...

# --- Row 1 ---
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
with col2:
    gender = st.selectbox("Gender", ['Male', 'Female'])
with col3:
    marital_status = st.selectbox("Marital Status", ['Unmarried', 'Married'])

# --- Row 2 ---
col4, col5, col6 = st.columns(3)
with col4:
    region = st.selectbox("Region", ['Northwest', 'Southeast', 'Northeast', 'Southwest'])
with col5:
    dependants = st.number_input("Number of Dependants", min_value=0, max_value=20, step=1)
with col6:
    bmi_category = st.selectbox("BMI Category", ['Normal', 'Obesity', 'Overweight', 'Underweight'])

# --- Row 3 ---
col7, col8, col9 = st.columns(3)
with col7:
    smoking_status = st.selectbox("Smoking Status", ['No Smoking', 'Occasional', 'Regular'])
with col8:
    employment_status = st.selectbox("Employment Status", ['Salaried', 'Self-Employed', 'Freelancer'])
with col9:
    genetical_risk = st.number_input("Genetical Risk", min_value=0, max_value=20, step=1)

# --- Row 4 ---
col10, col11, col12 = st.columns(3)
with col10:
    income_lakhs = st.number_input("Income (in Lakhs)", min_value=0.0, max_value=200.0, step=0.5)
with col11:
    medical_history = st.selectbox(
        "Medical History",
        [
            'No Disease',
            'Diabetes',
            'High blood pressure',
            'Thyroid',
            'Heart disease',
            'Diabetes & High blood pressure',
            'High blood pressure & Heart disease',
            'Diabetes & Thyroid',
            'Diabetes & Heart disease'
        ]
    )
with col12:
    insurance_plan = st.selectbox("Insurance Plan", ['Bronze', 'Silver', 'Gold'])


# Now all variables like age, gender, etc. exist, so we can safely create the DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Region': region,
    'Marital_status': marital_status,
    'Number Of Dependants': dependants,
    'BMI_Category': bmi_category,
    'Smoking_Status': smoking_status,
    'Employment_Status': employment_status,
    'Income_Lakhs': income_lakhs,
    'Medical History': medical_history,
    'Insurance_Plan': insurance_plan,
    'genetical_risk': genetical_risk
}])

#Predict Button
predict_clicked = st.button("Predict")

if predict_clicked:
    prediction_value = predict(input_data)
    st.success(f"Predicted Insurance Premium: {prediction_value:.2f}")