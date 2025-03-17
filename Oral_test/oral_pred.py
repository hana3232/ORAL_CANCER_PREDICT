import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("survive1.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Oral Cancer Survival Rate Predictor")

# User Inputs
country = st.selectbox("Country", ['Italy', 'Japan', 'UK', 'Sri Lanka', 'South Africa', 'Taiwan', 'USA',
                                   'Germany', 'France', 'Australia', 'Brazil', 'Pakistan', 'Kenya', 
                                   'Russia', 'Nigeria', 'Egypt', 'India'])

age = st.number_input("Age", min_value=1, max_value=120, value=30)

gender = st.selectbox("Gender", ['Female', 'Male'])
tobacco_use = st.selectbox("Tobacco Use", ['Yes', 'No'])
alcohol_consumption = st.selectbox("Alcohol Consumption", ['Yes', 'No'])
hpv_infection = st.selectbox("HPV Infection", ['Yes', 'No'])
betel_quid_use = st.selectbox("Betel Quid Use", ['No', 'Yes'])
chronic_sun_exposure = st.selectbox("Chronic Sun Exposure", ['No', 'Yes'])
poor_oral_hygiene = st.selectbox("Poor Oral Hygiene", ['Yes', 'No'])
diet = st.selectbox("Diet (Fruits & Vegetables Intake)", ['Low', 'High', 'Moderate'])
family_history = st.selectbox("Family History of Cancer", ['No', 'Yes'])
immune_system = st.selectbox("Compromised Immune System", ['No', 'Yes'])
oral_lesions = st.selectbox("Oral Lesions", ['No', 'Yes'])
unexplained_bleeding = st.selectbox("Unexplained Bleeding", ['No', 'Yes'])
difficulty_swallowing = st.selectbox("Difficulty Swallowing", ['No', 'Yes'])
patches = st.selectbox("White or Red Patches in Mouth", ['No', 'Yes'])
tumor_size = st.number_input("Tumor Size (cm)", min_value=0.1, max_value=10.0, value=2.0)
cancer_stage = st.slider("Cancer Stage", 1, 4, 2)
treatment_type = st.selectbox("Treatment Type", ['No Treatment', 'Surgery', 'Radiation', 'Targeted Therapy', 'Chemotherapy'])
early_diagnosis = st.selectbox("Early Diagnosis", ['No', 'Yes'])

# Encode categorical variables as numbers
encoding_dict = {
    "Gender": {'Female': 0, 'Male': 1},
    "Tobacco Use": {'No': 0, 'Yes': 1},
    "Alcohol Consumption": {'No': 0, 'Yes': 1},
    "HPV Infection": {'No': 0, 'Yes': 1},
    "Betel Quid Use": {'No': 0, 'Yes': 1},
    "Chronic Sun Exposure": {'No': 0, 'Yes': 1},
    "Poor Oral Hygiene": {'No': 0, 'Yes': 1},
    "Diet (Fruits & Vegetables Intake)": {'Low': 0, 'Moderate': 1, 'High': 2},
    "Family History of Cancer": {'No': 0, 'Yes': 1},
    "Compromised Immune System": {'No': 0, 'Yes': 1},
    "Oral Lesions": {'No': 0, 'Yes': 1},
    "Unexplained Bleeding": {'No': 0, 'Yes': 1},
    "Difficulty Swallowing": {'No': 0, 'Yes': 1},
    "White or Red Patches in Mouth": {'No': 0, 'Yes': 1},
    "Country": {'Italy': 6, 'Japan': 7, 'UK': 15, 'Sri Lanka': 13, 'South Africa': 12, 'Taiwan': 14, 'USA': 16, 'Germany': 4,
                'France': 3, 'Australia': 0, 'Brazil': 1, 'Pakistan': 10, 'Kenya': 8, 'Russia': 11, 'Nigeria': 9,
                'Egypt': 2, 'India': 5},
    "Treatment Type": {'No Treatment': 0, 'Surgery': 1, 'Radiation': 2, 'Targeted Therapy': 3, 'Chemotherapy': 4},
    "Early Diagnosis": {'No': 0, 'Yes': 1}
}

# Convert user input to numerical format
input_data = pd.DataFrame([[country, age, gender, tobacco_use, alcohol_consumption, hpv_infection,
                            betel_quid_use, chronic_sun_exposure, poor_oral_hygiene, diet,
                            family_history, immune_system, oral_lesions, unexplained_bleeding,
                            difficulty_swallowing, patches, tumor_size, cancer_stage, treatment_type,
                            early_diagnosis]],
                          columns=["Country", "Age", "Gender", "Tobacco Use", "Alcohol Consumption", 
                                   "HPV Infection", "Betel Quid Use", "Chronic Sun Exposure", 
                                   "Poor Oral Hygiene", "Diet (Fruits & Vegetables Intake)", 
                                   "Family History of Cancer", "Compromised Immune System", "Oral Lesions", 
                                   "Unexplained Bleeding", "Difficulty Swallowing", "White or Red Patches in Mouth", 
                                   "Tumor Size (cm)", "Cancer Stage", "Treatment Type", "Early Diagnosis"])

# Apply encoding
for col in encoding_dict.keys():
    input_data[col] = input_data[col].map(encoding_dict[col])

# Convert to NumPy array for prediction
features = input_data.to_numpy()

# Predict survival rate
if st.button("Predict Survival Rate"):
    prediction = model.predict(features)
    st.success(f"Predicted 5-Year Survival Rate: {prediction[0]:.2f}%")
