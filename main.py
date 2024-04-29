import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
model_file = 'knn_model.pkl'
with open(model_file, 'rb') as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Function to make prediction
def predict_stroke(input_data):
    scaled_input = preprocess_input(input_data)
    prediction = model.predict(scaled_input)
    return prediction

# Streamlit UI
st.title('Gravis')

# Input fields
st.write('Enter Patient Information:')
gender = st.radio('Gender:', ['Male', 'Female'])
age = st.slider('Age:', min_value=0, max_value=100, value=50, step=1)
bmi = st.number_input('BMI:', min_value=10.0, max_value=60.0, value=25.0, step=0.1)
hypertension = st.selectbox('Hypertension:', ['No', 'Yes'])
heart_disease = st.selectbox('Heart Disease:', ['No', 'Yes'])
avg_glucose_level = st.number_input('Average Glucose Level:', min_value=50.0, max_value=300.0, value=100.0, step=0.1)
ever_married = st.selectbox('Ever Married:', ['No', 'Yes'])
work_type = st.selectbox('Work Type:', ['Private', 'Self-employed', 'Govt_job', 'Never_worked'])
residence_type = st.selectbox('Residence Type:', ['Urban', 'Rural'])
smoking_status = st.selectbox('Smoking Status:', ['never smoked', 'formerly smoked', 'smokes'])

# Convert categorical input to numerical
gender_val = 1 if gender == 'Male' else 0
hypertension_val = 1 if hypertension == 'Yes' else 0
heart_disease_val = 1 if heart_disease == 'Yes' else 0
ever_married_val = 1 if ever_married == 'Yes' else 0

work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Never_worked': 4}
work_type_val = work_type_mapping[work_type]

residence_type_val = 1 if residence_type == 'Urban' else 0

smoking_status_mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2}
smoking_status_val = smoking_status_mapping[smoking_status]

# Predict button
if st.button('Predict Stroke'):
    input_data = {
        'Gender': [gender_val],
        'Age': [age],
        'BMI': [bmi],
        'Hypertension': [hypertension_val],
        'HeartDisease': [heart_disease_val],
        'AverageGlucoseLevel': [avg_glucose_level],
        'EverMarried': [ever_married_val],
        'WorkType': [work_type_val],
        'ResidenceType': [residence_type_val],
        'SmokingStatus': [smoking_status_val]
    }
    input_df = pd.DataFrame(input_data)
    prediction = predict_stroke(input_df)
    if prediction[0] == 1:
        st.error("You have high risk of stroke. Please seek advice from Heathcare professional")
    else:
        st.success("Congratulations you have low risk of stroke")
