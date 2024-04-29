import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('highest_accuracy_model1_knn.pkl', 'rb'))

# Function to predict stroke based on input features
def predict_stroke(features):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]
    return prediction, probability

# Create a Streamlit web app
def main():
    # Set app title and description
    st.title("Gravis")
    st.write("Enter the required information to predict the likelihood of stroke.")

    # Create input fields for user to enter information
    age = st.number_input("Age", min_value=1, max_value=100, value=30)
    gender = st.selectbox("Gender", ("Male", "Female"))  # Moved "Gender" closer to "Age"
    hypertension = st.selectbox("Hypertension", ("Yes", "No"))
    heart_disease = st.selectbox("Heart Disease", ("Yes", "No"))
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=80.0)
    bmi = st.number_input("BMI", min_value=0.0, value=20.0)
    smoking_status = st.selectbox("Smoking Status", ("Unknown", "Formerly Smoked", "Never Smoked", "Smokes"))
    ever_married = st.selectbox("Ever Married", ("Yes", "No"))

    # Convert categorical inputs to numerical values
    hypertension = 1 if hypertension == "Yes" else 0
    heart_disease = 1 if heart_disease == "Yes" else 0
    gender = 1 if gender == "Male" else 0
    smoking_status = 0 if smoking_status == "Never Smoked" else 1  # Consider never smoked as non-smoker
    ever_married = 1 if ever_married == "Yes" else 0

    # Create a button to predict stroke
    if st.button("Predict Stroke"):
        # Gather input features
        features = [age, hypertension, heart_disease, avg_glucose_level, bmi, gender, smoking_status, ever_married]

        # Predict stroke and probability
        prediction, probability = predict_stroke(features)

        # Display the prediction
        if prediction[0] == 0:
            st.write("Congratulations! You have a low risk of stroke.")
        else:
            st.write("Warning! You are at a high risk of stroke.")
            st.write("Probability of stroke:", probability)

# Run the web app
if __name__ == "__main__":
    main()
