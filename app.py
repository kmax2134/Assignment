import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

def predict_active_doctors(input_time, num_rows=10):
    """Predict doctors likely to be active at the given input time."""
    try:
        # Load trained model and encoders
        model = joblib.load("doctor_activity_model.pkl")
        label_encoders = joblib.load("label_encoders.pkl")

    
        df = pd.read_csv("dummy_npi_data.xlsx - Dataset.csv")

   
        df["Login Time"] = pd.to_datetime(df["Login Time"])
        df["Logout Time"] = pd.to_datetime(df["Logout Time"])

       
        df["Login Hour"] = df["Login Time"].dt.hour
        df["Logout Hour"] = df["Logout Time"].dt.hour

        # Extract target hour from input time
        target_hour = int(input_time.split(":")[0])

        # Create input features for prediction
        X_pred = df.copy()
        X_pred["Target Hour"] = target_hour

        # Encode categorical features
        categorical_cols = ["State", "Region", "Speciality"]
        for col in categorical_cols:
            le = label_encoders[col]
            X_pred[col] = le.transform(X_pred[col])

        # Predict probability of being active
        X_pred["Probability"] = model.predict_proba(X_pred[["State", "Region", "Speciality", "Login Hour", "Logout Hour", "Usage Time (mins)", "Count of Survey Attempts", "Target Hour"]])[:, 1]

        # Filter & sort doctors based on probability & survey attempts
        active_doctors = X_pred[X_pred["Probability"] > 0.5]
        active_doctors = active_doctors.sort_values(by=["Probability", "Count of Survey Attempts"], ascending=False)

        # Limit results to user-specified number
        active_doctors = active_doctors.head(num_rows)

        # Decode categorical values back to original form
        for col in categorical_cols:
            active_doctors[col] = label_encoders[col].inverse_transform(active_doctors[col])

        return active_doctors[["NPI", "State", "Speciality", "Region", "Count of Survey Attempts", "Probability"]]

    except Exception as e:
        st.error(f"Error: {e}")

def main():
    st.title("Doctor Survey")
    st.write("Enter the time and number of doctors to retrieve the most likely active doctors.")

    input_time = st.text_input("Enter time (HH:MM):", "12:00")
    num_doctors = st.number_input("Enter number of doctors to retrieve:", min_value=1, value=10)

    if st.button("Predict"):
        result = predict_active_doctors(input_time, num_doctors)
        st.write("Predicted Active Doctors:")
        st.dataframe(result)

if __name__ == "__main__":
    main()