import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    df = pd.read_csv("dummy_npi_data.xlsx - Dataset.csv")

    # Convert login/logout times to datetime
    df["Login Time"] = pd.to_datetime(df["Login Time"])
    df["Logout Time"] = pd.to_datetime(df["Logout Time"])

    df["Login Hour"] = df["Login Time"].dt.hour
    df["Logout Hour"] = df["Logout Time"].dt.hour

    # Encode categorical features
    label_encoders = {}
    categorical_cols = ["State", "Region", "Speciality"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  

    # Create training data for all hours (0-23)
    training_data = []
    for hour in range(24):
        temp_df = df.copy()
        temp_df["Target Hour"] = hour
        temp_df["Active"] = ((temp_df["Login Hour"] <= hour) & (temp_df["Logout Hour"] >= hour)).astype(int)
        training_data.append(temp_df)

    # Combine all hours data
    final_df = pd.concat(training_data)
    final_df.to_csv("final_training_data.csv", index=False)

    # Select features and target
    X = final_df[["State", "Region", "Speciality", "Login Hour", "Logout Hour", "Usage Time (mins)", "Count of Survey Attempts", "Target Hour"]]
    y = final_df["Active"]

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

   
    joblib.dump(model, "doctor_activity_model.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

    print("Model training complete and saved!")

if __name__ == "__main__":
    train_and_save_model()