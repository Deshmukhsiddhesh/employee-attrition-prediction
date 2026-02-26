import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("attrition_model.pkl")

st.title("Employee Attrition Predictor")

st.write("Fill employee details:")

# -------- USER INPUTS -------- #

age = st.slider("Age", 18, 60)
monthly_income = st.number_input("Monthly Income", 1000, 200000)
years_at_company = st.slider("Years at Company", 0, 40)
job_satisfaction = st.slider("Job Satisfaction", 1, 4)
environment_satisfaction = st.slider("Environment Satisfaction", 1, 4)
work_life_balance = st.slider("Work Life Balance", 1, 4)
training_times = st.slider("Training Times Last Year", 0, 10)

overtime = st.selectbox("OverTime", ["Yes", "No"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.selectbox("Job Role", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative",
    "Manager", "Sales Representative", "Research Director", "Human Resources"
])

business_travel = st.selectbox("Business Travel", [
    "Travel_Rarely", "Travel_Frequently", "Non-Travel"
])

gender = st.selectbox("Gender", ["Male", "Female"])

education = st.slider("Education Level", 1, 5)

distance_from_home = st.slider("Distance From Home", 1, 30)

# -------- CREATE FULL INPUT DATAFRAME -------- #

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Encode dataset exactly like training
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# create empty row
input_data = df.drop("Attrition", axis=1).iloc[:1].copy()

# replace fields
input_data["Age"] = age
input_data["MonthlyIncome"] = monthly_income
input_data["YearsAtCompany"] = years_at_company
input_data["JobSatisfaction"] = job_satisfaction
input_data["EnvironmentSatisfaction"] = environment_satisfaction
input_data["WorkLifeBalance"] = work_life_balance
input_data["TrainingTimesLastYear"] = training_times
input_data["OverTime"] = 1 if overtime == "Yes" else 0
input_data["Education"] = education
input_data["DistanceFromHome"] = distance_from_home

# -------- ENCODE SELECTBOX VALUES -------- #

# Map text to dataset encoding
input_data["MaritalStatus"] = df["MaritalStatus"].mode()[0]
input_data["Department"] = df["Department"].mode()[0]
input_data["JobRole"] = df["JobRole"].mode()[0]
input_data["BusinessTravel"] = df["BusinessTravel"].mode()[0]
input_data["Gender"] = df["Gender"].mode()[0]

# -------- PREDICTION -------- #

if st.button("Predict Attrition"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("Employee likely to leave")
    else:
        st.success("Employee likely to stay")

    st.write("Probability of leaving:", round(probability[0][1]*100, 2), "%")