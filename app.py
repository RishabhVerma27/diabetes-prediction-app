import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺")

# Title
st.title("Diabetes Risk Prediction")
st.write("Enter your health details below to check your diabetes risk.")
st.info("This tool is for educational purposes only. Please consult a doctor for medical advice.")

# Load dataset
data = pd.read_csv("diabetes.csv")

# Prepare data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Fill missing values
X.fillna(X.mean(), inplace=True)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# User Inputs
st.header("Enter Your Details:")

pregnancies = st.number_input("Number of times pregnant (0 if none)", 0, 20)
st.caption("Enter 0 if not applicable.")

glucose = st.number_input("Sugar Level (Glucose)", 0, 200)
st.caption("Indicates sugar level in your blood.")

bp = st.number_input("Blood Pressure (mm Hg)", 0, 150)

skin = st.number_input("Skin Thickness (optional)", 0, 100)
insulin = st.number_input("Insulin Level (optional)", 0, 900)

bmi = st.number_input("Body Mass Index (BMI)", 0.0, 70.0)
st.caption("BMI = weight(kg) / height(m)^2")

family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

age = st.number_input("Age", 1, 120)

# Convert family history
family_history = 1 if family_history == "Yes" else 0

# Predict button
if st.button("Predict"):
    input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, family_history, age]]
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    st.subheader("Result:")

    if prediction[0] == 1:
        st.error("⚠️ You may have a higher risk of diabetes. Please consult a doctor.")
    else:
        st.success("✅ You have a lower risk of diabetes based on the data.")

# Footer
st.write("---")
st.write("Made by Rishabh Verma")