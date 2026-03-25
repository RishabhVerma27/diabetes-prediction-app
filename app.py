import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Page config
st.set_page_config(page_title="Diabetes AI Predictor", page_icon="🩺", layout="wide")

# Custom CSS (Premium UI)
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
.stButton>button {
    background-color: #00c6ff;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("AI Diabetes Risk Predictor")
st.markdown("### Smart prediction using Machine Learning")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
This AI tool predicts diabetes risk using health data.

✔ Built with Machine Learning  
⚠ Not a medical diagnosis  
""")

# Load data
data = pd.read_csv("diabetes.csv")

# Features & target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model (better)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")
    pregnancies = st.number_input("Pregnancy Count", 0, 20)
    age = st.number_input("Age", 1, 120)
    family_history = st.selectbox("Family History", ["No", "Yes"])

with col2:
    st.subheader("Health Details")
    glucose = st.number_input("Sugar Level", 0, 200)
    bp = st.number_input("Blood Pressure", 0, 150)
    bmi = st.number_input("BMI", 0.0, 70.0)

# Optional
st.subheader("Optional")
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)

family_history = 1 if family_history == "Yes" else 0

# Predict
st.markdown("---")
if st.button("🔍 Predict"):
    input_data = [[pregnancies, glucose, bp, skin, insulin, bmi, family_history, age]]
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("## 🧾 Result")

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({(1-probability)*100:.2f}%)")

# Show accuracy
st.markdown("---")
st.info(f"📈 Model Accuracy: {accuracy*100:.2f}%")

# Footer
st.caption("Built by Rishabh Verma")
