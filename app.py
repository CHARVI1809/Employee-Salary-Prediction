import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼", layout="centered")

st.title("💼Employee Salary Prediction App")
st.markdown("Predict the salary of an employee")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 50, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education Level", [ "High School","Bachelor's", "Master's", "PhD"])
job_title = st.sidebar.selectbox("Job Title", [
    "Account Manager", "Accountant", "Administrative Assistant", "Back end Developer", "Business Analyst",
"Business Development Manager", "Business Intelligence Analyst", "Chief Data Officer", "Content Marketing Manager",
"Copywriter", "Creative Director", "Customer Service Manager", "Customer Service Rep", "Customer Service Representative",
"Customer Success Manager", "Customer Success Rep", "Data Analyst", "Data Entry Clerk", "Data Scientist", 
"Delivery Driver", "Digital Content Producer", "Digital Marketing Manager", "Digital Marketing Specialist", 
"Director of Business Development", "Director of Data Science", "Director of Engineering", "Director of Finance",
"Director of HR", "Director of Marketing", "Director of Operations", "Director of Product Management", "Director of Sales",
"Director of Sales and Marketing", "Event Coordinator", "Financial Advisor", "Financial Analyst", "Financial Manager",
"Front End Developer", "Front end Developer", "Full Stack Engineer", "Graphic Designer", "Help Desk Analyst", 
"HR Generalist", "HR Manager", "Human Resources Coordinator", "Human Resources Director", "Human Resources Manager",
"IT Manager", "IT Support", "IT Support Specialist", "Juniour HR Coordinator", "Juniour HR Generalist",
"Junior Account Manager", "Junior Accountant", "Junior Advertising Coordinator", "Junior Business Analyst",
"Junior Business Development Associate", "Junior Business Operations Analyst", "Junior Copywriter",
"Junior Customer Support Specialist", "Junior Data Analyst", "Junior Data Scientist", "Junior Designer", 
"Junior Developer", "Junior Financial Advisor", "Junior Financial Analyst", "Junior HR Coordinator", 
"Junior HR Generalist", "Junior Marketing Analyst", "Junior Marketing Coordinator", "Junior Marketing Manager",
"Junior Marketing Specialist", "Junior Operations Analyst", "Junior Operations Coordinator", "Junior Operations Manager", 
"Junior Product Manager", "Junior Project Manager", "Junior Recruiter", "Junior Research Scientist",
"Junior Sales Associate", "Junior Sales Representative", "Junior Social Media Manager", "Junior Social Media Specialist",
"Junior Software Developer", "Junior Software Engineer", "Junior UX Designer", "Junior Web Designer", 
"Junior Web Developer", "Marketing Analyst", "Marketing Coordinator", "Marketing Director", "Marketing Manager", 
"Marketing Specialist", "Network Engineer", "Office Manager", "Operations Analyst", "Operations Director", "Operations Manager",
"Principal Engineer", "Principal Scientist", "Product Designer", "Product Manager", "Product Marketing Manager", 
"Project Engineer", "Project Manager", "Public Relations Manager", "Receptionist", "Recruiter", "Research Director",
"Research Scientist", "Sales Associate", "Sales Director", "Sales Executive", "Sales Manager", "Sales Operations Manager", 
"Sales Representative", "Senior Accountant", "Senior Account Executive", "Senior Account Manager",
"Senior Advertising Coordinator", "Senior Analyst", "Senior Business Analyst", "Senior Business Development Manager",
"Senior Consultant", "Senior Copywriter", "Senior Customer Service Manager", "Senior Data Analyst", "Senior Data Engineer",
"Senior Data Scientist", "Senior Designer", "Senior Developer", "Senior Engineer", "Senior Financial Advisor", 
"Senior Financial Analyst", "Senior Financial Manager", "Senior Graphic Designer", "Senior HR Generalist", "Senior HR Manager",
"Senior HR Specialist", "Senior Human Resources Coordinator", "Senior Human Resources Manager", "Senior Human Resources Specialist",
"Senior IT Consultant", "Senior IT Project Manager", "Senior IT Support Specialist", "Senior Manager", "Senior Marketing Analyst",
"Senior Marketing Coordinator", "Senior Marketing Director", "Senior Marketing Manager", "Senior Marketing Specialist", 
"Senior Operations Analyst", "Senior Operations Coordinator", "Senior Operations Manager", "Senior Product Designer",
"Senior Product Development Manager", "Senior Product Manager", "Senior Product Marketing Manager", 
"Senior Project Coordinator", "Senior Project Engineer", "Senior Project Manager", "Senior Quality Assurance Analyst", 
"Senior Research Scientist", "Senior Researcher", "Senior Sales Manager", "Senior Sales Representative", 
"Senior Scientist", "Senior Software Architect", "Senior Software Developer", "Senior Software Engineer", 
"Senior Software Manager", "Senior Strategy Consultant", "Senior Training Specialist", "Senior UX Designer", 
"Social Media Man", "Social Media Manager", "Social Media Specialist", "Software Developer", "Software Engineer",
"Software Engineer Manager", "Software Manager", "Software Project Manager", "Strategy Consultant", "Supply Chain Analyst",
"Supply Chain Manager", "Technical Recruiter", "Technical Support Specialist", "Technical Writer",
"Training Specialist", "UX Designer", "UX Researcher", "VP of Finance", "VP of Operations", "Web Developer"

])
experience = st.sidebar.slider("Years of Experience", 0, 23, 5)

# Build input DataFrame
input_df = pd.DataFrame({
    'Age': [age],
    'Gender': [gender],
    'Education Level': [education],
    'Job Title': [job_title],
    'Years of Experience': [experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_data = batch_data.dropna()
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
