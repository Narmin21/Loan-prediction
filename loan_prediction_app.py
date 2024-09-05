import streamlit as st
import joblib  # ya da pickle
import numpy as np

# Modeli yükləyin
xgb_model = joblib.load('xgb_model.pkl')

# Streamlit interfeysi üçün başlıq
st.title("Loan Approval Prediction Using XGBoost")

# İstifadəçidən inputları toplamaq
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Income ($)", min_value=0, max_value=1000000, value=50000)
loan_amount = st.number_input("Loan Amount ($)", min_value=0, max_value=1000000, value=10000)
credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650)
months_employed = st.number_input("Months Employed", min_value=0, max_value=480, value=12)
num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, max_value=20, value=5)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_term = st.number_input("Loan Term (months)", min_value=0, max_value=360, value=60)
dti_ratio = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=100.0, value=20.0)

# Education inputunu bir dəyişən kimi qəbul edin
education = st.radio("Education Level", options=['High School', 'Bachelor', 'Master', 'PhD'])

# Təhsili rəqəmlərlə kodlaşdırın
education_dict = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
education_value = education_dict[education]

# **Income Bin üçün Radio Button**
income_bin = st.radio("Income Bin", options=['Low', 'Medium', 'High'])

# Income Bin-i tək bir input kimi qəbul edin
income_bin_dict = {'Low': 0, 'Medium': 1, 'High': 2}
income_bin_value = income_bin_dict[income_bin]

lti_ratio = st.number_input("Loan-to-Income Ratio (LTI)", min_value=0.0, max_value=100.0, value=10.0)
credit_utilization = st.number_input("Credit Utilization (%)", min_value=0.0, max_value=100.0, value=30.0)

# Employment type (one-hot encoded)
employment_type = st.selectbox("Employment Type", options=['Full-time', 'Part-time', 'Self-employed', 'Unemployed'])
employment_type_part_time = 1 if employment_type == 'Part-time' else 0
employment_type_self_employed = 1 if employment_type == 'Self-employed' else 0
employment_type_unemployed = 1 if employment_type == 'Unemployed' else 0

# Marital status (one-hot encoded)
marital_status = st.selectbox("Marital Status", options=['Married', 'Single'])
marital_status_married = 1 if marital_status == 'Married' else 0
marital_status_single = 1 if marital_status == 'Single' else 0

# Has Mortgage (binary)
has_mortgage = st.selectbox("Has a Mortgage", options=['Yes', 'No'])
has_mortgage_yes = 1 if has_mortgage == 'Yes' else 0

# Has Dependents (binary)
has_dependents = st.selectbox("Has Dependents", options=['Yes', 'No'])
has_dependents_yes = 1 if has_dependents == 'Yes' else 0

# Loan Purpose (one-hot encoded)
loan_purpose = st.selectbox("Loan Purpose", options=['Business', 'Education', 'Home', 'Other'])
loan_purpose_business = 1 if loan_purpose == 'Business' else 0
loan_purpose_education = 1 if loan_purpose == 'Education' else 0
loan_purpose_home = 1 if loan_purpose == 'Home' else 0
loan_purpose_other = 1 if loan_purpose == 'Other' else 0

# İstifadəçi məlumatlarını bir araya gətirək
input_data = np.array([age, income, loan_amount, credit_score, months_employed, num_credit_lines, 
                       interest_rate, loan_term, dti_ratio, education_value,  # Təhsil bir input kimi daxil edilir
                       income_bin_value,  # Income Bin bir input kimi daxil edilir
                       lti_ratio, credit_utilization, employment_type_part_time, 
                       employment_type_self_employed, employment_type_unemployed, 
                       marital_status_married, marital_status_single, has_mortgage_yes, 
                       has_dependents_yes, loan_purpose_business, loan_purpose_education, 
                       loan_purpose_home, loan_purpose_other]).reshape(1, -1)

# Proqnoz düyməsi
if st.button("Predict"):
    prediction = xgb_model.predict(input_data)
    
    # Nəticəni göstərmək
    st.write(f"Prediction: {'Approved' if prediction[0] == 1 else 'Denied'}")
