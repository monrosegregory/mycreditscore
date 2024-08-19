import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Constants for categorical options
EDUCATION_OPTIONS = ["Graduate School", "University", "High School", "Others"]
MARRIAGE_OPTIONS = ["Married", "Single", "Others"]
PAY_STATUS_OPTIONS = ["On Time", "1 Month Late", "2 Months Late", "3+ Months Late"]

@st.cache_data
def create_model():
    """Create and train the model."""
    feature_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                     'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                     'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                     'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                     'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    X_train = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
    y_train = np.random.randint(2, size=100)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    
    return model, feature_names

def get_user_input():
    """Collect and return user input as a dictionary."""
    st.header("Input Your Financial Data")

    with st.form(key='user_input_form'):
        col1, col2 = st.columns(2)

        with col1:
            limit_bal = st.number_input("Credit Limit Balance", min_value=0, help="Total amount of credit available to you.")
            sex = st.selectbox("Sex", options=["Male", "Female"], help="Your gender.")
            education = st.selectbox("Education Level", options=EDUCATION_OPTIONS, help="Your highest education level.")
            marriage = st.selectbox("Marital Status", options=MARRIAGE_OPTIONS, help="Your marital status.")
            age = st.slider("Age", min_value=18, max_value=100, help="Your current age.")

        with col2:
            st.subheader("Payment History (Past 7 Months)")
            pay_status = []
            for i in range(7):
                pay_status.append(st.selectbox(f"Payment Status {i+1} Month(s) Ago", options=PAY_STATUS_OPTIONS, index=0))
            
            st.subheader("Billing & Payment Amounts")
            bill_amt = []
            pay_amt = []
            for i in range(1, 7):
                bill_amt.append(st.number_input(f"Bill Amount for Month {i}", min_value=0, help=f"Bill amount for the {i}th last month."))
                pay_amt.append(st.number_input(f"Payment Amount for Month {i}", min_value=0, help=f"Payment amount for the {i}th last month."))

        submit_button = st.form_submit_button(label='Get Prediction')
    
    if submit_button:
        return {
            'LIMIT_BAL': limit_bal,
            'SEX': 1 if sex == "Male" else 2,
            'EDUCATION': EDUCATION_OPTIONS.index(education) + 1,
            'MARRIAGE': MARRIAGE_OPTIONS.index(marriage) + 1,
            'AGE': age,
            'PAY_0': PAY_STATUS_OPTIONS.index(pay_status[0]) - 1,
            'PAY_2': PAY_STATUS_OPTIONS.index(pay_status[1]) - 1,
            'PAY_3': PAY_STATUS_OPTIONS.index(pay_status[2]) - 1,
            'PAY_4': PAY_STATUS_OPTIONS.index(pay_status[3]) - 1,
            'PAY_5': PAY_STATUS_OPTIONS.index(pay_status[4]) - 1,
            'PAY_6': PAY_STATUS_OPTIONS.index(pay_status[5]) - 1,
            'BILL_AMT1': bill_amt[0],
            'BILL_AMT2': bill_amt[1],
            'BILL_AMT3': bill_amt[2],
            'BILL_AMT4': bill_amt[3],
            'BILL_AMT5': bill_amt[4],
            'BILL_AMT6': bill_amt[5],
            'PAY_AMT1': pay_amt[0],
            'PAY_AMT2': pay_amt[1],
            'PAY_AMT3': pay_amt[2],
            'PAY_AMT4': pay_amt[3],
            'PAY_AMT5': pay_amt[4],
            'PAY_AMT6': pay_amt[5],
        }
    return None

def main():
    """Main function to run the Streamlit app."""
    st.title("Credit Scoring AI Prototype")

    st.write("""
    ### Understand Your Credit Risk
    Fill in your financial data below, and this tool will help predict the likelihood of defaulting on credit. 
    Your information will be used to estimate the probability that you may not meet future credit obligations.
    """)

    model, feature_names = create_model()
    
    user_input = get_user_input()

    if user_input:
        user_data = pd.DataFrame(user_input, index=[0])
        
        prob_default = model.predict_proba(user_data)[0][1]
        
        if prob_default < 0.3:
            risk_category = "Low Risk"
            risk_color = "green"
        elif prob_default < 0.7:
            risk_category = "Moderate Risk"
            risk_color = "yellow"
        else:
            risk_category = "High Risk"
            risk_color = "red"
        
        st.subheader(f"Predicted Probability of Default: {prob_default:.2f}")
        st.markdown(f"### Risk Category: <span style='color:{risk_color}'>{risk_category}</span>", unsafe_allow_html=True)
        
        st.write("""
        #### What does this mean?
        - A **Low Risk** score means you are unlikely to default on your credit.
        - A **Moderate Risk** score suggests you might have some risk of defaulting.
        - A **High Risk** score indicates a higher likelihood of defaulting.
        
        We suggest reviewing your payment history and making sure to stay on top of your bills to improve your score.
        """)

if __name__ == "__main__":
    main()
