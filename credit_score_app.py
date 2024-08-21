import streamlit as st
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Sample trained logistic regression model
feature_names = ['Total Credit Available', 'Gender', 'Education Level', 'Marital Status', 'Age',
                 'Recent Payment Status', 'Payment Status 2 Months Ago', 'Payment Status 3 Months Ago',
                 'Bill Amount 1', 'Bill Amount 2', 'Bill Amount 3', 'Bill Amount 4',
                 'Bill Amount 5', 'Bill Amount 6', 'Payment Amount 1', 'Payment Amount 2', 
                 'Payment Amount 3', 'Payment Amount 4', 'Payment Amount 5', 'Payment Amount 6']

# Creating a dummy logistic regression model
X_train = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
y_train = np.random.randint(2, size=100)

# Create a pipeline that scales the data then applies logistic regression
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)

# Streamlit UI
st.title("Your Personalized Credit Risk Report")
st.write("Understand your credit risk and what factors are affecting it. This tool will help you see where you stand and offer suggestions on how to improve your credit score.")

# Transparency Section: Explain How the Model Works
st.header("How This Tool Works")
st.write("""
This tool uses a statistical model called **Logistic Regression** to predict your likelihood of defaulting on a credit payment.
We analyze various factors like your total available credit, payment history, and demographic information to estimate this risk.
""")

st.write("**Data Privacy:** We respect your privacy. Your data is used solely for this prediction and is not stored or shared.")

# User Inputs with Guided Steps
st.header("Step 1: Your Basic Information")

# Mapping for Gender
gender_options = {
    "Male": 1,
    "Female": 2
}

# Mapping for Education Level
education_options = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Other": 4
}

# Mapping for Marital Status
marriage_options = {
    "Married": 1,
    "Single": 2,
    "Other": 3
}

with st.expander("Personal Information"):
    st.markdown("**Total Credit Available**: The maximum amount of credit you can use across all your accounts.")
    total_credit_available = st.number_input("Total Credit Available", min_value=0, help="Enter the total credit limit available to you across all your credit cards and lines of credit.")
    
    gender = st.selectbox("Gender", options=list(gender_options.keys()), help="Select your gender.")
    education = st.selectbox("Education Level", options=list(education_options.keys()), 
                             help="Select your highest level of education.")
    marital_status = st.selectbox("Marital Status", options=list(marriage_options.keys()), help="Select your marital status.")
    age = st.number_input("Your Age", min_value=18, max_value=100, help="Enter your age.")

st.header("Step 2: Your Payment History")

with st.expander("Payment History"):
    st.write("""
    **Why is Payment History Important?**

    Your payment history is the most significant factor that influences your credit score. It shows how reliably you've paid your bills in the past. Consistently paying on time builds trust with lenders, while late or missed payments can harm your credit score.

    **Understanding Your Payment Status:**
    - **On-Time Payment (-1):** This means you paid your bill by the due date. Paying on time is crucial for maintaining a good credit score.
    - **1 Month Late (1):** This indicates that your payment was one month late. Late payments can negatively impact your score.
    - **2 Months Late (2) and Beyond:** The later your payment, the more significant the impact on your credit score. It signals to lenders that you're struggling to manage your debt.

    **How It Works:**
    You'll need to input your payment status for the last three months. This helps us understand your recent payment behavior, which heavily influences your credit risk.
    """)

    # Payment Status Inputs
    recent_payment_status = st.number_input("Most Recent Payment Status", min_value=-2, max_value=8, 
                            help="Enter the status of your most recent payment. -1 = On time, 1 = 1 month late, 2 = 2 months late, etc.")
    payment_status_2_months = st.number_input("Payment Status 2 Months Ago", min_value=-2, max_value=8, 
                            help="Enter the status of your payment two months ago.")
    payment_status_3_months = st.number_input("Payment Status 3 Months Ago", min_value=-2, max_value=8, 
                            help="Enter the status of your payment three months ago.")

    st.write("""
    **Tips for Maintaining a Positive Payment History:**
    - Set up automatic payments or reminders to ensure you never miss a due date.
    - If you can't pay the full amount, try to at least make the minimum payment to avoid a late mark.
    - Contact your lender if you're having trouble making a payment; sometimes, they can offer a temporary solution.
    """)

    st.write("Your recent payment history is a strong indicator of your future behavior, which is why lenders weigh it so heavily in their decisions.")

# Prediction Button
if st.button("Get My Credit Risk Report"):
    # Convert user input into numerical values using the mappings
    gender_value = gender_options[gender]
    education_value = education_options[education]
    marital_status_value = marriage_options[marital_status]

    # Create a data frame with the input values and include all necessary features with defaults
    user_data = pd.DataFrame({
        'Total Credit Available': [total_credit_available],
        'Gender': [gender_value],
        'Education Level': [education_value],
        'Marital Status': [marital_status_value],
        'Age': [age],
        'Recent Payment Status': [recent_payment_status],
        'Payment Status 2 Months Ago': [payment_status_2_months],
        'Payment Status 3 Months Ago': [payment_status_3_months],
        'Bill Amount 1': [0],  # Default to 0 or allow user input
        'Bill Amount 2': [0],  # Default to 0 or allow user input
        'Bill Amount 3': [0],  # Default to 0 or allow user input
        'Bill Amount 4': [0],  # Default to 0 or allow user input
        'Bill Amount 5': [0],  # Default to 0 or allow user input
        'Bill Amount 6': [0],  # Default to 0 or allow user input
        'Payment Amount 1': [0],  # Default to 0 or allow user input
        'Payment Amount 2': [0],  # Default to 0 or allow user input
        'Payment Amount 3': [0],  # Default to 0 or allow user input
        'Payment Amount 4': [0],  # Default to 0 or allow user input
        'Payment Amount 5': [0],  # Default to 0 or allow user input
        'Payment Amount 6': [0]  # Default to 0 or allow user input
    })

    # Predict probability of default
    prob_default = model.predict_proba(user_data)[0][1]

    st.subheader(f"Your Predicted Risk of Default: {prob_default:.2f}")

    # Explanation
    logistic_model = model.named_steps['logisticregression']
    importance = logistic_model.coef_[0]

    # Sorting the features by importance
    indices = np.argsort(importance)[::-1]
    sorted_feature_names = np.array(feature_names)[indices]
    sorted_importance = importance[indices]

    # Plotting feature importance
    fig, ax = plt.subplots()
    ax.barh(sorted_feature_names, sorted_importance, color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Factors Influencing Your Credit Risk')
    ax.invert_yaxis()
    st.pyplot(fig)

    # Feature Importance Explanation
    st.write("""
    ### What Does This Mean?
    - The chart above shows the different factors that influence your credit score. The taller the bar, the more that factor affects your score.
    - For example, if 'Payment History' has the tallest bar, it means that your recent payments have the biggest impact on your credit risk.
    - By understanding which factors are most important, you can focus on the areas that will help you improve your credit score the most.
    """)

    st.write("""
    ### Try It Yourself
    - Adjust the information you entered above, like your 'Total Credit Available' or 'Payment History', and see how the feature importance changes. This will help you learn which financial behaviors are most important for maintaining a good credit score.
    """)

    st.write("""
    ### Personalized Recommendations:
    - **Pay Your Bills On Time**: Late payments can drastically increase your credit risk.
    - **Reduce Credit Usage**: Try to use less than 30% of your available credit.
    - **Maintain Stable Credit History**: Avoid frequent applications for new credit.
    """)

# Disclaimer at the bottom
st.markdown("""
## Disclaimer
Please note that this tool uses a logistic regression model trained on sample data to provide an illustrative assessment of credit risk. The results are intended to help you understand the factors that may influence credit risk, rather than provide definitive financial advice.
""")
