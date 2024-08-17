import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

# Streamlit UI
st.title('Credit Score Prediction with Explainable AI')

# Load the dataset from the uploaded file
data = pd.read_csv('/mnt/data/default_of_credit_card_clients.csv')

# Preprocess data
if 'ID' in data.columns:
    data.drop('ID', axis=1, inplace=True)

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')

data.fillna(data.median(), inplace=True)

target_column = 'default payment next month'
if target_column not in data.columns:
    target_column = 'default.payment.next.month'

X = data.drop(target_column, axis=1)
y = data[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# User inputs
st.sidebar.header('User Input Parameters')

def user_input_features():
    features = {}
    for col in X.columns:
        features[col] = st.sidebar.number_input(f'{col}', min_value=float(X[col].min()), max_value=float(X[col].max()), value=float(X[col].mean()))
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Scale the input data
input_scaled = scaler.transform(input_df)

# Predict the credit score using the trained model
prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediction')
st.write('Default' if prediction[0] else 'No Default')

st.subheader('Prediction Probability')
st.write(f'Probability of Default: {prediction_proba[0][1]:.2f}')

# SHAP explanation
st.subheader('Explanation using SHAP')

# Initialize SHAP explainer
explainer = shap.LinearExplainer(model, X_train_scaled)
shap_values_input = explainer.shap_values(input_scaled)

# Plot SHAP force plot
shap_plot = shap.force_plot(explainer.expected_value, shap_values_input[0, :], input_df, matplotlib=True)

# Display the force plot in Streamlit
st.pyplot(plt.gcf())

# SHAP Summary Plot
st.subheader('SHAP Summary Plot')

fig, ax = plt.subplots()
shap.summary_plot(shap_values_input, input_df, show=False)
st.pyplot(fig)

# Additional Feature: SHAP Bar Plot for better interpretability
st.subheader('SHAP Bar Plot of Feature Importance')
shap.bar_plot = shap.plots.bar(shap_values_input)
st.pyplot(plt.gcf())

# Feedback form
st.subheader('User Feedback')
feedback = st.text_area('Provide your feedback on the model and its explanation:')
if st.button('Submit Feedback'):
    st.write('Thank you for your feedback!')
