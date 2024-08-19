import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    if 'ID' in data.columns:
        data.drop('ID', axis=1, inplace=True)

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data.fillna(data.median(), inplace=True)
    return data

# Function to validate and extract target column
def get_target_column(data):
    target_column = 'default payment next month'
    if target_column not in data.columns:
        alternative_column = 'default.payment.next.month'
        if alternative_column in data.columns:
            target_column = alternative_column
        else:
            st.error(f"The target column '{target_column}' was not found in the dataset.")
            st.stop()
    return target_column

# Function to split and scale data
def split_and_scale_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to train the model
def train_model(X_train_scaled, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

# Function to get user input
def user_input_features(X):
    features = {}
    st.sidebar.header('User Input Parameters')
    for col in X.columns:
        features[col] = st.sidebar.number_input(f'{col}', min_value=float(X[col].min()), max_value=float(X[col].max()), value=float(X[col].mean()))
    return pd.DataFrame(features, index=[0])

# Function to generate SHAP plots
def generate_shap_plots(model, X_train_scaled, input_scaled, input_df):
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values_input = explainer.shap_values(input_scaled)

    st.subheader('SHAP Force Plot')
    shap_plot = shap.force_plot(explainer.expected_value, shap_values_input[0, :], input_df, matplotlib=True)
    st.pyplot(plt.gcf())

    st.subheader('SHAP Summary Plot')
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values_input, input_df, show=False)
    st.pyplot(fig)

    st.subheader('SHAP Bar Plot of Feature Importance')
    shap.bar_plot = shap.plots.bar(shap_values_input)
    st.pyplot(plt.gcf())

# Main Streamlit UI
def main():
    st.title('Credit Score Prediction with Explainable AI')
    st.markdown("### Understand your financial health with AI-driven insights.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = load_and_preprocess_data(uploaded_file)
        st.write("### Columns in the dataset:")
        st.write(data.columns)
    else:
        st.error("Please upload the dataset to proceed.")
        st.stop()

    target_column = get_target_column(data)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(data, target_column)
    model = train_model(X_train_scaled, y_train)

    input_df = user_input_features(data.drop(target_column, axis=1))
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    st.subheader('Prediction')
    st.write('Your predicted credit status is:', 'Default' if prediction[0] else 'No Default')

    st.subheader('Prediction Probability')
    st.write(f'Probability of Default: {prediction_proba[0][1]:.2f}')

    generate_shap_plots(model, X_train_scaled, input_scaled, input_df)

    st.subheader('User Feedback')
    feedback = st.text_area('Provide your feedback on the model and its explanation:')
    if st.button('Submit Feedback'):
        st.write('Thank you for your feedback!')

if __name__ == "__main__":
    main()
