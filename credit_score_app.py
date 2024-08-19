import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt

# Load and preprocess predefined data
def load_and_preprocess_data():
    # Create a sample dataset
    data = pd.DataFrame({
        'LIMIT_BAL': np.random.randint(10000, 500000, 1000),
        'SEX': np.random.choice([1, 2], 1000),
        'EDUCATION': np.random.choice([1, 2, 3, 4], 1000),
        'MARRIAGE': np.random.choice([1, 2, 3], 1000),
        'AGE': np.random.randint(21, 79, 1000),
        'PAY_0': np.random.choice([-1, 0, 1, 2, 3], 1000),
        'PAY_2': np.random.choice([-1, 0, 1, 2, 3], 1000),
        'PAY_3': np.random.choice([-1, 0, 1, 2, 3], 1000),
        'PAY_4': np.random.choice([-1, 0, 1, 2, 3], 1000),
        'PAY_5': np.random.choice([-1, 0, 1, 2, 3], 1000),
        'PAY_6': np.random.choice([-1, 0, 1, 2, 3], 1000),
        'BILL_AMT1': np.random.randint(0, 100000, 1000),
        'BILL_AMT2': np.random.randint(0, 100000, 1000),
        'BILL_AMT3': np.random.randint(0, 100000, 1000),
        'BILL_AMT4': np.random.randint(0, 100000, 1000),
        'BILL_AMT5': np.random.randint(0, 100000, 1000),
        'BILL_AMT6': np.random.randint(0, 100000, 1000),
        'PAY_AMT1': np.random.randint(0, 50000, 1000),
        'PAY_AMT2': np.random.randint(0, 50000, 1000),
        'PAY_AMT3': np.random.randint(0, 50000, 1000),
        'PAY_AMT4': np.random.randint(0, 50000, 1000),
        'PAY_AMT5': np.random.randint(0, 50000, 1000),
        'PAY_AMT6': np.random.randint(0, 50000, 1000),
        'default payment next month': np.random.choice([0, 1], 1000)
    })
    return data

# Validate and extract target column
def get_target_column(data):
    target_column = 'default payment next month'
    if target_column not in data.columns:
        st.error(f"The target column '{target_column}' was not found in the dataset.")
        st.stop()
    return target_column

# Split and scale data
def split_and_scale_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the model
def train_model(X_train_scaled, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

# Get user input features
def user_input_features(X):
    features = {}
    st.sidebar.header('User Input Parameters')
    for col in X.columns:
        features[col] = st.sidebar.number_input(f'{col}', min_value=float(X[col].min()), max_value=float(X[col].max()), value=float(X[col].mean()))
    return pd.DataFrame(features, index=[0])

# Generate SHAP plots
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

    data = load_and_preprocess_data()
    st.write("### Columns in the dataset:")
    st.write(data.columns)

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
