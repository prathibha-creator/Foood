
import streamlit as st
import pandas as pd
import joblib
import json

@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_rf_model.pkl")
    with open("feature_columns.json", "r") as f:
        feature_info = json.load(f)
    return model, feature_info

model, feature_info = load_artifacts()

st.title("Food Delivery Customer Churn Prediction")

st.write("Provide customer/order details below to predict churn probability.")

input_data = {}

categorical_cols = feature_info["categorical_cols"]
numeric_cols = feature_info["numeric_cols"]
categories = feature_info.get("categories", {})
input_order = feature_info["input_columns"]

# Build form
with st.form("churn_form"):
    for col in input_order:
        if col in categorical_cols:
            options = categories.get(col, [])
            if len(options) == 0:
                val = st.text_input(f"{col} (categorical)")
            else:
                val = st.selectbox(col, options)
            input_data[col] = val
        elif col in numeric_cols:
            val = st.number_input(col, value=0.0)
            input_data[col] = val
        else:
            # Fallback (should not happen)
            val = st.text_input(f"{col}")
            input_data[col] = val

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Create DataFrame in expected format
    input_df = pd.DataFrame([input_data])
    # Predict
    prob = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: **{prob:.2f}**")
    st.write(f"Predicted Class: **{'Churned' if pred == 1 else 'Active'}**")
