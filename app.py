import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Diabetes Diagnosis", page_icon="🏥")

@st.cache_resource
def load_assets():
    return joblib.load("diabetes_model_assets.pkl")

assets = load_assets()
model = assets["model"]
scaler = assets["scaler"]
selected_indices = assets["selected_indices"]
all_features = assets["feature_names"]

st.title("🏥 Diabetes Diagnosis System")
st.sidebar.info(f"Optimization: Bat Algorithm selected {len(selected_indices)} features.")

# Navigation
menu = st.sidebar.radio("Menu", ["Predict", "Analysis", "Dataset"])

if menu == "Predict":
    with st.form("input_form"):
        st.subheader("Patient Vitals")
        cols = st.columns(2)
        
        # We need all 8 inputs because the Scaler was trained on all 8
        inputs = {}
        for i, feat in enumerate(all_features):
            with cols[i % 2]:
                inputs[feat] = st.number_input(f"{feat}", value=0.0)
        
        submit = st.form_submit_button("Run Diagnosis")

    if submit:
        # 1. Format input
        raw_data = np.array([list(inputs.values())])
        
        # 2. Scale (using the full 8-feature scaler)
        scaled_data = scaler.transform(raw_data)
        
        # 3. Select only the features the Bat Algorithm chose
        final_input = scaled_data[:, selected_indices]
        
        # 4. Predict
        prediction = model.predict(final_input)[0]
        prob = model.predict_proba(final_input)[0]

        if prediction == 1:
            st.error(f"Result: Diabetic (Confidence: {prob[1]:.2%})")
        else:
            st.success(f"Result: Non-Diabetic (Confidence: {prob[0]:.2%})")

elif menu == "Analysis":
    st.subheader("Optimization Results")
    st.write("The Bat Algorithm identified these features as most predictive:")
    st.write(assets["selected_features"])

elif menu == "Dataset":
    st.subheader("Sample Data")
    df = pd.read_csv("diabetes.csv")
    st.dataframe(df.head(10))