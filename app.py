import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Lead Scorer", page_icon="🤖")

st.title("My First AI App")
st.write("Enter values and click Predict.")

model = joblib.load("lead_model.joblib")

company_size = st.number_input("Company size", min_value=1, max_value=5000, value=50)
has_dpo = st.selectbox("Has DPO?", [0, 1])
lead_source = st.selectbox("Lead source", ["inbound", "outbound", "partner"])

lead_source_inbound = 1 if lead_source == "inbound" else 0
lead_source_outbound = 1 if lead_source == "outbound" else 0
lead_source_partner = 1 if lead_source == "partner" else 0

input_df = pd.DataFrame([{
    "company_size": company_size,
    "has_dpo": has_dpo,
    "lead_source_inbound": lead_source_inbound,
    "lead_source_outbound": lead_source_outbound,
    "lead_source_partner": lead_source_partner
}])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
st.subheader("Feature Importances")
