
# ============================================================
# DebtShield AI — Streamlit App
# ============================================================
# Run locally:
# pip install streamlit pandas numpy scikit-learn xgboost shap matplotlib joblib
# streamlit run debtshield_streamlit_app.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import joblib
except Exception:
    st.error("Install dependencies: pip install scikit-learn joblib")
    st.stop()

st.set_page_config(
    page_title="DebtShield AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DebtShield AI")
st.subheader("An Explainable Machine Learning Platform for Predicting Financial Distress, Poverty Risk, Housing Instability, and Economic Vulnerability")

st.markdown("""
DebtShield AI uses economic, housing, debt, energy, food-access, and vulnerability indicators to estimate community financial distress risk.
Upload the cleaned dataset or use the demo inputs below.
""")

def clean_numeric_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return x
    s = str(x).strip()
    if s.lower() in ["", "nan", "none", "null"]:
        return np.nan
    s = (
        s.replace("$", "")
         .replace(",", "")
         .replace("%", "")
         .replace("/year", "")
         .replace("/hr", "")
         .replace(" miles", "")
         .replace("mile", "")
    )
    try:
        return float(s)
    except:
        return x

def engineer_features(input_df):
    data = input_df.copy()

    data["ds_rent_to_income_pct"] = (
        (data["acs_median_gross_rent"] * 12) / data["acs_median_household_income"] * 100
    )

    data["ds_housing_stress_score"] = (
        0.40 * data["acs_rent_burden_pct"] +
        0.25 * data["ds_rent_to_income_pct"] +
        0.20 * data["eviction_filing_rate"] * 10 +
        0.15 * data["zillow_rent_growth_yoy_pct"] * 5
    )

    data["ds_debt_to_income_pct"] = data["scf_debt_to_income_ratio"] * 100
    data["ds_household_debt_to_income_pct"] = (
        data["scf_avg_household_debt"] / data["acs_median_household_income"] * 100
    )
    data["ds_debt_stress_score"] = (
        0.45 * data["ds_debt_to_income_pct"] +
        0.35 * data["ds_household_debt_to_income_pct"] +
        0.20 * data["nyfed_credit_card_delinquency_rate"] * 10
    )

    data["ds_cost_pressure_score"] = (
        (data["mit_housing_cost_annual"] + data["mit_food_cost_annual"] + data["mit_transportation_cost_annual"])
        / data["acs_median_household_income"] * 100
    )

    data["ds_energy_stress_score"] = (
        0.60 * data["doe_total_energy_burden_pct"] +
        0.25 * data["doe_electricity_burden_pct"] +
        0.15 * data["doe_heating_burden_pct"]
    )

    data["ds_food_access_risk_score"] = (
        0.60 * data["usda_low_income_low_access_pct"] +
        0.25 * data["usda_food_desert_indicator"] +
        0.15 * data["usda_grocery_distance_score"] * 5
    )

    data["ds_employment_stress_score"] = (
        0.75 * data["bls_unemployment_rate"] * 10 -
        0.25 * data["bls_job_growth_pct"] * 5
    )

    data["ds_social_vulnerability_score"] = data["cdc_svi_overall_score"] * 100

    data["financial_distress_index"] = (
        0.25 * data["ds_housing_stress_score"] +
        0.30 * data["ds_debt_stress_score"] +
        0.15 * data["ds_cost_pressure_score"] +
        0.10 * data["ds_energy_stress_score"] +
        0.10 * data["ds_food_access_risk_score"] +
        0.05 * data["ds_employment_stress_score"] +
        0.05 * data["ds_social_vulnerability_score"]
    )

    data["financial_distress_index"] = data["financial_distress_index"].clip(0, 100)
    data["risk_level"] = data["financial_distress_index"].apply(risk_label)
    return data

def risk_label(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    return "High"

def generate_recommendations(row):
    recs = []
    if row.get("ds_housing_stress_score", 0) > 40:
        recs.append("🏠 Housing: expand rental assistance, affordable housing supply, and eviction prevention.")
    if row.get("ds_debt_stress_score", 0) > 70:
        recs.append("💳 Debt: provide debt counseling, credit restructuring support, and emergency cash assistance.")
    if row.get("ds_energy_stress_score", 0) > 7:
        recs.append("⚡ Energy: increase utility assistance and weatherization programs.")
    if row.get("ds_food_access_risk_score", 0) > 20:
        recs.append("🥫 Food access: support grocery access, SNAP outreach, and food distribution programs.")
    if row.get("bls_unemployment_rate", 0) > 6:
        recs.append("💼 Employment: expand workforce training and job placement support.")
    if not recs:
        recs.append("✅ Maintain monitoring; no major immediate intervention flagged.")
    return recs

uploaded_file = st.sidebar.file_uploader("Upload cleaned financial distress CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using demo national benchmark values.")
    df = pd.DataFrame([{
        "state": "United States",
        "county": "National Average",
        "year": 2025,
        "acs_median_household_income": 81604,
        "acs_median_gross_rent": 1487,
        "acs_rent_burden_pct": 48.23,
        "acs_poverty_rate": 12.1,
        "zillow_rent_growth_yoy_pct": 1.92,
        "eviction_filing_rate": 2.79,
        "scf_avg_household_debt": 155600,
        "scf_credit_card_debt": 6270,
        "scf_debt_to_income_ratio": 1.41,
        "bls_unemployment_rate": 4.2,
        "bls_median_wage": 49500,
        "bls_job_growth_pct": 1.3,
        "mit_housing_cost_annual": 13305,
        "mit_food_cost_annual": 4504,
        "mit_transportation_cost_annual": 8385,
        "cdc_svi_overall_score": 0.5,
        "cdc_no_vehicle_pct": 7.81,
        "usda_food_desert_indicator": 17.4,
        "usda_low_income_low_access_pct": 17.4,
        "usda_grocery_distance_score": 1.8,
        "doe_total_energy_burden_pct": 6.5,
        "doe_electricity_burden_pct": 3.8,
        "doe_heating_burden_pct": 1.7,
        "nyfed_credit_card_delinquency_rate": 3.2,
        "nyfed_mortgage_delinquency_rate": 1.3
    }])

for col in df.columns:
    if col not in ["state", "county", "risk_level", "notes"]:
        df[col] = df[col].apply(clean_numeric_value)

df = df.dropna(how="all")
df = engineer_features(df)

st.sidebar.markdown("### Dataset")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")

row_index = st.sidebar.selectbox("Select row/community", options=list(range(len(df))), format_func=lambda i: f"{i}: {df.iloc[i].get('county', 'Unknown')}")

row = df.iloc[row_index]

score = row["financial_distress_index"]
risk = row["risk_level"]

col1, col2, col3 = st.columns(3)
col1.metric("Financial Distress Index", f"{score:.2f}")
col2.metric("Risk Level", risk)
col3.metric("Community", str(row.get("county", "Unknown")))

st.markdown("---")

risk_drivers = pd.DataFrame({
    "Driver": [
        "Housing Stress",
        "Debt Stress",
        "Cost Pressure",
        "Energy Stress",
        "Food Access Risk",
        "Employment Stress",
        "Social Vulnerability"
    ],
    "Score": [
        row.get("ds_housing_stress_score", 0),
        row.get("ds_debt_stress_score", 0),
        row.get("ds_cost_pressure_score", 0),
        row.get("ds_energy_stress_score", 0),
        row.get("ds_food_access_risk_score", 0),
        row.get("ds_employment_stress_score", 0),
        row.get("ds_social_vulnerability_score", 0)
    ]
}).sort_values("Score", ascending=False)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Top Risk Drivers")
    st.bar_chart(risk_drivers.set_index("Driver"))

with right:
    st.subheader("Policy Recommendations")
    for rec in generate_recommendations(row):
        st.write(rec)

st.markdown("---")
st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Risk Distribution")
if "risk_level" in df.columns:
    st.bar_chart(df["risk_level"].value_counts())

st.subheader("Download Processed Dataset")
st.download_button(
    "Download DebtShield Processed CSV",
    data=df.to_csv(index=False),
    file_name="debtshield_processed_dataset.csv",
    mime="text/csv"
)
