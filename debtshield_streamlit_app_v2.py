import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="DebtShield AI", page_icon="🛡️", layout="wide")

st.markdown('''
<style>
.block-container {padding-top: 1.5rem;}
.main-title {font-size: 3rem; font-weight: 800; color: #0f172a; margin-bottom: 0rem;}
.subtitle {font-size: 1.1rem; color: #475569; margin-bottom: 1rem;}
.note {font-size: 0.9rem; color: #64748b;}
</style>
''', unsafe_allow_html=True)

# ------------------ helpers ------------------
def clean_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return x
    s = str(x).strip().lower()
    if s in ["", "nan", "none", "null", "n/a", "na"]: return np.nan
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("/year", "").replace("/hr", "").replace(" miles", "")
    try: return float(s)
    except: return x

def score_between(s, low, high):
    s = pd.to_numeric(s, errors="coerce").astype(float)
    return ((s - low) / (high - low) * 100).clip(0, 100)

def inv_score(s, low, high):
    return 100 - score_between(s, low, high)

def col(df, name, default=0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)

def risk_label(x):
    if x < 35: return "Low"
    if x < 65: return "Medium"
    return "High"

def clean_df(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    text_cols = {"state", "county", "risk_level", "notes", "display_name"}
    for c in df.columns:
        if c.lower() not in text_cols:
            df[c] = df[c].apply(clean_num)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.dropna(how="all", subset=numeric_cols)
    for c in numeric_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].fillna("Unknown")
    return df.reset_index(drop=True)

def engineer(df):
    out = df.copy()
    income = col(out, "acs_median_household_income", 81604).replace(0, np.nan)
    annual_rent = col(out, "acs_median_gross_rent", 1487) * 12
    out["derived_rent_to_income_ratio"] = (annual_rent / income * 100).fillna(0)
    living_wage = col(out, "mit_living_wage_annual", income)
    out["derived_living_wage_gap"] = ((living_wage - income) / living_wage.replace(0, np.nan) * 100).fillna(0)

    out["derived_housing_stress_score"] = (
        .35 * score_between(col(out, "acs_rent_burden_pct"), 20, 60) +
        .25 * score_between(out["derived_rent_to_income_ratio"], 15, 45) +
        .20 * score_between(col(out, "eviction_filing_rate"), 0, 8) +
        .20 * score_between(col(out, "cdc_crowded_housing_pct"), 0, 8)
    )
    debt_to_income = (col(out, "scf_avg_household_debt") / income).fillna(0)
    out["derived_debt_stress_score"] = (
        .35 * score_between(col(out, "scf_debt_to_income_ratio"), .4, 2.5) +
        .25 * score_between(debt_to_income, .5, 3.0) +
        .20 * score_between(col(out, "nyfed_credit_card_delinquency_rate"), 0, 8) +
        .20 * score_between(col(out, "nyfed_mortgage_delinquency_rate"), 0, 6)
    )
    out["derived_cost_pressure_score"] = (
        .30 * score_between(col(out, "acs_poverty_rate"), 5, 30) +
        .30 * score_between(col(out, "bls_unemployment_rate"), 2, 12) +
        .25 * score_between(out["derived_living_wage_gap"], -100, 30) +
        .15 * inv_score(col(out, "bls_job_growth_pct"), -5, 5)
    )
    out["derived_energy_stress_score"] = (
        .50 * score_between(col(out, "doe_total_energy_burden_pct"), 2, 12) +
        .25 * score_between(col(out, "doe_electricity_burden_pct"), 1, 7) +
        .25 * score_between(col(out, "doe_heating_burden_pct"), .5, 5)
    )
    out["derived_food_access_risk_score"] = (
        .35 * score_between(col(out, "usda_food_desert_indicator"), 0, 1) +
        .45 * score_between(col(out, "usda_low_income_low_access_pct"), 0, 30) +
        .20 * score_between(col(out, "usda_grocery_distance_score"), 0, 5)
    )
    out["derived_financial_distress_index"] = (
        .30 * out["derived_housing_stress_score"] +
        .25 * out["derived_debt_stress_score"] +
        .20 * out["derived_cost_pressure_score"] +
        .15 * out["derived_energy_stress_score"] +
        .10 * out["derived_food_access_risk_score"]
    ).clip(0, 100)
    out["risk_level"] = out["derived_financial_distress_index"].apply(risk_label)
    return out

def practical_name(row, i):
    drivers = {
        "Rent-Burdened Metro Area": row.get("derived_housing_stress_score", 0),
        "Debt-Stressed Community": row.get("derived_debt_stress_score", 0),
        "Energy-Burdened Region": row.get("derived_energy_stress_score", 0),
        "Food-Access Vulnerability Zone": row.get("derived_food_access_risk_score", 0),
    }
    top = max(drivers, key=drivers.get)
    prefix = {"High":"High-Risk", "Medium":"Moderate-Risk", "Low":"Lower-Risk"}.get(row.get("risk_level"), "Risk")
    return f"{prefix} {top} #{i+1:03d}"

def add_names(df):
    names = []
    for i, r in df.iterrows():
        county = str(r.get("county", "")).strip()
        if county and county.lower() not in ["nan", "unknown", "national average"] and not county.lower().startswith("synthetic"):
            names.append(county)
        else:
            names.append(practical_name(r, i))
    df["display_name"] = names
    return df

def recs(row):
    out = []
    if row.get("derived_housing_stress_score",0) >= 60: out.append(("Housing", "Expand rental assistance, affordable housing supply, and eviction diversion."))
    if row.get("derived_debt_stress_score",0) >= 60: out.append(("Debt", "Provide debt counseling, credit repair, emergency cash support, and refinancing pathways."))
    if row.get("derived_cost_pressure_score",0) >= 60: out.append(("Cost of living", "Expand benefits enrollment, wage support, and job placement programs."))
    if row.get("derived_energy_stress_score",0) >= 60: out.append(("Energy", "Increase utility assistance, weatherization, and home efficiency upgrades."))
    if row.get("derived_food_access_risk_score",0) >= 60: out.append(("Food access", "Support grocery access, SNAP outreach, mobile markets, and transportation access."))
    if not out: out.append(("Monitoring", "Maintain prevention programs and monitor early-warning indicators."))
    return out

def bar(labels, vals, title):
    fig, ax = plt.subplots(figsize=(9,4.6))
    ax.bar(labels, vals)
    ax.set_ylim(0, max(100, max(vals)+10))
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=.25)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig

def risk_icon(r):
    return {"Low":"🟢", "Medium":"🟡", "High":"🔴"}.get(r,"⚪")

# ------------------ sidebar/data ------------------
st.sidebar.title("🛡️ DebtShield AI")
st.sidebar.caption("Financial distress risk engine")
page = st.sidebar.radio("Navigation", ["Overview", "Community Risk", "Risk Drivers", "Model Results", "Recommendations", "About"])
uploaded = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded:
    raw = pd.read_csv(uploaded)
else:
    st.sidebar.info("Using demo benchmark until a CSV is uploaded.")
    raw = pd.DataFrame([{"state":"United States","county":"National Average","year":2025,"acs_median_household_income":81604,"acs_median_gross_rent":1487,"acs_rent_burden_pct":48.23,"acs_poverty_rate":12.1,"eviction_filing_rate":2.79,"scf_avg_household_debt":155600,"scf_debt_to_income_ratio":1.41,"bls_unemployment_rate":4.2,"bls_job_growth_pct":1.3,"mit_living_wage_annual":23.32,"cdc_crowded_housing_pct":1.3,"usda_food_desert_indicator":17.4,"usda_low_income_low_access_pct":17.4,"usda_grocery_distance_score":1.8,"doe_total_energy_burden_pct":6.5,"doe_electricity_burden_pct":3.8,"doe_heating_burden_pct":1.7,"nyfed_credit_card_delinquency_rate":3.2,"nyfed_mortgage_delinquency_rate":1.3}])

df = add_names(engineer(clean_df(raw)))
st.sidebar.metric("Rows loaded", f"{len(df):,}")
st.sidebar.metric("Columns loaded", f"{len(df.columns):,}")
st.sidebar.caption("Synthetic communities are used for ML demonstration and scenario testing.")

selected = st.sidebar.selectbox("Select community", df["display_name"].tolist())
row = df[df["display_name"] == selected].iloc[0]

st.markdown('<div class="main-title">DebtShield AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explainable financial distress, poverty risk, housing instability, and economic vulnerability platform.</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------ pages ------------------
if page == "Overview":
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Communities", f"{len(df):,}")
    c2.metric("Data sources", "11")
    c3.metric("Features", "50+")
    c4.metric("Selected risk", f"{risk_icon(row['risk_level'])} {row['risk_level']}")
    st.subheader("Purpose")
    st.write("DebtShield AI combines housing, debt, labor, food access, energy burden, and social vulnerability indicators into a single financial risk platform. It helps identify communities that may need support before distress becomes a crisis.")
    counts = df["risk_level"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
    st.pyplot(bar(counts.index.tolist(), counts.values.tolist(), "Risk level distribution"))

elif page == "Community Risk":
    st.subheader(selected)
    c1,c2,c3 = st.columns(3)
    c1.metric("Financial Distress Index", f"{row['derived_financial_distress_index']:.2f}")
    c2.metric("Risk Level", f"{risk_icon(row['risk_level'])} {row['risk_level']}")
    c3.metric("Rent-to-Income", f"{row['derived_rent_to_income_ratio']:.2f}%")
    labels = ["Housing", "Debt", "Cost Pressure", "Energy", "Food Access"]
    vals = [row["derived_housing_stress_score"], row["derived_debt_stress_score"], row["derived_cost_pressure_score"], row["derived_energy_stress_score"], row["derived_food_access_risk_score"]]
    st.pyplot(bar(labels, vals, "Community stress profile"))
    with st.expander("View selected row"):
        st.dataframe(pd.DataFrame(row).rename(columns={row.name:"value"}), use_container_width=True)

elif page == "Risk Drivers":
    st.subheader("Risk driver ranking")
    driver_df = pd.DataFrame({"Driver":["Housing Stress","Debt Stress","Cost Pressure","Energy Stress","Food Access Risk"],"Score":[row["derived_housing_stress_score"],row["derived_debt_stress_score"],row["derived_cost_pressure_score"],row["derived_energy_stress_score"],row["derived_food_access_risk_score"]]}).sort_values("Score", ascending=False)
    st.dataframe(driver_df, use_container_width=True)
    st.pyplot(bar(driver_df["Driver"].tolist(), driver_df["Score"].tolist(), "Top risk drivers"))
    st.success(f"Strongest driver: {driver_df.iloc[0]['Driver']} ({driver_df.iloc[0]['Score']:.2f})")

elif page == "Model Results":
    st.subheader("Platform results")
    st.info("Synthetic communities are used for ML demonstration and scenario testing. The next upgrade is real county-level rows.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Index range", f"{df['derived_financial_distress_index'].min():.1f}–{df['derived_financial_distress_index'].max():.1f}")
    c2.metric("Average index", f"{df['derived_financial_distress_index'].mean():.1f}")
    c3.metric("High-risk rows", f"{(df['risk_level']=='High').sum():,}")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(df["derived_financial_distress_index"], bins=25)
    ax.set_title("Financial Distress Index distribution", fontweight="bold")
    ax.set_xlabel("Financial Distress Index")
    ax.set_ylabel("Communities")
    ax.grid(axis="y", alpha=.25)
    st.pyplot(fig)

elif page == "Recommendations":
    st.subheader("Policy recommendation engine")
    st.write(f"Selected community: **{selected}**")
    st.metric("Risk Level", f"{risk_icon(row['risk_level'])} {row['risk_level']}")
    for cat, text in recs(row):
        st.markdown(f"### {cat}")
        st.write(text)
    st.download_button("Download scored dataset", data=df.to_csv(index=False), file_name="debtshield_ai_scored_dataset.csv", mime="text/csv")

else:
    st.subheader("About DebtShield AI")
    st.write("DebtShield AI integrates 11 public datasets to evaluate financial vulnerability using feature engineering, risk scoring, and explainable recommendations.")
    st.markdown("### Data sources")
    st.write("ACS, Zillow, Eviction Lab, Federal Reserve SCF, BLS, MIT Living Wage, CDC SVI, FHFA, USDA Food Access, DOE LEAD, and NY Fed Consumer Credit.")
    st.warning("Prototype note: synthetic communities are used for ML demonstration and scenario testing. Future versions should use real county-level observations.")
