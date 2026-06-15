
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib

st.set_page_config(
    page_title="DebtShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DebtShield AI v4 — polished platform version
# ============================================================

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1500px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f172a 0%, #111827 100%);}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 24px;
    background: radial-gradient(circle at top left, rgba(56,189,248,.35), transparent 35%),
                linear-gradient(135deg, #0f172a 0%, #172554 48%, #020617 100%);
    border: 1px solid rgba(148,163,184,.25);
    box-shadow: 0 12px 35px rgba(2,6,23,.35);
    margin-bottom: 1rem;
}
.hero-title {font-size: 3.1rem; font-weight: 900; color: #f8fafc; margin-bottom: .25rem;}
.hero-sub {font-size: 1.08rem; color: #cbd5e1; max-width: 1150px;}
.badge {
    display:inline-block; padding:.33rem .65rem; border-radius:999px;
    background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.35);
    color:#bae6fd; font-size:.78rem; font-weight:800; margin-right:.35rem; margin-bottom:.45rem;
}
.card {
    background: linear-gradient(135deg, #111827 0%, #182235 100%);
    border: 1px solid rgba(148,163,184,.22);
    border-radius: 21px;
    padding: 1.05rem 1.1rem;
    box-shadow: 0 8px 26px rgba(2,6,23,.25);
    min-height: 110px;
}
.card-light {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 21px;
    padding: 1.05rem 1.1rem;
    box-shadow: 0 6px 18px rgba(15,23,42,.08);
}
.kpi-label {font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:#94a3b8; font-weight:800;}
.kpi-value {font-size:2rem; font-weight:900; color:#f8fafc; line-height:1.1;}
.kpi-sub {font-size:.86rem; color:#cbd5e1;}
.panel-title {font-weight:900; font-size:1.25rem; margin-bottom:.5rem;}
.note {
    padding:.85rem 1rem; border-radius:16px; background:#eff6ff; border:1px solid #bfdbfe;
    color:#1e3a8a; margin:.6rem 0;
}
.warn {
    padding:.85rem 1rem; border-radius:16px; background:#fffbeb; border:1px solid #fde68a;
    color:#92400e; margin:.6rem 0;
}
.small {font-size:.86rem; color:#64748b;}
.risk-pill {
    display:inline-block; padding:.35rem .75rem; border-radius:999px; font-weight:900;
}
.low {background:#dcfce7; color:#166534;}
.medium {background:#fef3c7; color:#92400e;}
.high {background:#fee2e2; color:#991b1b;}
.footer {font-size:.8rem; color:#94a3b8; margin-top:1.2rem;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Data helpers
# -----------------------------
def clean_numeric_value(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return x
    s = str(x).strip()
    if s.lower() in ["", "nan", "none", "null", "na", "n/a"]:
        return np.nan
    s = (
        s.replace("$", "").replace(",", "").replace("%", "")
        .replace("/year", "").replace("/hr", "").replace(" miles", "")
        .replace("mile", "").replace("−", "-")
    )
    try:
        return float(s)
    except Exception:
        return x

def get_col(df, name, default=0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)

def score_between(series, low, high):
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if high == low:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return ((s - low) / (high - low) * 100).clip(0, 100)

def inverse_score_between(series, low, high):
    return (100 - score_between(series, low, high)).clip(0, 100)

def risk_label(score):
    if score < 35:
        return "Low"
    if score < 65:
        return "Medium"
    return "High"

def risk_icon(risk):
    return {"Low":"🟢", "Medium":"🟡", "High":"🔴"}.get(risk, "⚪")

def risk_class(risk):
    return {"Low":"low", "Medium":"medium", "High":"high"}.get(risk, "medium")

def clean_dataframe(df):
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    text_cols = {"state", "county", "risk_level", "notes", "display_name", "profile_name", "community_profile"}
    for c in out.columns:
        if c.lower() not in text_cols:
            out[c] = out[c].apply(clean_numeric_value)
    feature_cols = [c for c in out.columns if c.lower() not in text_cols]
    out = out.dropna(how="all", subset=feature_cols)
    nums = out.select_dtypes(include=[np.number]).columns
    for c in nums:
        out[c] = out[c].fillna(out[c].median())
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].fillna("Unknown")
    return out.reset_index(drop=True)

def engineer_features(df):
    out = df.copy()
    income = get_col(out, "acs_median_household_income", 81604).replace(0, np.nan)
    annual_rent = get_col(out, "acs_median_gross_rent", 1487) * 12
    out["derived_rent_to_income_ratio"] = (annual_rent / income * 100).fillna(0)

    living_wage = get_col(out, "mit_living_wage_annual", income)
    living_wage = pd.Series(np.where(living_wage < 200, living_wage * 2080, living_wage), index=out.index)
    out["derived_living_wage_gap"] = ((living_wage - income) / living_wage.replace(0, np.nan) * 100).fillna(0)

    out["derived_housing_stress_score"] = (
        0.35 * score_between(get_col(out, "acs_rent_burden_pct"), 20, 60) +
        0.25 * score_between(out["derived_rent_to_income_ratio"], 15, 45) +
        0.20 * score_between(get_col(out, "eviction_filing_rate"), 0, 8) +
        0.20 * score_between(get_col(out, "cdc_crowded_housing_pct"), 0, 8)
    ).clip(0,100)

    debt_amount_ratio = (get_col(out, "scf_avg_household_debt") / income).fillna(0)
    out["derived_debt_stress_score"] = (
        0.35 * score_between(get_col(out, "scf_debt_to_income_ratio"), 0.4, 2.5) +
        0.25 * score_between(debt_amount_ratio, 0.5, 3.0) +
        0.20 * score_between(get_col(out, "nyfed_credit_card_delinquency_rate"), 0, 8) +
        0.20 * score_between(get_col(out, "nyfed_mortgage_delinquency_rate"), 0, 6)
    ).clip(0,100)

    out["derived_cost_pressure_score"] = (
        0.30 * score_between(get_col(out, "acs_poverty_rate"), 5, 30) +
        0.30 * score_between(get_col(out, "bls_unemployment_rate"), 2, 12) +
        0.25 * score_between(out["derived_living_wage_gap"], -100, 30) +
        0.15 * inverse_score_between(get_col(out, "bls_job_growth_pct"), -5, 5)
    ).clip(0,100)

    out["derived_energy_stress_score"] = (
        0.50 * score_between(get_col(out, "doe_total_energy_burden_pct"), 2, 12) +
        0.25 * score_between(get_col(out, "doe_electricity_burden_pct"), 1, 7) +
        0.25 * score_between(get_col(out, "doe_heating_burden_pct"), 0.5, 5)
    ).clip(0,100)

    out["derived_food_access_risk_score"] = (
        0.35 * score_between(get_col(out, "usda_food_desert_indicator"), 0, 1) +
        0.45 * score_between(get_col(out, "usda_low_income_low_access_pct"), 0, 30) +
        0.20 * score_between(get_col(out, "usda_grocery_distance_score"), 0, 5)
    ).clip(0,100)

    out["derived_financial_distress_index"] = (
        0.30 * out["derived_housing_stress_score"] +
        0.25 * out["derived_debt_stress_score"] +
        0.20 * out["derived_cost_pressure_score"] +
        0.15 * out["derived_energy_stress_score"] +
        0.10 * out["derived_food_access_risk_score"]
    ).clip(0,100)
    out["risk_level"] = out["derived_financial_distress_index"].apply(risk_label)
    return out

# -----------------------------
# Better non-number community naming
# -----------------------------
PLACE_NAMES = {
    "Housing": [
        "Riverside Rent-Burden Corridor", "East Market Housing Pressure Zone", "Lakeside Tenant Stress District",
        "Metro Edge Affordability Cluster", "Oakview Rental Strain Area", "Central Station Housing Burden Hub",
        "Harborview Tenant Risk Corridor", "Southgate Affordability Zone"
    ],
    "Debt": [
        "Brookfield Debt Pressure District", "Westhaven Credit Strain Area", "Maple Ridge Household Debt Cluster",
        "Northpoint Delinquency Watch Zone", "Cedar Valley Consumer Debt Corridor", "Fairview Credit Risk District",
        "Pinecrest Household Balance Stress Area", "Hillcrest Debt Vulnerability Hub"
    ],
    "Cost": [
        "Southridge Cost-of-Living Pressure Zone", "Meadow Park Basic Needs Strain Area", "Union Heights Wage Gap Corridor",
        "Liberty Crossing Cost Pressure District", "Stonebridge Income Stress Cluster", "Clearwater Living Cost Watch Area",
        "Highland Wage-Support Priority Zone", "Willow Creek Household Expense Corridor"
    ],
    "Energy": [
        "Coldstream Utility Burden District", "North Mill Energy Cost Corridor", "Pine Hollow Heating Burden Area",
        "Redwood Utility Stress Zone", "West Ridge Energy Affordability Cluster", "Lakewood Weatherization Priority Area",
        "Summit View Power Cost District", "Elmwood Energy Burden Watch Zone"
    ],
    "Food": [
        "Greenfield Grocery Access Gap", "Eastwood Food Access Priority Area", "Cedar Grove Low-Access Food Corridor",
        "Riverbend Nutrition Access Zone", "Northside Grocery Distance Cluster", "Valley View Food Desert Watch Area",
        "South Park Market Access Gap", "Brookline Food Access Vulnerability Area"
    ]
}

def top_driver(row):
    drivers = {
        "Housing": row.get("derived_housing_stress_score", 0),
        "Debt": row.get("derived_debt_stress_score", 0),
        "Cost": row.get("derived_cost_pressure_score", 0),
        "Energy": row.get("derived_energy_stress_score", 0),
        "Food": row.get("derived_food_access_risk_score", 0),
    }
    key = max(drivers, key=drivers.get)
    return key, drivers[key]

def stable_pick(options, row):
    raw = f"{row.get('derived_financial_distress_index',0):.4f}-{row.get('derived_debt_stress_score',0):.4f}-{row.get('derived_housing_stress_score',0):.4f}"
    h = int(hashlib.sha256(raw.encode()).hexdigest(), 16)
    return options[h % len(options)]

def make_profile_name(row):
    county = str(row.get("county", "")).strip()
    if county and county.lower() not in ["nan", "unknown", "national average"] and not county.lower().startswith("synthetic"):
        return county

    driver, _ = top_driver(row)
    base = stable_pick(PLACE_NAMES[driver], row)
    risk = row.get("risk_level", "Medium")

    # No numbers; practical profile tags instead
    if risk == "High":
        prefix = "High-Risk"
    elif risk == "Medium":
        prefix = "Moderate-Risk"
    else:
        prefix = "Lower-Risk"

    return f"{prefix} {base}"

def add_names(df):
    out = df.copy()
    out["display_name"] = [make_profile_name(row) for _, row in out.iterrows()]
    # If duplicates happen, add a descriptive suffix, not a raw number
    counts = {}
    final = []
    suffixes = ["Alpha", "Beta", "Gamma", "Delta", "North", "South", "East", "West", "Central", "Metro"]
    for name in out["display_name"]:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            final.append(name)
        else:
            final.append(f"{name} — {suffixes[(counts[name]-2) % len(suffixes)]}")
    out["display_name"] = final
    return out

def recs(row):
    out = []
    if row.get("derived_housing_stress_score", 0) >= 50:
        out.append(("🏠 Housing", "Expand rental assistance, eviction diversion, and affordable housing supply."))
    if row.get("derived_debt_stress_score", 0) >= 50:
        out.append(("💳 Debt", "Offer debt counseling, delinquency outreach, credit restructuring, and emergency cash support."))
    if row.get("derived_cost_pressure_score", 0) >= 50:
        out.append(("💼 Cost of Living", "Increase benefits enrollment, job placement, wage support, and basic-needs assistance."))
    if row.get("derived_energy_stress_score", 0) >= 50:
        out.append(("⚡ Energy", "Increase utility assistance, weatherization, and energy-efficiency upgrades."))
    if row.get("derived_food_access_risk_score", 0) >= 50:
        out.append(("🥫 Food Access", "Support grocery access, mobile markets, SNAP outreach, and transportation to food retailers."))
    if not out:
        out.append(("✅ Monitoring", "Maintain prevention programs and continue monitoring early-warning indicators."))
    return out

# -----------------------------
# Chart helpers
# -----------------------------
def plot_bar(labels, values, title, ylabel="Score", cap=110):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(labels, values)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, cap)
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=.22)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+1, f"{h:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    return fig

def plot_risk_counts(df):
    counts = df["risk_level"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    bars = ax.bar(counts.index, counts.values)
    ax.set_title("Dataset risk distribution", fontweight="bold", fontsize=14)
    ax.set_ylabel("Community profiles")
    ax.grid(axis="y", alpha=.22)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.5, f"{int(h)}", ha="center")
    plt.tight_layout()
    return fig

def plot_index_hist(df):
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.hist(df["derived_financial_distress_index"], bins=28)
    ax.set_title("Financial Distress Index distribution", fontweight="bold", fontsize=14)
    ax.set_xlabel("Index score")
    ax.set_ylabel("Community profiles")
    ax.grid(axis="y", alpha=.22)
    plt.tight_layout()
    return fig

# -----------------------------
# Load data
# -----------------------------
st.sidebar.title("🛡️ DebtShield AI")
st.sidebar.caption("Financial vulnerability intelligence platform")

uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = "Uploaded dataset"
else:
    raw = pd.DataFrame([{
        "state":"United States", "county":"National Average", "year":2025,
        "acs_median_household_income":81604, "acs_median_gross_rent":1487, "acs_rent_burden_pct":48.23,
        "acs_poverty_rate":12.1, "eviction_filing_rate":2.79, "scf_avg_household_debt":155600,
        "scf_credit_card_debt":6270, "scf_debt_to_income_ratio":1.41, "bls_unemployment_rate":4.2,
        "bls_median_wage":49500, "bls_job_growth_pct":1.3, "mit_living_wage_annual":23.32,
        "cdc_crowded_housing_pct":1.3, "usda_food_desert_indicator":17.4,
        "usda_low_income_low_access_pct":17.4, "usda_grocery_distance_score":1.8,
        "doe_total_energy_burden_pct":6.5, "doe_electricity_burden_pct":3.8, "doe_heating_burden_pct":1.7,
        "nyfed_credit_card_delinquency_rate":3.2, "nyfed_mortgage_delinquency_rate":1.3
    }])
    source = "Demo benchmark"

df = add_names(engineer_features(clean_dataframe(raw)))

page = st.sidebar.radio(
    "Navigation",
    ["Executive Dashboard", "Community Profile", "Risk Drivers", "Model Analytics", "Recommendations", "About"]
)

st.sidebar.markdown("---")
st.sidebar.metric("Profiles loaded", f"{len(df):,}")
st.sidebar.metric("Features available", f"{len(df.columns):,}")
st.sidebar.caption("Synthetic profiles are demonstration scenarios until real county-level rows are added.")

selected = st.sidebar.selectbox("Select community profile", df["display_name"].tolist())
row = df[df["display_name"] == selected].iloc[0]

# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="hero">
  <span class="badge">LIVE WEB APP</span>
  <span class="badge">11 PUBLIC DATA SOURCES</span>
  <span class="badge">EXPLAINABLE RISK ENGINE</span>
  <span class="badge">POLICY RECOMMENDATIONS</span>
  <div class="hero-title">🛡️ DebtShield AI</div>
  <div class="hero-sub">
    A modern financial vulnerability platform that identifies high-risk community profiles,
    explains the drivers behind distress, and recommends targeted interventions.
  </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Pages
# -----------------------------
if page == "Executive Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    high = int((df["risk_level"]=="High").sum())
    med = int((df["risk_level"]=="Medium").sum())
    avg = df["derived_financial_distress_index"].mean()
    maxname = df.sort_values("derived_financial_distress_index", ascending=False).iloc[0]["display_name"]

    c1.markdown(f'<div class="card"><div class="kpi-label">Profiles analyzed</div><div class="kpi-value">{len(df):,}</div><div class="kpi-sub">{source}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="kpi-label">Average index</div><div class="kpi-value">{avg:.1f}</div><div class="kpi-sub">Dataset-wide metric</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="kpi-label">High-risk flags</div><div class="kpi-value">{high:,}</div><div class="kpi-sub">{med:,} moderate-risk profiles</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><div class="kpi-label">Data architecture</div><div class="kpi-value">11</div><div class="kpi-sub">Sources integrated</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="note"><b>Dashboard behavior:</b> These overview charts summarize the entire uploaded dataset. They stay stable when you switch profiles and change only when a new CSV is uploaded.</div>', unsafe_allow_html=True)

    left, right = st.columns([1,1])
    with left:
        st.pyplot(plot_risk_counts(df))
    with right:
        st.pyplot(plot_index_hist(df))

    st.markdown("### Priority watchlist")
    watch = df.sort_values("derived_financial_distress_index", ascending=False)[
        ["display_name", "risk_level", "derived_financial_distress_index", "derived_debt_stress_score", "derived_housing_stress_score"]
    ].head(10).rename(columns={
        "display_name":"Community profile",
        "risk_level":"Risk",
        "derived_financial_distress_index":"Index",
        "derived_debt_stress_score":"Debt stress",
        "derived_housing_stress_score":"Housing stress"
    })
    st.dataframe(watch, use_container_width=True, hide_index=True)

elif page == "Community Profile":
    st.markdown(f"## {selected}")
    risk = row["risk_level"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Financial Distress Index", f"{row['derived_financial_distress_index']:.2f}")
    c2.markdown(f'<div class="card-light"><div class="kpi-label">Risk level</div><div class="kpi-value" style="color:#0f172a;">{risk_icon(risk)} {risk}</div><div class="kpi-sub" style="color:#475569;">Selected profile</div></div>', unsafe_allow_html=True)
    c3.metric("Top Driver", top_driver(row)[0])
    c4.metric("Rent-to-Income", f"{row['derived_rent_to_income_ratio']:.2f}%")

    labels = ["Housing", "Debt", "Cost", "Energy", "Food"]
    vals = [
        row["derived_housing_stress_score"], row["derived_debt_stress_score"],
        row["derived_cost_pressure_score"], row["derived_energy_stress_score"],
        row["derived_food_access_risk_score"]
    ]
    st.pyplot(plot_bar(labels, vals, "Selected profile stress breakdown"))

    driver, val = top_driver(row)
    if risk == "High":
        st.error(f"This profile is high risk. The strongest driver is {driver} with a score of {val:.1f}.")
    elif risk == "Medium":
        st.warning(f"This profile is moderate risk. The strongest driver is {driver} with a score of {val:.1f}.")
    else:
        st.success(f"This profile is lower risk. The strongest driver is {driver} with a score of {val:.1f}.")

elif page == "Risk Drivers":
    st.markdown("## Risk driver analysis")
    d = pd.DataFrame({
        "Driver":["Housing stress","Debt stress","Cost pressure","Energy stress","Food access risk"],
        "Score":[
            row["derived_housing_stress_score"], row["derived_debt_stress_score"],
            row["derived_cost_pressure_score"], row["derived_energy_stress_score"],
            row["derived_food_access_risk_score"]
        ]
    }).sort_values("Score", ascending=False)

    left, right = st.columns([1.1,.9])
    with left:
        st.pyplot(plot_bar(d["Driver"], d["Score"], "Ranked drivers for selected profile"))
    with right:
        st.dataframe(d, use_container_width=True, hide_index=True)
        st.success(f"Primary issue: {d.iloc[0]['Driver']}")

elif page == "Model Analytics":
    st.markdown("## Model analytics")
    st.markdown('<div class="warn"><b>Prototype note:</b> Synthetic profiles are used for machine-learning demonstration. The next major upgrade is replacing these profiles with real county-level rows.</div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric("Minimum index", f"{df['derived_financial_distress_index'].min():.1f}")
    b.metric("Median index", f"{df['derived_financial_distress_index'].median():.1f}")
    c.metric("Maximum index", f"{df['derived_financial_distress_index'].max():.1f}")

    left, right = st.columns([1,1])
    with left:
        st.pyplot(plot_index_hist(df))
    with right:
        corr_cols = [
            "derived_financial_distress_index","derived_housing_stress_score","derived_debt_stress_score",
            "derived_cost_pressure_score","derived_energy_stress_score","derived_food_access_risk_score"
        ]
        corr = df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(corr)
        labs = ["Index","Housing","Debt","Cost","Energy","Food"]
        ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=35, ha="right"); ax.set_yticklabels(labs)
        fig.colorbar(im, ax=ax)
        ax.set_title("Risk factor correlation matrix", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

elif page == "Recommendations":
    st.markdown("## Recommendation engine")
    st.write(f"Selected profile: **{selected}**")
    st.metric("Risk classification", f"{risk_icon(row['risk_level'])} {row['risk_level']}")

    for title, text in recs(row):
        st.markdown(f"### {title}")
        st.write(text)

    st.download_button(
        "Download scored dataset",
        data=df.to_csv(index=False),
        file_name="debtshield_ai_scored_dataset.csv",
        mime="text/csv"
    )

elif page == "About":
    st.markdown("## About DebtShield AI")
    st.write(
        "DebtShield AI integrates housing, debt, labor, food access, energy burden, and social vulnerability indicators into one financial risk platform. "
        "It generates a Financial Distress Index, explains the strongest risk drivers, and produces policy recommendations."
    )
    st.markdown("### Why the names are not real places")
    st.write(
        "The current version uses synthetic demonstration profiles. Names such as 'High-Risk Debt-Stressed Community' are practical profile labels, not real counties. "
        "This makes the app easier to understand while the next version is expanded to real county-level observations."
    )
    st.markdown("### Data sources")
    st.write("ACS, Zillow, Eviction Lab, Federal Reserve SCF, BLS, MIT Living Wage, CDC SVI, FHFA, USDA Food Access, DOE LEAD, and NY Fed Consumer Credit.")
    st.markdown("### Future upgrades")
    st.write("- Real county-level rows")
    st.write("- Pennsylvania and U.S. risk maps")
    st.write("- Future risk simulator")
    st.write("- SHAP explanation panel")

st.markdown('<div class="footer">DebtShield AI v4 · Financial vulnerability intelligence platform · Prototype for research and demonstration.</div>', unsafe_allow_html=True)
