import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import plotly.graph_objects as go
import hashlib
import joblib
from pathlib import Path

st.set_page_config(
    page_title="DebtShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = Path(__file__).resolve().parent
WEEK1_PATH = APP_DIR / "week1_cleaned_financial_distress_dataset.csv"
REAL_DATA_PATH = APP_DIR / "real_county_data.csv"
RF_MODEL_PATH = APP_DIR / "phase2_best_random_forest_model.pkl"
MODEL_COMPARISON_PATH = APP_DIR / "phase2_model_comparison_results.csv"
FEATURE_IMPORTANCE_PATH = APP_DIR / "phase2_feature_importance.csv"

# ============================================================
# DebtShield AI V2 — real county data, selectors, comparison,
# model connection, scenario simulator, map, improved chatbot
# ============================================================

st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 85% 0%, rgba(147,197,253,0.18), transparent 26%),
      linear-gradient(135deg, #f8fbff 0%, #eef6ff 42%, #ffffff 100%);
}
.block-container { padding-top: 1.05rem; padding-bottom: 2.5rem; max-width: 1480px; }
h1, h2, h3, h4, p, div { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #0f172a; }
p, li, span { color: #1e293b; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #ffffff 0%, #f8fbff 55%, #eef6ff 100%); border-right: 1px solid #dbeafe; }
[data-testid="stSidebar"] * { color: #0f172a !important; }
.sidebar-title { display:flex; align-items:center; gap:.6rem; margin-bottom:.1rem; }
.sidebar-logo { font-size:2.2rem; }
.sidebar-name { font-size:1.35rem; font-weight:950; color:#0f172a; }
.sidebar-subtitle { color:#64748b; font-size:.92rem; line-height:1.35; margin-bottom:1.2rem; }
.hero {
    position: relative; overflow: hidden; padding: 1.35rem 1.65rem 1.55rem 1.65rem; border-radius: 30px;
    background: radial-gradient(circle at 12% 18%, rgba(59,130,246,.17), transparent 27%),
      radial-gradient(circle at 94% 84%, rgba(217,70,239,.22), transparent 33%),
      linear-gradient(135deg, #f8fbff 0%, #e8f2ff 48%, #dbeafe 100%);
    border: 1px solid #bfdbfe; box-shadow: 0 22px 46px rgba(59,130,246,.19); margin-bottom: 1.05rem;
}
.badge { display:inline-block; padding:.38rem .72rem; border-radius:999px; background:linear-gradient(135deg,#dbeafe,#bfdbfe);
    border:1px solid #93c5fd; color:#0757bd; font-size:.78rem; font-weight:950; margin-right:.38rem; margin-bottom:.62rem; }
.hero-main { display:flex; align-items:center; gap:1rem; }
.hero-logo { font-size:3.2rem; }
.hero-title { font-size: 3.1rem; font-weight: 950; color: #0f172a; letter-spacing: -0.035em; }
.hero-sub { font-size: 1.04rem; color: #334155; max-width: 970px; line-height: 1.58; margin-top:.45rem; }
.kpi-card { display: flex; align-items: center; gap: 1rem; background: rgba(255,255,255,.94); border: 1px solid #dbeafe;
    border-radius: 24px; padding: 1.08rem 1.12rem; box-shadow: 0 14px 29px rgba(15,23,42,.08); min-height: 120px; }
.kpi-icon { width: 55px; height: 55px; border-radius: 19px; display: flex; align-items: center; justify-content: center;
    font-size: 1.65rem; background: linear-gradient(135deg, #dbeafe, #bfdbfe); }
.kpi-icon.green { background: linear-gradient(135deg, #dcfce7, #bbf7d0); }
.kpi-icon.red { background: linear-gradient(135deg, #fee2e2, #fecaca); }
.kpi-icon.purple { background: linear-gradient(135deg, #ede9fe, #ddd6fe); }
.kpi-label { font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; color:#64748b; font-weight:950; }
.kpi-value { font-size:2rem; font-weight:950; color:#0f172a; line-height:1.1; }
.kpi-sub { font-size:.87rem; color:#475569; }
.chart-card { background: rgba(255,255,255,.96); border: 1px solid #e2e8f0; border-radius: 24px; padding: 1rem; box-shadow: 0 12px 28px rgba(15,23,42,.075); }
.note { padding:.9rem 1rem; border-radius:18px; background: linear-gradient(135deg,#eff6ff,#f8fbff); border:1px solid #bfdbfe; color:#1e3a8a; margin:.8rem 0 1rem 0; }
.warn { padding:.9rem 1rem; border-radius:18px; background: linear-gradient(135deg,#fff7ed,#fffbeb); border:1px solid #fed7aa; color:#92400e; margin:.8rem 0; }
.error-box { padding:1rem 1.1rem; border-radius:18px; background: linear-gradient(135deg,#fef2f2,#fff1f2); border:1px solid #fecaca; color:#991b1b; margin:.9rem 0; }
.info-card { background: rgba(255,255,255,.97); border: 1px solid #dbeafe; border-radius: 22px; padding: 1rem 1.1rem; box-shadow: 0 10px 24px rgba(15,23,42,.06); color: #0f172a !important; }
.info-card * { color: #0f172a !important; }
.footer { font-size:.8rem; color:#64748b; margin-top:1.2rem; }
/* Force dropdown menus (state/county/search selects, multiselect) to always
   have a light background with dark text - prevents invisible black-on-black
   text if a visitor's system is in dark mode. */
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #ffffff !important;
}
[data-baseweb="menu"] li, [role="option"], [role="listbox"] * {
    background: #ffffff !important;
    color: #0f172a !important;
}
[role="option"]:hover, [data-baseweb="menu"] li:hover {
    background: #eaf2ff !important;
}
[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #0f172a !important;
}
@media (max-width: 640px) {
    .hero-title { font-size: 2rem; }
    .kpi-card { min-height: auto; padding: .8rem; }
    .block-container { padding-left: .6rem; padding-right: .6rem; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Data loading (reliable, validated, no silent fallback)
# -----------------------------
REQUIRED_COLUMNS = ["state", "county", "acs_median_household_income", "acs_poverty_rate"]

@st.cache_data
def load_real_county_data():
    """Loads the real, Census-sourced multi-county dataset if present."""
    if REAL_DATA_PATH.exists():
        df = pd.read_csv(REAL_DATA_PATH)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            return None, f"real_county_data.csv is missing required columns: {missing}"
        df = df.dropna(subset=["county", "state", "acs_median_household_income"])
        if len(df) == 0:
            return None, "real_county_data.csv has no usable rows after validation."
        return df, None
    return None, None

@st.cache_data
def load_legacy_benchmark():
    """Loads the Week 1 file's one real benchmark row only (never duplicates it)."""
    if WEEK1_PATH.exists():
        raw = pd.read_csv(WEEK1_PATH)
        raw = raw.dropna(subset=["county"]) if "county" in raw.columns else raw.iloc[0:0]
        return raw
    return pd.DataFrame()

def get_source_dataset():
    real_df, err = load_real_county_data()
    if err:
        st.markdown(f'<div class="error-box"><b>Data error:</b> {err} Falling back to the single national benchmark row until this is fixed.</div>', unsafe_allow_html=True)
    if real_df is not None and len(real_df) > 0:
        return real_df, f"Real Census ACS data ({len(real_df)} counties)"
    legacy = load_legacy_benchmark()
    if len(legacy) > 0:
        return legacy, "Legacy benchmark (1 row — upload real_county_data.csv for full county coverage)"
    demo = pd.DataFrame([{
        "state": "United States", "county": "National Average", "year": 2025,
        "acs_median_household_income": 81604, "acs_median_gross_rent": 1487, "acs_rent_burden_pct": 48.23,
        "acs_poverty_rate": 12.1, "bls_unemployment_rate": 4.2
    }])
    return demo, "Emergency demo benchmark (no data files found)"

# -----------------------------
# Feature engineering
# -----------------------------
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
    return {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")

def clean_dataframe(df):
    """Drops genuinely incomplete rows instead of median-filling them into
    duplicates of the only complete row (the V1 bug)."""
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    out = out.dropna(subset=["county", "state"]).reset_index(drop=True)
    nums = out.select_dtypes(include=[np.number]).columns
    for c in nums:
        out[c] = out[c].fillna(out[c].median())
    # Avoid select_dtypes(include=["object"]): in recent pandas versions this
    # also matches PyArrow-backed string columns, and reassigning into those
    # via out[c] = ... segfaults (a real pandas/PyArrow bug, not a fluke).
    # Casting explicitly to plain object dtype first sidesteps it.
    text_cols = [c for c in out.columns if c not in nums]
    for c in text_cols:
        out[c] = out[c].astype(object).fillna("Unknown")
    return out.reset_index(drop=True)

def engineer_features(df):
    out = df.copy()
    income = get_col(out, "acs_median_household_income", 81604).replace(0, np.nan)
    annual_rent = get_col(out, "acs_median_gross_rent", 1487) * 12
    out["derived_rent_to_income_ratio"] = (annual_rent / income * 100).fillna(0)

    out["derived_housing_stress_score"] = (
        0.45 * score_between(get_col(out, "acs_rent_burden_pct"), 20, 60) +
        0.35 * score_between(out["derived_rent_to_income_ratio"], 15, 45) +
        0.20 * score_between(get_col(out, "eviction_filing_rate"), 0, 8)
    ).clip(0, 100)

    debt_amount_ratio = (get_col(out, "scf_avg_household_debt") / income).fillna(0)
    out["derived_debt_stress_score"] = (
        0.40 * score_between(get_col(out, "scf_debt_to_income_ratio"), 0.4, 2.5) +
        0.30 * score_between(debt_amount_ratio, 0.5, 3.0) +
        0.30 * score_between(get_col(out, "nyfed_credit_card_delinquency_rate"), 0, 8)
    ).clip(0, 100)

    out["derived_cost_pressure_score"] = (
        0.40 * score_between(get_col(out, "acs_poverty_rate"), 5, 30) +
        0.40 * score_between(get_col(out, "bls_unemployment_rate"), 2, 12) +
        0.20 * inverse_score_between(get_col(out, "bls_job_growth_pct"), -5, 5)
    ).clip(0, 100)

    out["derived_energy_stress_score"] = score_between(get_col(out, "doe_total_energy_burden_pct", 6.5), 2, 12).clip(0, 100)
    out["derived_food_access_risk_score"] = score_between(get_col(out, "usda_low_income_low_access_pct", 17.4), 0, 30).clip(0, 100)

    # Only weight categories that have real underlying source columns, so a
    # missing data source (e.g. no SCF debt data yet) doesn't silently dilute
    # the index toward the middle for every county uniformly.
    base_weights = {"housing": 0.30, "debt": 0.25, "cost": 0.20, "energy": 0.15, "food": 0.10}
    has_debt = any(c in df.columns for c in ["scf_avg_household_debt", "scf_debt_to_income_ratio", "nyfed_credit_card_delinquency_rate"])
    has_energy = "doe_total_energy_burden_pct" in df.columns
    has_food = "usda_low_income_low_access_pct" in df.columns
    active_weights = {"housing": base_weights["housing"], "cost": base_weights["cost"]}
    if has_debt:
        active_weights["debt"] = base_weights["debt"]
    if has_energy:
        active_weights["energy"] = base_weights["energy"]
    if has_food:
        active_weights["food"] = base_weights["food"]
    weight_total = sum(active_weights.values())
    norm_weights = {k: v / weight_total for k, v in active_weights.items()}

    score_map = {
        "housing": out["derived_housing_stress_score"],
        "debt": out["derived_debt_stress_score"],
        "cost": out["derived_cost_pressure_score"],
        "energy": out["derived_energy_stress_score"],
        "food": out["derived_food_access_risk_score"],
    }
    out["derived_financial_distress_index"] = sum(
        norm_weights[k] * score_map[k] for k in active_weights
    ).clip(0, 100)
    out["risk_level"] = out["derived_financial_distress_index"].apply(risk_label)
    out["display_name"] = out["county"].astype(str) + ", " + out["state"].astype(str)
    return out

def top_driver(row):
    drivers = {
        "Housing": row.get("derived_housing_stress_score", 0),
        "Debt": row.get("derived_debt_stress_score", 0),
        "Cost of Living": row.get("derived_cost_pressure_score", 0),
        "Energy": row.get("derived_energy_stress_score", 0),
        "Food Access": row.get("derived_food_access_risk_score", 0),
    }
    key = max(drivers, key=drivers.get)
    return key, drivers[key]

def recs(row):
    out = []
    if row.get("derived_housing_stress_score", 0) >= 50:
        out.append(("🏠 Housing", "Expand rental assistance, eviction diversion, and affordable housing supply."))
    if row.get("derived_debt_stress_score", 0) >= 50:
        out.append(("💳 Debt", "Offer debt counseling, delinquency outreach, and emergency cash support."))
    if row.get("derived_cost_pressure_score", 0) >= 50:
        out.append(("💼 Cost of Living", "Increase benefits enrollment, job placement, and basic-needs assistance."))
    if row.get("derived_energy_stress_score", 0) >= 50:
        out.append(("⚡ Energy", "Increase utility assistance and weatherization support."))
    if row.get("derived_food_access_risk_score", 0) >= 50:
        out.append(("🥫 Food Access", "Support grocery access, mobile markets, and SNAP outreach."))
    if not out:
        out.append(("✅ Monitoring", "Maintain prevention programs and continue monitoring."))
    return out

# -----------------------------
# Random Forest model connection
# -----------------------------
@st.cache_resource
def load_rf_model():
    if RF_MODEL_PATH.exists():
        try:
            return joblib.load(RF_MODEL_PATH)
        except Exception:
            return None
    return None

def rf_predict(model, row, feature_cols):
    try:
        x = pd.DataFrame([{c: row.get(c, 0) for c in feature_cols}])
        pred = model.predict(x)[0]
        proba = model.predict_proba(x).max() if hasattr(model, "predict_proba") else None
        return pred, proba
    except Exception:
        return None, None

# -----------------------------
# Chatbot
# -----------------------------
def debtshield_chatbot(question, row, selected, df=None):
    q = str(question).strip().lower()
    risk = str(row.get("risk_level", "Unknown"))
    index = float(row.get("derived_financial_distress_index", 0))
    driver, driver_score = top_driver(row)

    scores = {
        "Housing": float(row.get("derived_housing_stress_score", 0)),
        "Debt": float(row.get("derived_debt_stress_score", 0)),
        "Cost of Living": float(row.get("derived_cost_pressure_score", 0)),
        "Energy": float(row.get("derived_energy_stress_score", 0)),
        "Food Access": float(row.get("derived_food_access_risk_score", 0)),
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    disclaimer = " *DebtShield AI provides educational and analytical information, not legal, financial, or government-benefit advice.*"

    if not q:
        return "Ask me about this county's risk level, strongest drivers, recommended interventions, or try comparing two counties."

    # Compare two named counties ("compare X and Y" or "X vs Y")
    if df is not None:
        parts = None
        if "compare" in q and " and " in q:
            parts = q.split("compare")[-1].split(" and ")
        elif " vs " in q:
            parts = q.split(" vs ")
        elif " versus " in q:
            parts = q.split(" versus ")
        if parts and len(parts) == 2:
            a_name, b_name = parts[0].strip(), parts[1].strip()
            match_a = df[df["display_name"].str.lower().str.contains(a_name)]
            match_b = df[df["display_name"].str.lower().str.contains(b_name)]
            if len(match_a) and len(match_b):
                ra, rb = match_a.iloc[0], match_b.iloc[0]
                return (f"**{ra['display_name']}** is **{ra['risk_level']} risk** (index {ra['derived_financial_distress_index']:.1f}), "
                        f"while **{rb['display_name']}** is **{rb['risk_level']} risk** (index {rb['derived_financial_distress_index']:.1f}). "
                        f"{'The first county is under greater financial distress.' if ra['derived_financial_distress_index'] > rb['derived_financial_distress_index'] else 'The second county is under greater financial distress.'}" + disclaimer)

    # Riskiest / safest county in the loaded dataset
    if df is not None and any(w in q for w in ["riskiest", "worst", "highest risk", "most at risk"]):
        top_row = df.sort_values("derived_financial_distress_index", ascending=False).iloc[0]
        return (f"The highest-risk county currently loaded is **{top_row['display_name']}** "
                f"({top_row['risk_level']} risk, index {top_row['derived_financial_distress_index']:.1f}).") + disclaimer

    if df is not None and any(w in q for w in ["safest", "lowest risk", "least at risk", "best county"]):
        bottom_row = df.sort_values("derived_financial_distress_index", ascending=True).iloc[0]
        return (f"The lowest-risk county currently loaded is **{bottom_row['display_name']}** "
                f"({bottom_row['risk_level']} risk, index {bottom_row['derived_financial_distress_index']:.1f}).") + disclaimer

    # What does a risk level actually mean
    if any(w in q for w in ["what does low", "what is low risk", "mean by low"]):
        return "**Low risk** means financial pressure (housing, cost of living, and related factors) is below typical levels for the counties in this dataset." + disclaimer
    if any(w in q for w in ["what does medium", "what is medium risk", "what does moderate", "mean by medium", "mean by moderate"]):
        return "**Medium risk** means the county shows some meaningful financial pressure, but it isn't severe or compounding across categories." + disclaimer
    if any(w in q for w in ["what does high", "what is high risk", "mean by high"]):
        return "**High risk** means the county shows severe financial pressure, often compounding across several categories (e.g. both housing and cost-of-living stress)." + disclaimer

    if any(w in q for w in ["financial distress index", "what is fdi", "explain the index", "what is the index"]):
        return "The **Financial Distress Index** is a 0-100 score combining housing stress, debt stress, cost-of-living pressure, energy burden, and food access into one number. Higher = more financial pressure." + disclaimer

    if any(w in q for w in ["hello", "hi ", "hey"]) or q in ["hi", "hey"]:
        return f"Hello! I can explain **{selected}** — its risk level, drivers, and recommended actions. You can also ask me to compare it to another county."

    if any(w in q for w in ["summary", "summarize", "overview"]):
        return (f"**{selected}** is **{risk} risk** with a Financial Distress Index of **{index:.1f}**. "
                f"Its strongest driver is **{driver} ({driver_score:.1f})**.") + disclaimer

    if any(w in q for w in ["why", "driver", "cause", "highest", "reason"]):
        return (f"The strongest driver is **{driver}** ({driver_score:.1f}). Next: **{ranked[1][0]} ({ranked[1][1]:.1f})** "
                f"and **{ranked[2][0]} ({ranked[2][1]:.1f})**.") + disclaimer

    if any(w in q for w in ["recommend", "action", "solution", "priority", "help", "fix", "improve", "policy"]):
        items = recs(row)
        actions = " ".join([f"**{t}:** {x}" for t, x in items])
        return f"Priority: **{driver}**. {actions}" + disclaimer

    for key in ["housing", "rent", "debt", "credit", "cost", "poverty", "unemploy", "energy", "utility", "food", "grocery"]:
        if key in q:
            label_map = {"housing": "Housing", "rent": "Housing", "debt": "Debt", "credit": "Debt",
                         "cost": "Cost of Living", "poverty": "Cost of Living", "unemploy": "Cost of Living",
                         "energy": "Energy", "utility": "Energy", "food": "Food Access", "grocery": "Food Access"}
            label = label_map[key]
            return f"{label} stress for {selected} is **{scores[label]:.1f}** out of 100." + disclaimer

    return (f"For **{selected}**, the main issue is **{driver} ({driver_score:.1f})**, overall risk **{risk}**. "
            f"Try: \u201cWhy is this county at risk?\u201d, \u201c{selected} vs [another county]\u201d, \u201cwhich county is riskiest?\u201d, "
            f"or \u201cwhat does medium risk mean?\u201d") + disclaimer

# -----------------------------
# Charts
# -----------------------------
def plot_bar(labels, values, title, ylabel="Score", cap=110):
    fig = Figure(figsize=(9, 4.6))
    ax = fig.subplots()
    bars = ax.bar(labels, values)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, cap)
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=.22)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig

def plot_risk_counts(df):
    counts = df["risk_level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig = Figure(figsize=(7, 4.3))
    ax = fig.subplots()
    bars = ax.bar(counts.index, counts.values)
    ax.set_title("Risk distribution across loaded counties", fontweight="bold", fontsize=14)
    ax.set_ylabel("Counties")
    ax.grid(axis="y", alpha=.22)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.2, f"{int(h)}", ha="center")
    fig.tight_layout()
    return fig

COUNTY_CENTROIDS = {
    "loudoun": (39.115, -77.564), "fairfax": (38.846, -77.306), "marin": (38.065, -122.727),
    "santa clara": (37.339, -121.895), "nassau": (40.749, -73.641), "maricopa": (33.448, -112.074),
    "cook": (41.878, -87.630), "harris": (29.760, -95.370), "wayne": (42.331, -83.046),
    "franklin": (39.961, -82.999), "los angeles": (34.052, -118.244), "philadelphia": (39.953, -75.165),
    "mcdowell": (37.431, -81.585), "starr": (26.379, -98.820), "owsley": (37.475, -83.700),
    "zavala": (28.687, -99.828), "wilcox": (31.993, -87.290), "east carroll": (32.803, -91.166),
}
RISK_COLOR = {"Low": "#16a34a", "Medium": "#eab308", "High": "#dc2626"}

def build_map(df):
    lats, lons, texts, colors = [], [], [], []
    for _, row in df.iterrows():
        key = str(row["county"]).lower().replace(" county", "").replace(" parish", "").strip()
        if key in COUNTY_CENTROIDS:
            lat, lon = COUNTY_CENTROIDS[key]
            lats.append(lat)
            lons.append(lon)
            texts.append(f"{row['display_name']}<br>Risk: {row['risk_level']}<br>Index: {row['derived_financial_distress_index']:.1f}")
            colors.append(RISK_COLOR.get(row["risk_level"], "#94a3b8"))
    fig = go.Figure(go.Scattergeo(
        lon=lons, lat=lats, text=texts, mode="markers",
        marker=dict(size=14, color=colors, line=dict(width=1, color="white")),
        hoverinfo="text"
    ))
    fig.update_layout(
        geo=dict(scope="usa", showland=True, landcolor="#f1f5f9", subunitcolor="#cbd5e1"),
        margin=dict(l=0, r=0, t=0, b=0), height=480
    )
    return fig

# -----------------------------
# Load data
# -----------------------------
st.sidebar.markdown("""
<div class="sidebar-title"><div class="sidebar-logo">🛡️</div><div><div class="sidebar-name">DebtShield AI</div></div></div>
<div class="sidebar-subtitle">Financial vulnerability intelligence platform — V2</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("⚙️ Advanced: upload a different CSV"):
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded is not None:
    raw = pd.read_csv(uploaded)
    source = "Uploaded dataset"
else:
    raw, source = get_source_dataset()

try:
    df = engineer_features(clean_dataframe(raw))
except Exception as e:
    st.markdown(f'<div class="error-box"><b>The dataset could not be processed.</b> {e}<br>Check that your CSV includes at minimum: {REQUIRED_COLUMNS}</div>', unsafe_allow_html=True)
    st.stop()

if len(df) == 0:
    st.markdown('<div class="error-box"><b>No usable rows found.</b> Please check that the built-in dataset is included in the GitHub repository next to this app file.</div>', unsafe_allow_html=True)
    st.stop()

rf_model = load_rf_model()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Executive Dashboard", "👥 County Profile", "⚖️ Compare Counties", "📊 Risk Drivers",
     "🎛️ Scenario Simulator", "🗺️ Risk Map", "📈 Model Performance", "🧭 Recommendations", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Source: {source}")
st.sidebar.metric("Counties loaded", f"{len(df):,}")

states = sorted(df["state"].dropna().unique().tolist())

if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "recently_viewed" not in st.session_state:
    st.session_state.recently_viewed = []

search_term = st.sidebar.text_input("🔎 Search any county", placeholder="e.g. Cook, Maricopa...")
if st.sidebar.button("↺ Reset filters"):
    search_term = ""
    st.session_state.pop("state_select", None)
    st.session_state.pop("county_select", None)
    st.rerun()

if search_term:
    matches = df[df["display_name"].str.contains(search_term, case=False, na=False)]
    if len(matches) > 0:
        selected = st.sidebar.selectbox("Search results", sorted(matches["display_name"].tolist()))
        sel_state = df[df["display_name"] == selected].iloc[0]["state"]
    else:
        st.sidebar.warning("No county matches that search.")
        sel_state = states[0]
        selected = sorted(df[df["state"] == sel_state]["display_name"].tolist())[0]
else:
    sel_state = st.sidebar.selectbox("State", states, key="state_select")
    county_options = sorted(df[df["state"] == sel_state]["display_name"].tolist())
    selected = st.sidebar.selectbox("County", county_options, key="county_select")

# Always defined, regardless of which branch above ran - other pages (e.g.
# Compare Counties) reference this unconditionally.
county_options = sorted(df[df["state"] == sel_state]["display_name"].tolist())

if st.session_state.favorites:
    fav_pick = st.sidebar.selectbox("⭐ Favorites", ["—"] + st.session_state.favorites)
    if fav_pick != "—":
        selected = fav_pick

if st.session_state.recently_viewed:
    st.sidebar.caption("🕓 Recently viewed: " + ", ".join(st.session_state.recently_viewed[-5:]))

row = df[df["display_name"] == selected].iloc[0]

fav_col1, fav_col2 = st.sidebar.columns(2)
if fav_col1.button("⭐ Save" if selected not in st.session_state.favorites else "★ Saved"):
    if selected not in st.session_state.favorites:
        st.session_state.favorites.append(selected)
if fav_col2.button("✕ Unsave") and selected in st.session_state.favorites:
    st.session_state.favorites.remove(selected)

if not st.session_state.recently_viewed or st.session_state.recently_viewed[-1] != selected:
    st.session_state.recently_viewed.append(selected)

st.markdown("""
<div class="hero"><div class="hero-content">
<span class="badge">🌐 LIVE WEB APP</span><span class="badge">🗄️ REAL CENSUS DATA</span>
<span class="badge">🧠 EXPLAINABLE RISK ENGINE</span><span class="badge">🤖 RF MODEL CONNECTED</span>
<div class="hero-main"><div class="hero-logo">🛡️</div><div class="hero-title">DebtShield AI</div></div>
<div class="hero-sub">A financial vulnerability platform that identifies high-risk counties, explains the drivers behind distress, and recommends targeted interventions.</div>
</div></div>
""", unsafe_allow_html=True)

# -----------------------------
# Pages
# -----------------------------
if page == "🏠 Executive Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    high = int((df["risk_level"] == "High").sum())
    med = int((df["risk_level"] == "Medium").sum())
    avg = df["derived_financial_distress_index"].mean()
    c1.markdown(f'<div class="kpi-card"><div class="kpi-icon purple">👥</div><div><div class="kpi-label">Counties</div><div class="kpi-value">{len(df):,}</div><div class="kpi-sub">{source}</div></div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-icon green">📈</div><div><div class="kpi-label">Average index</div><div class="kpi-value">{avg:.1f}</div></div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-icon red">🚩</div><div><div class="kpi-label">High-risk</div><div class="kpi-value">{high:,}</div><div class="kpi-sub">{med:,} medium-risk</div></div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-icon">🗄️</div><div><div class="kpi-label">Model</div><div class="kpi-value">{"✅" if rf_model else "—"}</div><div class="kpi-sub">{"RF connected" if rf_model else "Not loaded"}</div></div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="note" style="display:flex; gap:1.5rem; flex-wrap:wrap; align-items:center;">
    <b>What the risk levels mean:</b>
    <span>🟢 <b>Low</b> — financial pressure below typical levels</span>
    <span>🟡 <b>Medium</b> — some meaningful financial pressure</span>
    <span>🔴 <b>High</b> — severe, compounding financial pressure</span>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.pyplot(plot_risk_counts(df))
    with right:
        watch = df.sort_values("derived_financial_distress_index", ascending=False)[
            ["display_name", "risk_level", "derived_financial_distress_index"]
        ].head(10).rename(columns={"display_name": "County", "risk_level": "Risk", "derived_financial_distress_index": "Index"})
        st.dataframe(watch, use_container_width=True, hide_index=True)

elif page == "👥 County Profile":
    st.markdown(f"## {selected}")
    risk = row["risk_level"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Financial Distress Index", f"{row['derived_financial_distress_index']:.2f}")
    c2.metric("Risk Level", f"{risk_icon(risk)} {risk}")
    c3.metric("Top Driver", top_driver(row)[0])

    labels = ["Housing", "Debt", "Cost", "Energy", "Food"]
    vals = [row["derived_housing_stress_score"], row["derived_debt_stress_score"],
            row["derived_cost_pressure_score"], row["derived_energy_stress_score"], row["derived_food_access_risk_score"]]
    st.pyplot(plot_bar(labels, vals, "Stress breakdown"))

elif page == "⚖️ Compare Counties":
    st.markdown("## Compare counties")
    choices = st.multiselect("Select 2–4 counties", df["display_name"].tolist(), default=county_options[:2] if len(county_options) >= 2 else df["display_name"].tolist()[:2])
    if len(choices) >= 2:
        comp = df[df["display_name"].isin(choices)]
        metrics = ["derived_financial_distress_index", "derived_housing_stress_score", "derived_debt_stress_score",
                   "derived_cost_pressure_score", "derived_energy_stress_score", "derived_food_access_risk_score"]
        labels = ["Overall Index", "Housing", "Debt", "Cost", "Energy", "Food"]
        fig = Figure(figsize=(10, 5))
        ax = fig.subplots()
        x = np.arange(len(labels))
        width = 0.8 / len(comp)
        for i, (_, r) in enumerate(comp.iterrows()):
            ax.bar(x + i * width, [r[m] for m in metrics], width, label=r["display_name"])
        ax.set_xticks(x + width * (len(comp) - 1) / 2)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=.22)
        st.pyplot(fig)
        st.dataframe(comp[["display_name", "risk_level"] + metrics].rename(columns={"display_name": "County", "risk_level": "Risk"}), use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 counties to compare.")

elif page == "📊 Risk Drivers":
    st.markdown("## Risk driver analysis")
    d = pd.DataFrame({
        "Driver": ["Housing stress", "Debt stress", "Cost pressure", "Energy stress", "Food access risk"],
        "Score": [row["derived_housing_stress_score"], row["derived_debt_stress_score"],
                  row["derived_cost_pressure_score"], row["derived_energy_stress_score"], row["derived_food_access_risk_score"]]
    }).sort_values("Score", ascending=False)
    left, right = st.columns([1.1, .9])
    with left:
        st.pyplot(plot_bar(d["Driver"], d["Score"], f"Ranked drivers — {selected}"))
    with right:
        st.dataframe(d, use_container_width=True, hide_index=True)

elif page == "🎛️ Scenario Simulator":
    st.markdown(f"## Scenario simulator — {selected}")
    st.markdown('<div class="note">Adjust inputs below to estimate how the risk index would change. Results are estimates, not forecasts.</div>', unsafe_allow_html=True)
    base_income = float(row.get("acs_median_household_income", 60000))
    base_poverty = float(row.get("acs_poverty_rate", 12))
    base_unemp = float(row.get("bls_unemployment_rate", 4))
    base_rent_burden = float(row.get("acs_rent_burden_pct", 30))

    c1, c2 = st.columns(2)
    with c1:
        sim_income = st.slider("Median household income ($)", 15000, 250000, int(base_income), step=1000)
        sim_poverty = st.slider("Poverty rate (%)", 0.0, 45.0, base_poverty, step=0.5)
    with c2:
        sim_unemp = st.slider("Unemployment rate (%)", 0.0, 20.0, base_unemp, step=0.1)
        sim_rent_burden = st.slider("Rent burden (%)", 0.0, 80.0, base_rent_burden, step=0.5)

    sim_row = row.copy()
    sim_row["acs_median_household_income"] = sim_income
    sim_row["acs_poverty_rate"] = sim_poverty
    sim_row["bls_unemployment_rate"] = sim_unemp
    sim_row["acs_rent_burden_pct"] = sim_rent_burden
    sim_df = engineer_features(clean_dataframe(pd.DataFrame([sim_row])))
    sim_result = sim_df.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Original index", f"{row['derived_financial_distress_index']:.1f}")
    c2.metric("Simulated index", f"{sim_result['derived_financial_distress_index']:.1f}",
              delta=f"{sim_result['derived_financial_distress_index'] - row['derived_financial_distress_index']:.1f}")
    c3.metric("Simulated risk", f"{risk_icon(sim_result['risk_level'])} {sim_result['risk_level']}")

elif page == "🗺️ Risk Map":
    st.markdown("## County risk map")
    mapped = df[df["county"].str.lower().str.replace(" county", "", regex=False).str.replace(" parish", "", regex=False).str.strip().isin(COUNTY_CENTROIDS.keys())]
    if len(mapped) > 0:
        st.plotly_chart(build_map(df), use_container_width=True)
        st.caption("🟢 Low risk · 🟡 Medium risk · 🔴 High risk. Map currently covers counties with known centroids; expand COUNTY_CENTROIDS as more counties are added.")
    else:
        st.info("No mapped counties in the current dataset yet.")

elif page == "📈 Model Performance":
    st.markdown("## Model performance")
    if MODEL_COMPARISON_PATH.exists():
        comp = pd.read_csv(MODEL_COMPARISON_PATH)
        st.dataframe(comp, use_container_width=True, hide_index=True)
    else:
        st.markdown('<div class="warn">phase2_model_comparison_results.csv not found in the repository.</div>', unsafe_allow_html=True)
    if FEATURE_IMPORTANCE_PATH.exists():
        fi = pd.read_csv(FEATURE_IMPORTANCE_PATH).sort_values("Importance", ascending=False)
        st.pyplot(plot_bar(fi["Feature"].head(10), fi["Importance"].head(10), "Top feature importances", ylabel="Importance", cap=fi["Importance"].max() * 1.2))
    else:
        st.markdown('<div class="warn">phase2_feature_importance.csv not found.</div>', unsafe_allow_html=True)

    st.markdown("### Live prediction for selected county")
    if rf_model is not None:
        feature_cols = ["derived_housing_stress_score", "derived_cost_pressure_score", "derived_energy_stress_score",
                         "derived_debt_stress_score", "bls_unemployment_rate", "derived_food_access_risk_score",
                         "doe_total_energy_burden_pct", "acs_poverty_rate", "nyfed_credit_card_delinquency_rate",
                         "scf_debt_to_income_ratio", "derived_rent_to_income_ratio", "usda_low_income_low_access_pct",
                         "nyfed_mortgage_delinquency_rate"]
        pred, proba = rf_predict(rf_model, row, feature_cols)
        if pred is not None:
            st.success(f"Random Forest prediction: **{pred}**" + (f" (confidence {proba:.0%})" if proba else ""))
            st.caption("This is the trained model's independent prediction — distinct from the rule-based Financial Distress Index above, since the model was trained on a related but separate labeling process.")
        else:
            st.markdown('<div class="warn">The model could not score this county — its input columns don\'t match what the model expects. This is expected until the training feature set is finalized.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn">phase2_best_random_forest_model.pkl not found in the repository, or could not be loaded.</div>', unsafe_allow_html=True)

elif page == "🧭 Recommendations":
    st.markdown("## Recommendation engine")
    st.write(f"Selected county: **{selected}**")
    st.metric("Risk classification", f"{risk_icon(row['risk_level'])} {row['risk_level']}")
    for title, text in recs(row):
        st.markdown(f"### {title}")
        st.write(text)

    st.markdown("---")
    st.markdown("## 💬 Ask DebtShield")
    st.markdown('<div class="note">Ask about the selected county\'s risk, drivers, recommendations, or say "compare County A and County B".</div>', unsafe_allow_html=True)

    chat_key = f"debtshield_chat_{selected}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [{"role": "assistant", "content": f"I'm ready to explain **{selected}**. Ask about its risk, drivers, or try comparing it to another county."}]

    q1, q2, q3 = st.columns(3)
    quick_prompt = None
    if q1.button("Why is this at risk?", use_container_width=True):
        quick_prompt = "Why is this profile at risk?"
    if q2.button("What should be prioritized?", use_container_width=True):
        quick_prompt = "What should policymakers prioritize?"
    if q3.button("Short summary", use_container_width=True):
        quick_prompt = "Give me a short summary."

    for message in st.session_state[chat_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    typed_prompt = st.chat_input("Ask DebtShield...")
    prompt = typed_prompt or quick_prompt
    if prompt:
        st.session_state[chat_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        answer = debtshield_chatbot(prompt, row, selected, df)
        st.session_state[chat_key].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.download_button("Download scored dataset", data=df.to_csv(index=False), file_name="debtshield_ai_scored_dataset.csv", mime="text/csv")

elif page == "ℹ️ About":
    st.markdown("## About DebtShield AI")
    st.markdown('<div class="info-card"><b>DebtShield AI</b> integrates housing, debt, labor, food access, and energy burden indicators into one financial risk platform.</div>', unsafe_allow_html=True)
    st.markdown("### Data honesty")
    real_df, _ = load_real_county_data()
    if real_df is not None:
        st.markdown(f'<div class="info-card">This version is running on <b>real Census ACS 5-Year data</b> for {len(real_df)} counties (income, rent burden, poverty, unemployment). Debt, energy, and food-access indicators still use category defaults until those sources (Eviction Lab, SCF, DOE LEAD, USDA) are integrated.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn">real_county_data.csv is not yet loaded. This instance is running on a single national benchmark row — county-level variation will not be accurate until real_county_data.csv is added to the repository.</div>', unsafe_allow_html=True)
    st.markdown("### Model vs. index")
    st.markdown('<div class="info-card">The Financial Distress Index is a transparent, rule-based score. The Random Forest prediction (Model Performance page) is a separately trained model. They are related but not identical, and are always labeled separately in this app.</div>', unsafe_allow_html=True)
    st.markdown("### Disclaimer")
    st.markdown('<div class="info-card">DebtShield AI provides educational and analytical information. It does not provide legal, financial, or government-benefit advice.</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">DebtShield AI V2 · Real-data pipeline, county selectors, comparison, model connection, scenario simulator, map · Research prototype.</div>', unsafe_allow_html=True)
