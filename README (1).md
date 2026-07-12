# DebtShield AI

A financial-distress risk platform for U.S. counties. Combines housing, poverty,
employment, and income indicators into a Financial Distress Index and Low/Medium/High
risk classification.

## Live application

[Add your Streamlit Cloud URL here]

## What's real vs. what's a placeholder (read this first)

This matters more than any feature list. Be exact with anyone reviewing this project:

**Real, live-pulled data (Census ACS 5-Year, 2023):**
- Median household income
- Median gross rent
- Rent burden (% of income)
- Poverty rate
- Unemployment rate (ACS employment-status proxy, not true BLS LAUS)

**Not yet real — flat national defaults, identical across every county:**
- Debt stress (Federal Reserve SCF)
- Energy burden (DOE LEAD)
- Food access (USDA Food Access Research Atlas)

These three categories are excluded from the Financial Distress Index's weighting
until real data is added — the index only weights categories it has real numbers
for, so a missing source doesn't quietly drag every county toward "Medium."

**Why the other three aren't done yet, specifically:**
- **SCF (household debt)** is a national household survey. It does not produce
  county-level estimates in its public microdata. There is no version of "pull SCF
  by county" that exists — this input needs to be approximated some other way
  (e.g. modeled from income + regional debt indices) or dropped from the design.
- **DOE LEAD** and **USDA Food Access Research Atlas** publish real county-level
  data, but only as large Excel/zip bulk downloads, not an API. Getting these in
  requires manually downloading the file from ers.usda.gov / the DOE LEAD tool and
  running it through a parser (not yet written).
- **Eviction Lab** publishes bulk CSVs by county-year; same situation as above.

## Data pipeline

`pull_real_county_data.py` calls the free Census ACS 5-Year API for every county
in the U.S. (~3,100 counties, ~51 requests) and writes `real_county_data.csv`.
Requires a free key from https://api.census.gov/data/key_signup.html.

```
pip3 install requests pandas
python3 pull_real_county_data.py
```

The app auto-loads `real_county_data.csv` if present in the same folder as
`debtshield_streamlit_app.py`. If it's missing, the app falls back to a single
national benchmark row and displays a visible warning — it will not silently
show fake variation.

## The machine learning model — known limitation

`phase2_best_random_forest_model.pkl` is connected to the app (Model Performance
page) and produces a live prediction, shown separately from the rule-based
Financial Distress Index.

**This is not an independently validated predictive model.** It was trained on
labels derived from the same or closely related indicators used to build the
index it's being compared against — meaning high accuracy here mostly reflects
that the model learned the labeling formula, not an independent real-world
outcome. Fixing this requires either:
1. Multi-year panel data (train on year T inputs → predict year T+1 outcome), or
2. A genuinely external outcome (e.g. future eviction filings, foreclosure rate)
   that isn't part of the index formula itself.

Neither exists in this project yet. Treat the RF prediction as a technical
demonstration of the modeling pipeline, not a validated forecast.

## App pages

- Executive Dashboard — dataset-wide overview
- County Profile — single county drill-down, with search / favorites / recently viewed
- Compare Counties — side-by-side comparison of 2-4 counties
- Risk Drivers — ranked stress categories for the selected county
- Scenario Simulator — slider-based what-if on income, poverty, unemployment, rent burden
- Risk Map — county-level map (currently covers counties with known coordinates)
- Model Performance — model comparison metrics, feature importance, live RF prediction
- Recommendations + Ask DebtShield chatbot — rule-based, not a connected LLM
- About — data-source honesty, disclaimers

## Chatbot — known limitation

The "Ask DebtShield" assistant is rule-based keyword matching, not a language
model. It can explain the selected county, compare two named counties, and answer
category questions (housing/debt/cost/energy/food), but it has no real conversational
memory and can't handle open-ended follow-up questions the way an LLM would.
Upgrading this to a real LLM backend requires wiring in an API (e.g. Anthropic's
Claude API) with your own API key — not yet implemented.

## Repository files

| File | Purpose |
|---|---|
| `debtshield_streamlit_app.py` | Main app |
| `pull_real_county_data.py` | Census data pull script |
| `real_county_data.csv` | Output of the pull script — real county data |
| `phase2_best_random_forest_model.pkl` | Trained RF model (see limitation above) |
| `phase2_model_comparison_results.csv` | Model comparison metrics |
| `phase2_feature_importance.csv` | Feature importance from training |
| `requirements.txt` | Python dependencies |

## Disclaimer

DebtShield AI provides educational and analytical information. It does not
provide legal, financial, or government-benefit advice.

## Roadmap (honest, not aspirational)

- [ ] Parse DOE LEAD + USDA Food Access bulk downloads into real per-county data
- [ ] Find or construct a defensible regional proxy for household debt (SCF has no county granularity)
- [ ] Acquire multi-year panel data to enable genuine predictive (not circular) modeling
- [ ] Wire chatbot to a real LLM API for true conversational ability
- [ ] Full county-level choropleth map (current map only plots counties with hardcoded coordinates)
- [ ] Browser-based click-through QA of every page (only server-boot tested so far)

## Author

Shaurya Thakor
