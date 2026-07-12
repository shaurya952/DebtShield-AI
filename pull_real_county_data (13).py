"""
Pulls REAL data from the U.S. Census Bureau's ACS 5-Year API for EVERY
county in the United States (~3,144 counties, all 50 states + DC) and
saves it as real_county_data.csv in the format DebtShield AI expects.

Run this locally:
    pip3 install requests pandas
    python3 pull_real_county_data.py

Uses one API call per state (~51 calls total) instead of one per county,
via the Census API's county:* wildcard.

WHAT THIS PULLS (real, live Census data, for all ~3,144 counties):
  - acs_median_household_income   (table B19013)
  - acs_median_gross_rent          (table B25064)
  - acs_rent_burden_pct            (table B25071 - median rent as % of income)
  - acs_poverty_rate               (table B17001, computed)
  - bls_unemployment_rate          (table B23025, computed - ACS employment status,
                                     used as a proxy for true BLS LAUS unemployment)

WHAT THIS DOES NOT PULL (per your Source_Tracker, still "Not started"):
  Zillow, Eviction Lab, SCF debt data, MIT Living Wage, CDC SVI, FHFA HPI,
  USDA Food Access, DOE LEAD energy burden, NY Fed delinquency rates.
  See the project README for why these can't be pulled the same way.
"""

import requests
import pandas as pd
import time

API_KEY = "f23badc68f18d53d425cba2177c56deb8954c6f7"

STATE_FIPS = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California",
    "08": "Colorado", "09": "Connecticut", "10": "Delaware", "11": "District of Columbia",
    "12": "Florida", "13": "Georgia", "15": "Hawaii", "16": "Idaho", "17": "Illinois",
    "18": "Indiana", "19": "Iowa", "20": "Kansas", "21": "Kentucky", "22": "Louisiana",
    "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
    "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York",
    "37": "North Carolina", "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon",
    "42": "Pennsylvania", "44": "Rhode Island", "45": "South Carolina", "46": "South Dakota",
    "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont", "51": "Virginia",
    "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}

VARS = [
    "NAME",
    "B19013_001E",  # median household income
    "B25064_001E",  # median gross rent
    "B25071_001E",  # median gross rent as % of household income
    "B17001_002E",  # below poverty count
    "B17001_001E",  # poverty universe
    "B23025_005E",  # unemployed
    "B23025_002E",  # labor force
]

BASE_URL = "https://api.census.gov/data/2023/acs/acs5"
HEADERS = {"User-Agent": "Mozilla/5.0 (DebtShield data pull script)"}

rows = []
for state_fips, state_name in STATE_FIPS.items():
    params = {
        "get": ",".join(VARS),
        "for": "county:*",
        "in": f"state:{state_fips}",
        "key": API_KEY,
    }
    resp = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=60)
    try:
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        print(f"\nFAILED on {state_name}")
        print(f"Status code: {resp.status_code}")
        print(f"Response text: {resp.text[:500]}")
        continue

    header, records = data[0], data[1:]
    for rec_values in records:
        rec = dict(zip(header, rec_values))
        income = float(rec["B19013_001E"]) if rec["B19013_001E"] not in (None, "-666666666") else None
        rent = float(rec["B25064_001E"]) if rec["B25064_001E"] not in (None, "-666666666") else None
        rent_burden = float(rec["B25071_001E"]) if rec["B25071_001E"] not in (None, "-666666666") else None
        try:
            poverty_n = float(rec["B17001_002E"])
            poverty_universe = float(rec["B17001_001E"])
            poverty_rate = round(poverty_n / poverty_universe * 100, 2) if poverty_universe else None
        except (TypeError, ValueError):
            poverty_rate = None
        try:
            unemployed = float(rec["B23025_005E"])
            labor_force = float(rec["B23025_002E"])
            unemployment_rate = round(unemployed / labor_force * 100, 2) if labor_force else None
        except (TypeError, ValueError):
            unemployment_rate = None

        county_name = rec["NAME"].split(",")[0].strip()
        rows.append({
            "fips": state_fips + rec["county"],
            "state": state_name,
            "county": county_name,
            "year": 2023,
            "acs_median_household_income": income,
            "acs_median_gross_rent": rent,
            "acs_rent_burden_pct": rent_burden,
            "acs_poverty_rate": poverty_rate,
            "bls_unemployment_rate": unemployment_rate,
        })
    print(f"Pulled {len(records)} counties in {state_name}")
    time.sleep(0.3)

df = pd.DataFrame(rows)
df = df.dropna(subset=["acs_median_household_income", "acs_poverty_rate"])
df.to_csv("real_county_data.csv", index=False)
print(f"\nSaved real_county_data.csv with {len(df)} counties across {df['state'].nunique()} states.")
