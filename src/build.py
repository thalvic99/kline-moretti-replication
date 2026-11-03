"""
Variable construction.

Python port of lines 1-235 of evaluation.do (Kline & Moretti replication
archive). Quirks of the original code are kept on purpose and flagged with
"kept as in original" so results match the published tables.
"""

import numpy as np
import pandas as pd

YEARS = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "2000"]

# Covariates ($X in the do-file). Stata accepts unambiguous abbreviations,
# so the original list writes nowage20 / nowage30 / notwage30 / PRADIO; the
# actual variables are the ones below.
X_VARS = [
    "lnelevmax", "lnelevrang", "lnarea",
    "lnpop20", "lnpop20sq", "lnpop30", "lnpop30sq", "popdifsq",
    "agrshr20", "agrshr20sq", "agrshr30", "agrshr30sq",
    "manufshr20", "manufshr30",
    "nowage20dum", "nowage30dum", "lnwage20", "lnwage30",
    "notwage30dum", "lntwage30",
    "lnemp20", "lnemp30", "urbshare20", "urbshare30",
    "lnfaval20", "lnfaval30", "lnmedhsval30", "lnmedrnt30",
    "white20", "white20sq", "white30", "white30sq",
    "pctil20", "pctil30", "PRADIO", "urate30", "fbshr20", "fbshr30",
]


def stata_ln(s: pd.Series) -> pd.Series:
    """Stata ln(): missing for x <= 0."""
    return np.log(s.where(s > 0))


def build_variables(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()

    def cpi(yr):
        return d[f"cpi{yr}"] / 100

    # --- real wages: manufacturing and trade -------------------------------
    wage_src = {"0": "00", "20": "20", "30": "30", "40": "39", "50": "47",
                "60": "58", "70": "67", "80": "82", "90": "87", "2000": "97"}
    for yr, src in wage_src.items():
        d[f"wage{yr}"] = d[f"pcmwage{src}"] / cpi(yr)

    twage_src = {"30": "30", "40": "40", "50": "54", "60": "63", "70": "72",
                 "80": "82", "90": "87", "2000": "97"}
    for yr, src in twage_src.items():
        d[f"twage{yr}"] = d[f"pctwage{src}"] / cpi(yr)

    # --- agricultural land values ------------------------------------------
    faval_src = {"0": "900", "10": "910", "20": "920", "30": "930", "40": "940",
                 "50": "950", "60": "959", "70": "1969", "80": "1982",
                 "90": "1992", "2000": "2002"}
    for yr, src in faval_src.items():
        d[f"lnfaval{yr}"] = stata_ln(d[f"faval{src}"] / cpi(yr))

    # --- median family income (1990 uses the 1980 level: kept as in original)
    for yr in ["50", "60", "70", "80", "2000"]:
        d[f"lnmedfaminc{yr}"] = stata_ln(d[f"medfaminc{yr}"] / cpi(yr))
    d["lnmedfaminc90"] = stata_ln(d["medfaminc80"] / cpi("90"))

    # --- value of farm production ------------------------------------------
    for yr in ["30", "40", "50", "60", "70", "80", "90", "2000"]:
        d[f"lnvfprod{yr}"] = stata_ln(d[f"vfprod{yr}"] / cpi(yr))

    # --- foreign born -------------------------------------------------------
    d["fb0"] = d["fbwmtot00"] + d["fbwftot00"]
    d["fb20"] = d["fbwmtot20"] + d["fbwftot20"]
    d["fb30"] = d["fbwmtot"] + d["fbwftot"]
    d["fbshr20"] = d["fb20"] / (d["wmtot20"] + d["wftot20"])
    d["fbshr30"] = d["fb30"] / (d["wmtot"] + d["wftot"])

    # --- housing values and rents, deflated --------------------------------
    d = d.rename(columns={"medrnt30_NHGIS": "medrnt30",
                          "medhsval30_NHGIS": "medhsval30",
                          "var88_county72": "medhsval70",
                          "var89_county72": "medrnt70"})
    for var in ["medhsval", "medrnt"]:
        for yr in ["30", "40", "50", "60", "70", "80", "90", "2000"]:
            if f"{var}{yr}" in d.columns:
                d[f"{var}{yr}"] = d[f"{var}{yr}"] / cpi(yr)

    # --- drop counties whose area changed a lot (missing -> dropped, as Stata)
    gap = (d["b1_lnd01_county00"] - d["area"]) / (d["b1_lnd01_county00"] + d["area"]) / 2
    d = d[gap.abs() <= 0.03].copy()
    d["area"] = (d["b1_lnd01_county00"] + d["area"]) / 2

    # --- standardize employment variable names -----------------------------
    d = d.rename(columns={"emp00": "emp0", "manuf_jobs_00": "manuf_jobs_0"})
    for yr in YEARS:
        d = d.drop(columns=[c for c in (f"manuf{yr}", f"agr{yr}") if c in d.columns])
        d = d.rename(columns={f"manuf_jobs_{yr}": f"manuf{yr}",
                              f"ag_jobs{yr}": f"agr{yr}"})

    # --- sector shares -----------------------------------------------------
    for var in ["manuf", "agr"]:
        for yr in YEARS:
            if f"{var}shr{yr}" not in d.columns and {f"{var}{yr}", f"emp{yr}"} <= set(d.columns):
                d[f"{var}shr{yr}"] = d[f"{var}{yr}"] / d[f"emp{yr}"]

    # --- log outcomes ------------------------------------------------------
    for var in ["pop", "emp", "house", "wage", "twage", "agr", "manuf",
                "medhsval", "medrnt", "fb"]:
        for yr in YEARS:
            if f"ln{var}{yr}" not in d.columns and f"{var}{yr}" in d.columns:
                d[f"ln{var}{yr}"] = stata_ln(d[f"{var}{yr}"])

    # --- urban shares (1930 divides by 1920 population: kept as in original)
    d["urbshare20"] = d["popurb20"] / d["pop20"]
    d["urbshare30"] = d["popurb30"] / d["pop20"]

    # --- zero employment in 1910-1930 -> log set to 0 ----------------------
    for var in ["agr", "manuf"]:
        for yr in ["10", "20", "30"]:
            d.loc[d[f"{var}{yr}"] <= 0, f"ln{var}{yr}"] = 0

    # --- missing 1920/1930 wages -> log set to -1 plus a dummy -------------
    for var in ["wage", "twage"]:
        for yr in ["20", "30"]:
            if f"{var}{yr}" in d.columns:
                miss = d[f"{var}{yr}"].isna()
                d.loc[miss, f"ln{var}{yr}"] = -1
                d[f"no{var}{yr}dum"] = miss.astype(float)

    # --- remaining controls ------------------------------------------------
    for var in ["elevmax", "elevrang", "area"]:
        d[f"ln{var}"] = stata_ln(d[var])
    for var in ["lnpop20", "lnpop30", "white20", "white30"]:
        d[f"{var}sq"] = d[var] ** 2
    d["popdifsq"] = (d["lnpop30"] - d["lnpop20"]) ** 2
    d["agrshr20sq"] = d["agrshr20"] ** 2
    d["agrshr30sq"] = d["agrshr30"] ** 2
    d["pctil20"] = d["tillit1020"] / d["t10tot20"]
    d["pctil30"] = d["tillit10"] / d["t10tot"]
    d["PRADIO"] = d["PRADIO30"] / 100
    d["urate30"] = d["totunemp"] / (d["totunemp"] + d["emp30"])

    # --- Table I: observed (not imputed) wages and 1920-30 changes ---------
    d["lnwageobs20"] = stata_ln(d["wage20"])
    d["lnwageobs30"] = stata_ln(d["wage30"])
    for var in ["lnpop", "lnemp", "lnhouse", "lnwageobs", "manufshr", "agrshr",
                "white", "urbshare", "pctil", "fbshr", "lnfaval"]:
        d[f"D_{var}"] = d[f"{var}30"] - d[f"{var}20"]

    return d
