"""
Replication of Kline & Moretti (2014, QJE), "Local Economic Development,
Agglomeration Economies, and the Big Push: 100 Years of Evidence from the
Tennessee Valley Authority" -- Tables I-IV and Figure III.

Python port of evaluation.do and evaluation_aut.do from the authors'
replication archive (https://eml.berkeley.edu/~pkline/papers/tva_replication.zip).

Usage:
    python replicate.py --data path/to/build.dta

Outputs (in ./output):
    table1.csv ... table4.csv      our estimates next to the published ones
    table1.tex ... table4.tex      LaTeX tables (booktabs) for Overleaf
    figure3_weights.png / .pdf     Oaxaca-Blinder weights on control counties
    check_summary.txt              how many cells match the paper
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "src"))

from build import X_VARS, build_variables          # noqa: E402
from estimators import diff_means, oaxaca          # noqa: E402
import paper_values as PV                          # noqa: E402

OUT = Path(__file__).resolve().parent / "output"

LABELS = {
    "lnpop": "Population", "lnemp": "Total Employment", "lnhouse": "Housing Units",
    "lnwage": "Average Manufacturing Wage", "manufshr": "Manufacturing Share",
    "agrshr": "Agricultural Share", "lnfaval": "Average Agricultural Land Value",
    "lnagr": "Agricultural Employment", "lnmanuf": "Manufacturing Employment",
    "lnvfprod": "Value of Farm Production",
    "lnmedfaminc": "Median Family Income (1950--2000)",
    "lnmedhsval": "Median Housing Value",
}
T2_VARS = ["lnpop", "lnemp", "lnhouse", "lnwage", "manufshr", "agrshr", "lnfaval"]
T3_VARS = ["lnpop", "lnwage", "lnagr", "lnmanuf", "lnvfprod", "lnmedfaminc",
           "lnfaval", "lnmedhsval"]


# =============================================================================
# Samples
# =============================================================================

def stata_p25(x: pd.Series) -> float:
    """r(p25) from `summarize, detail`."""
    x = np.sort(x.dropna().to_numpy())
    k = len(x) * 0.25
    if float(k).is_integer():
        return (x[int(k) - 1] + x[int(k)]) / 2
    return x[int(np.ceil(k)) - 1]


def make_samples(d: pd.DataFrame, trim_q: float = 0.25) -> dict:
    """Propensity score, trimming and the three estimation samples.

    trim_q: share of control counties with the lowest propensity score to drop
    (0.25 in the paper; other values only for the robustness check)."""
    d = d.dropna(subset=X_VARS + ["tva"]).copy()

    # Stata: "nowage20dum != 0 predicts failure perfectly -- dropped and 83 obs
    # not used". Those counties leave e(sample); the variable is omitted.
    d = d[d["nowage20dum"] == 0].copy()
    xs = [v for v in X_VARS if v != "nowage20dum"]
    logit = sm.Logit(d["tva"], sm.add_constant(d[xs])).fit(method="newton", disp=0, maxiter=500)
    d["phat"] = logit.predict().astype(np.float32)        # `predict` stores a float

    # Trim controls with propensity score in the bottom quartile. Stata drops 582
    # controls: the one sitting exactly at p25 goes too (macro rounding), so <=.
    ctrl_phat = d.loc[d["tva"] == 0, "phat"]
    if trim_q == 0.25:
        cut = stata_p25(ctrl_phat)
    elif trim_q == 0:
        cut = -np.inf
    else:
        cut = np.quantile(ctrl_phat, trim_q, method="inverted_cdf")
    trim = d[~((d["phat"] <= cut) & (d["tva"] == 0))].copy()

    # evaluation_aut.do: no trimming, controls are the proposed authorities
    aut = d[(d["tva"] == 1) | (d["aut4"] == 1)].copy()

    return {"full": d, "trim": trim, "aut": aut, "cut": cut,
            "converged": logit.mle_retvals["converged"]}


def winsorize(s: pd.Series) -> pd.Series:
    """`centile Dy, c(1 99)` ((n+1)p rule) and clip -- over the whole sample in memory."""
    lo, hi = np.percentile(s.dropna(), [1, 99], method="weibull")
    return s.clip(lower=lo, upper=hi)


def change(df, var, start, end, decades):
    return winsorize((df[f"{var}{end}"] - df[f"{var}{start}"]) / decades)


def arrays(df):
    return dict(X=df[X_VARS].to_numpy(float), t=df["tva"].to_numpy(float),
                groups=df["state"].to_numpy(), lat=df["latitude"].to_numpy(float),
                lon=df["longitud"].to_numpy(float))


# =============================================================================
# Panels
# =============================================================================

def panel_A(dy, s):
    """TVA vs rest of US: unadjusted (cluster), Oaxaca (cluster and HAC)."""
    a = arrays(s)
    b0, se0, _ = diff_means(dy, a["t"], "cluster", groups=a["groups"])
    b1, se1, n = oaxaca(dy, a["X"], a["t"], "cluster", groups=a["groups"])
    _, se2, _ = oaxaca(dy, a["X"], a["t"], "conley", lat=a["lat"], lon=a["lon"])
    return dict(b_unadj=b0, se_unadj=se0, b_adj=b1, se_cl=se1, se_hac=se2, N=n)


def panel_B(dy, s):
    """TVA vs US South: same trimmed sample restricted to the South, HAC only."""
    m = (s["south"] == 1).to_numpy()
    a = arrays(s[m])
    y = dy.to_numpy()[m]
    b0, se0, _ = diff_means(y, a["t"], "conley", lat=a["lat"], lon=a["lon"])
    b1, se1, n = oaxaca(y, a["X"], a["t"], "conley", lat=a["lat"], lon=a["lon"])
    return dict(b_unadj=b0, se_unadj=se0, b_adj=b1, se_hac=se1, N=n)


def panel_C(dy, s):
    """TVA vs proposed authorities, HAC only (unadjusted = Oaxaca on T alone)."""
    a = arrays(s)
    y = dy.to_numpy()
    b0, se0, _ = oaxaca(y, a["t"][:, None], a["t"], "conley", lat=a["lat"], lon=a["lon"])
    b1, se1, n = oaxaca(y, a["X"], a["t"], "conley", lat=a["lat"], lon=a["lon"])
    return dict(b_unadj=b0, se_unadj=se0, b_adj=b1, se_hac=se1, N=n)


# =============================================================================
# Tables
# =============================================================================

def table1(S):
    rows = list(PV.TABLE1)[:-2]
    groups = [
        S["full"][S["full"]["tva"] == 1],
        S["full"][S["full"]["tva"] == 0],
        S["full"][(S["full"]["tva"] == 0) & (S["full"]["south"] == 1)],
        S["aut"][S["aut"]["tva"] == 0],
        S["trim"][S["trim"]["tva"] == 0],
        S["trim"][(S["trim"]["tva"] == 0) & (S["trim"]["south"] == 1)],
    ]
    out = pd.DataFrame({f"({j+1})": [g[r].mean() for r in rows] + [len(g), g["state"].nunique()]
                        for j, g in enumerate(groups)}, index=rows + ["N", "states"])
    return out


def table_long_run(S, var_list, periods, paper):
    """Tables II and III: one row per outcome and panel."""
    recs = []
    for var in var_list:
        start, end, dec = periods(var)
        dyA = change(S["trim"], var, start, end, dec)
        dyC = change(S["aut"], var, start, end, dec)
        for panel, res in (("A", panel_A(dyA, S["trim"])),
                           ("B", panel_B(dyA, S["trim"])),
                           ("C", panel_C(dyC, S["aut"]))):
            p = paper[panel][var]
            keys = ["b_unadj", "se_unadj", "b_adj", "se_cl", "se_hac", "N"] if panel == "A" \
                else ["b_unadj", "se_unadj", "b_adj", "se_hac", "N"]
            rec = {"panel": panel, "outcome": var}
            for k, pv in zip(keys, p):
                rec[k] = res[k]
                rec[f"paper_{k}"] = pv
            recs.append(rec)
    return pd.DataFrame(recs)


def table4(S):
    recs = []
    for var in T3_VARS:
        rec = {"outcome": var}
        for label, (start, end, dec) in (("4060", ("40", "60", 2)), ("6000", ("60", "2000", 4))):
            if var == "lnmedfaminc" and label == "4060":
                continue
            dyA = change(S["trim"], var, start, end, dec)
            dyC = change(S["aut"], var, start, end, dec)
            a = arrays(S["trim"])
            b, se, _ = oaxaca(dyA, a["X"], a["t"], "cluster", groups=a["groups"])
            rec[f"US_{label}"], rec[f"US_{label}_se"] = b, se
            pB = panel_B(dyA, S["trim"])
            rec[f"South_{label}"], rec[f"South_{label}_se"] = pB["b_adj"], pB["se_hac"]
            pC = panel_C(dyC, S["aut"])
            rec[f"Aut_{label}"], rec[f"Aut_{label}_se"] = pC["b_adj"], pC["se_hac"]
        cols = ["US_4060", "US_6000", "South_4060", "South_6000", "Aut_4060", "Aut_6000"]
        for c, pv in zip(cols, PV.TABLE4[var]):
            rec[f"paper_{c}"] = pv
        recs.append(rec)
    return pd.DataFrame(recs)


# =============================================================================
# Checks and LaTeX
# =============================================================================

def compare(ours, paper, decimals=3):
    """Count cells equal to the paper after rounding; return mismatches."""
    ok, bad = 0, []
    for name, o, p in zip(ours.index if hasattr(ours, "index") else range(len(ours)), ours, paper):
        if p is None or pd.isna(o):
            continue
        if round(float(o), decimals) == round(float(p), decimals) or \
                (decimals == 0 and int(o) == int(p)):
            ok += 1
        else:
            bad.append((name, float(o), p))
    return ok, bad


def stars(b, se, df=None):
    if se is None or pd.isna(se) or se == 0:
        return ""
    p = 2 * (1 - (stats.t.cdf(abs(b / se), df) if df else stats.norm.cdf(abs(b / se))))
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def fmt(b, se=None, df=None):
    return f"{b:.3f}{stars(b, se, df)}" if pd.notna(b) else ""


def se_fmt(x):
    return f"({x:.3f})" if pd.notna(x) else ""


def dag(ours, paper):
    """Dagger when the replicated value differs from the paper after rounding."""
    if paper is None or pd.isna(paper) or pd.isna(ours):
        return ""
    return "" if round(float(ours), 3) == round(float(paper), 3) else r"$^\dagger$"


def tex_long_run(df, caption, label):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & Unadjusted & S.E. & Controls & Clustered S.E. & Spatial HAC & N \\",
             r" & (1) & (2) & (3) & (4) & (5) & (6) \\", r"\midrule"]
    titles = {"A": "Panel A: TVA vs. rest of US", "B": "Panel B: TVA vs. US South",
              "C": "Panel C: TVA vs. proposed authorities"}
    G = 47  # number of states in Panel A sample (df for clustered stars = G-1)
    for panel in "ABC":
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{titles[panel]}}}}} \\")
        for _, r in df[df["panel"] == panel].iterrows():
            if panel == "A":
                cells = [fmt(r.b_unadj, r.se_unadj, G - 1), se_fmt(r.se_unadj),
                         fmt(r.b_adj, r.se_cl, G - 1), se_fmt(r.se_cl), se_fmt(r.se_hac)]
            else:
                cells = [fmt(r.b_unadj, r.se_unadj) + dag(r.b_unadj, r.paper_b_unadj), se_fmt(r.se_unadj),
                         fmt(r.b_adj, r.se_hac) + dag(r.b_adj, r.paper_b_adj), "", se_fmt(r.se_hac)]
            lines.append(f"{LABELS[r.outcome]} & " + " & ".join(cells) + f" & {int(r.N)} \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{minipage}{0.95\textwidth}\vspace{4pt}\scriptsize "
              r"Notes: Replication in Python of Kline and Moretti (2014). Decadalized changes. "
              r"Column (3) is the Oaxaca-Blinder estimate of Kline (2011). Clustered S.E. by state; "
              r"spatial HAC as in Conley (1999) with 200-mile bandwidth. Stars: clustered S.E. in "
              r"Panel A, spatial HAC in Panels B and C. Outcomes winsorized at the 1st and 99th percentiles, "
              r"as in the authors' code (not documented in the paper). $^\dagger$ differs from the published "
              r"value after rounding (see text). * 10\%, ** 5\%, *** 1\%."
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines)


def tex_table4(df):
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Decadalized impact of TVA over two sub-periods (replication of Table IV)}",
             r"\label{tab:km_t4}", r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Entire US} & \multicolumn{2}{c}{South} & "
             r"\multicolumn{2}{c}{Proposed authorities} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & 1940--60 & 1960--2000 & 1940--60 & 1960--2000 & 1940--60 & 1960--2000 \\",
             r"\midrule"]
    for _, r in df.iterrows():
        cells, ses = [], []
        for grp, dfree in (("US", 46), ("South", None), ("Aut", None)):
            for per in ("4060", "6000"):
                b, se = r.get(f"{grp}_{per}"), r.get(f"{grp}_{per}_se")
                cells.append(fmt(b, se, dfree) + dag(b, r.get(f"paper_{grp}_{per}")) if pd.notna(b) else "N/A")
                ses.append(se_fmt(se) if pd.notna(se) else "")
        lines.append(f"{LABELS[r.outcome]} & " + " & ".join(cells) + r" \\")
        lines.append(" & " + " & ".join(ses) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{minipage}{0.95\textwidth}\vspace{4pt}\scriptsize "
              r"Notes: Oaxaca-Blinder estimates with full controls. S.E. in parentheses: clustered by "
              r"state (entire US) or spatial HAC with 200-mile bandwidth (South, proposed authorities). "
              r"The paper reports point estimates only. $^\dagger$ differs from the published value after "
              r"rounding. * 10\%, ** 5\%, *** 1\%."
              r"\end{minipage}", r"\end{table}"]
    return "\n".join(lines)


def tex_table1(t1):
    labels = {
        "lnpop30": "Log Population", "lnemp30": "Log Employment", "lnhouse30": "Log \\# of Houses",
        "lnwageobs30": "Log Average Manufacturing Wage", "manufshr30": "Manufacturing Employment Share",
        "agrshr30": "Agricultural Employment Share", "white30": "\\% White",
        "urbshare30": "\\% Urbanized", "pctil30": "\\% Illiterate",
        "fbshr30": "\\% of Whites Foreign Born", "lnfaval30": "Log Average Farm Value",
        "lnmedhsval30": "Log Median Housing Value", "lnmedrnt30": "Log Median Contract Rent",
        "PRADIO": "\\% Own Radio", "elevmax": "Max Elevation (meters)",
        "elevrang": "Elevation Range (Max-Min)", "south": "\\% Counties in South",
    }
    lines = [r"\begin{table}[htbp]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Summary statistics (replication of Table I)}", r"\label{tab:km_t1}",
             r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Overall} & Proposed & \multicolumn{2}{c}{Trimmed sample} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
             r" & TVA & Non-TVA & Non-TVA South & authorities & Non-TVA & Non-TVA South \\",
             r" & (1) & (2) & (3) & (4) & (5) & (6) \\", r"\midrule",
             r"\multicolumn{7}{l}{\textit{1930 characteristics}} \\"]
    for r in t1.index:
        if r == "D_lnpop":
            lines += [r"\addlinespace", r"\multicolumn{7}{l}{\textit{Changes 1920--1930}} \\"]
        if r in ("N", "states"):
            continue
        name = labels.get(r, labels.get(r.replace("D_", "") + "30", r))
        lines.append(f"{name} & " + " & ".join(f"{v:.3f}" for v in t1.loc[r]) + r" \\")
    lines += [r"\addlinespace",
              r"\# of Observations & " + " & ".join(str(int(v)) for v in t1.loc["N"]) + r" \\",
              r"\# of States & " + " & ".join(str(int(v)) for v in t1.loc["states"]) + r" \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# =============================================================================
# Figure III
# =============================================================================

COUNTIES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def oaxaca_weights(s: pd.DataFrame) -> pd.Series:
    """Implicit weight on each control county (evaluation.do, Mata block):
    w_j = xbar_T' (X_C'X_C)^-1 x_j, with X including a constant."""
    from estimators import _independent_columns
    X = np.column_stack([np.ones(len(s)), s[X_VARS].to_numpy(float)])
    keep = _independent_columns(X)
    X = X[:, keep]
    t = s["tva"].to_numpy() == 1
    xbar = X[t].mean(axis=0)
    w = X[~t] @ np.linalg.solve(X[~t].T @ X[~t], xbar)
    return pd.Series(w, index=s.index[~t])


def figure3(S, counties_path: Path):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not counties_path.exists():
        import urllib.request
        counties_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(COUNTIES_URL, counties_path)

    s = S["trim"]
    w = oaxaca_weights(s)
    geo = gpd.read_file(counties_path)
    geo["fips"] = geo["id"].astype(int)

    ctrl = s.loc[w.index, ["fips"]].assign(w=w.values)
    ctrl["fips"] = ctrl["fips"].astype(int)
    tva = s.loc[s["tva"] == 1, ["fips"]].astype(int)

    g_ctrl = geo.merge(ctrl, on="fips")
    g_tva = geo.merge(tva, on="fips")

    # spmap default: 4 quantile classes
    breaks = np.quantile(w, [0, 0.25, 0.5, 0.75, 1])
    g_ctrl["cls"] = np.clip(np.searchsorted(breaks[1:-1], g_ctrl["w"], side="left"), 0, 3)
    from style import BLUE, EMPTY, INK, MUTED, TAUPE, CREAM, save_figure, titles
    shades = ["#ffffff", CREAM, TAUPE, "#5e5040"]          # thesis sequential palette (SEQ)

    fig, ax = plt.subplots(figsize=(11, 6.6))
    for k in range(4):
        g_ctrl[g_ctrl["cls"] == k].plot(ax=ax, color=shades[k], edgecolor=MUTED, linewidth=0.12)
    g_tva.plot(ax=ax, color=BLUE, edgecolor="white", linewidth=0.2)
    labels = [f"[{breaks[0]:.4f}, {breaks[1]:.4f}]"] + \
             [f"({breaks[k]:.4f}, {breaks[k+1]:.4f}]" for k in range(1, 4)]
    handles = [Patch(facecolor=shades[k], edgecolor=MUTED, label=labels[k]) for k in (3, 2, 1, 0)]
    handles.append(Patch(facecolor=BLUE, edgecolor="white", label="TVA counties"))
    ax.legend(handles=handles, title="Oaxaca weight (quartiles)", loc="lower left", fontsize=8,
              title_fontsize=8.5, frameon=False, alignment="left")
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_axis_off()
    titles(fig, "Weight on untreated counties (replication of Figure III)",
           "implicit Oaxaca-Blinder weight of each control county in the trimmed sample",
           "Weights can be negative; they sum to one. County boundaries by present-day FIPS code "
           "(plotly GeoJSON). Data: Kline and Moretti (2014) replication archive.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    save_figure(fig, OUT / "figure3_weights.png")

    unmatched = len(ctrl) + len(tva) - len(g_ctrl) - len(g_tva)
    return breaks, w.sum(), unmatched


# =============================================================================
# Main
# =============================================================================

def check_data(path):
    """Stop with a clear message if build.dta is not where we expect it."""
    if not Path(path).exists():
        sys.exit(
            f"\nbuild.dta not found at:\n  {path}\n\n"
            "Unzip tva_replication.zip (https://eml.berkeley.edu/~pkline/papers/tva_replication.zip)\n"
            "and copy build.dta into the 'data' folder next to replicate.py,\n"
            "or pass its location with:  python replicate.py --data <path to build.dta>\n")


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--data", default=str(here / "data" / "build.dta"),
                    help="build.dta from the replication archive")
    ap.add_argument("--counties", default=str(here / "data" / "geojson-counties-fips.json"),
                    help="county boundaries (downloaded automatically if missing)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    check_data(args.data)
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    raw = pd.read_stata(args.data, convert_categoricals=False)
    S = make_samples(build_variables(raw))
    log = [f"Logit converged: {S['converged']}",
           f"Samples: full={len(S['full'])} (Stata 2489), trimmed={len(S['trim'])} (Stata 1907), "
           f"authorities={len(S['aut'])} (Stata 991)"]

    # ---- Table I
    t1 = table1(S)
    paper1 = pd.DataFrame(PV.TABLE1, index=t1.columns).T
    t1.join(paper1.add_prefix("paper_")).to_csv(OUT / "table1.csv")
    (OUT / "table1.tex").write_text(tex_table1(t1))
    ok = tot = 0
    bad1 = []
    for r in t1.index:
        for c in t1.columns:
            tot += 1
            o, p = t1.loc[r, c], paper1.loc[r, c]
            if round(o, 3) == round(p, 3):
                ok += 1
            else:
                bad1.append((r, c, round(o, 3), p))
    log.append(f"Table I: {ok}/{tot} cells match the paper")
    log += [f"   {r} {c}: ours {o} vs paper {p}" for r, c, o, p in bad1]

    # ---- Tables II and III
    t2 = table_long_run(S, T2_VARS, lambda v: ("0", "40", 4), PV.TABLE2)
    t3 = table_long_run(S, T3_VARS,
                        lambda v: ("50", "2000", 5) if v == "lnmedfaminc" else ("40", "2000", 6),
                        PV.TABLE3)
    for name, t in (("II", t2), ("III", t3)):
        ok = tot = 0
        bad = []
        for _, r in t.iterrows():
            for k in ["b_unadj", "se_unadj", "b_adj", "se_cl", "se_hac", "N"]:
                if f"paper_{k}" not in r or pd.isna(r.get(f"paper_{k}")):
                    continue
                tot += 1
                o, p = r[k], r[f"paper_{k}"]
                same = int(o) == int(p) if k == "N" else round(o, 3) == round(p, 3)
                ok += same
                if not same:
                    bad.append(f"   Panel {r.panel} {r.outcome} {k}: ours {o:.4f} vs paper {p}")
        log.append(f"Table {name}: {ok}/{tot} cells match the paper")
        log += bad
    t2.to_csv(OUT / "table2.csv", index=False)
    t3.to_csv(OUT / "table3.csv", index=False)
    (OUT / "table2.tex").write_text(tex_long_run(
        t2, "Decadalized growth rates in TVA region vs. counterfactual regions, 1900--1940 "
            "(replication of Table II)", "tab:km_t2"))
    (OUT / "table3.tex").write_text(tex_long_run(
        t3, "Decadalized impact of TVA on growth rate of outcomes, 1940--2000 "
            "(replication of Table III)", "tab:km_t3"))

    # ---- Table IV
    t4 = table4(S)
    t4.to_csv(OUT / "table4.csv", index=False)
    (OUT / "table4.tex").write_text(tex_table4(t4))
    ok = tot = 0
    bad = []
    for _, r in t4.iterrows():
        for c in ["US_4060", "US_6000", "South_4060", "South_6000", "Aut_4060", "Aut_6000"]:
            p = r[f"paper_{c}"]
            if p is None or pd.isna(p):
                continue
            tot += 1
            same = round(r[c], 3) == round(p, 3) or (round(r[c], 3) == 0 and p == 0)
            ok += same
            if not same:
                bad.append(f"   {r.outcome} {c}: ours {r[c]:.4f} vs paper {p}")
    log.append(f"Table IV: {ok}/{tot} cells match the paper")
    log += bad

    # ---- Figure III
    try:
        import geopandas  # noqa: F401
    except ImportError:
        args.no_figure = True
        log.append("Figure III skipped: geopandas is not installed (pip install geopandas).")
    if not args.no_figure:
        breaks, wsum, unmatched = figure3(S, Path(args.counties))
        log.append("Figure III: quantile breaks " + ", ".join(f"{b:.7f}" for b in breaks)
                   + f" | weights sum to {wsum:.4f} | counties without a boundary: {unmatched}")
        log.append("   paper legend: -.0074167, -.000209, .000589, .001268, .0045465")

    summary = "\n".join(log)
    (OUT / "check_summary.txt").write_text(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
