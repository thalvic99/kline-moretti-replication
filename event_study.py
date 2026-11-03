"""
Extension: decade-by-decade event study of the TVA effect.

Instead of the two long differences in Tables III-IV, estimate the Oaxaca-Blinder
effect separately for each census year t, using the change y_t - y_1940
(cumulative log points, not decadalized). 1940 is the base year, as in Table III.

    t > 1940 : dynamic effect of TVA up to year t
    t < 1940 : pre-period placebo (should be ~0 if TVA counties were on the same
               path as the comparison group)

Caveat: the controls include 1920 and 1930 levels of population, employment,
sector shares and wages, so the 1920 and 1930 coefficients are partly pinned
down by construction. The informative pre-periods are 1900 and 1910. Also note
that TVA started in 1933, so every "pre" change measured up to 1940 includes
some treated years.

Same samples, controls and estimators as the replication:
  * TVA vs rest of US (trimmed sample), 95% CI from S.E. clustered by state;
  * TVA vs proposed authorities, 95% CI from spatial HAC (200 miles).

Usage:
    python event_study.py --data data/build.dta
Outputs: output/event_study.csv, output/event_study.png/.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from build import build_variables                     # noqa: E402
from estimators import oaxaca                         # noqa: E402
from replicate import OUT, arrays, check_data, make_samples, winsorize   # noqa: E402
from style import BG, CREAM, GREY, INK, MUTED, TAUPE, save_figure, style_axis, titles  # noqa: E402

YEARS = ["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "2000"]
BASE = "40"
OUTCOMES = {
    "lnmanuf": "Manufacturing employment",
    "lnagr": "Agricultural employment",
    "lnpop": "Population",
    "lnwage": "Average manufacturing wage",
}


def year_num(y: str) -> int:
    return 1900 if y == "0" else 2000 if y == "2000" else 1900 + int(y)


def estimate(S) -> pd.DataFrame:
    recs = []
    for var, label in OUTCOMES.items():
        for yr in YEARS:
            if yr == BASE or f"{var}{yr}" not in S["trim"].columns:
                continue
            for comp, s, se_type in (("Rest of US", S["trim"], "cluster"),
                                     ("Proposed authorities", S["aut"], "conley")):
                dy = winsorize(s[f"{var}{yr}"] - s[f"{var}{BASE}"])
                if dy.notna().sum() < 100:
                    continue
                a = arrays(s)
                b, se, n = oaxaca(dy, a["X"], a["t"], se_type, groups=a["groups"],
                                  lat=a["lat"], lon=a["lon"])
                recs.append(dict(outcome=var, label=label, comparison=comp,
                                 year=year_num(yr), effect=b, se=se, N=n))
    # base year: normalized to zero
    for var, label in OUTCOMES.items():
        for comp in ("Rest of US", "Proposed authorities"):
            recs.append(dict(outcome=var, label=label, comparison=comp, year=1940,
                             effect=0.0, se=0.0, N=np.nan))
    return pd.DataFrame(recs).sort_values(["outcome", "comparison", "year"])


def plot(es: pd.DataFrame):
    """Thesis style: black squares (rest of US) and taupe diamonds (proposed authorities);
    filled = 95% CI excludes 0."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
    styles = {"Rest of US": dict(color=INK, marker="s", offset=-0.9),
              "Proposed authorities": dict(color=TAUPE, marker="D", offset=0.9)}
    for ax, (var, label) in zip(axes.flat, OUTCOMES.items()):
        sub = es[es["outcome"] == var]
        ax.axvspan(1933, 1958, color=CREAM, alpha=0.55, lw=0, zorder=0)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.axvline(1940, color=MUTED, ls=":", lw=0.8)
        for comp, st in styles.items():
            g = sub[sub["comparison"] == comp].sort_values("year")
            x = g["year"] + st["offset"]
            lo, hi = g["effect"] - 1.96 * g["se"], g["effect"] + 1.96 * g["se"]
            ax.plot(x, g["effect"], color=st["color"], lw=0.8, alpha=0.6)
            ax.vlines(x, lo, hi, color=st["color"], lw=1.6)
            sig = (lo > 0) | (hi < 0)
            for xi, yi, s_ in zip(x, g["effect"], sig):
                ax.plot(xi, yi, marker=st["marker"], ms=5.5, color=st["color"],
                        mfc=st["color"] if s_ else BG, mew=1.4, zorder=3)
        ax.set_title(label, loc="left", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("log points relative to 1940", fontsize=9, color=MUTED)
        ax.tick_params(labelsize=8.5)
        style_axis(ax)
    for ax in axes[1]:
        ax.set_xticks(range(1900, 2001, 20))
    handles = [plt.Line2D([], [], color=st["color"], marker=st["marker"], lw=1.4, mfc=st["color"])
               for st in styles.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=CREAM, alpha=0.55))
    fig.legend(handles, list(styles) + ["1933–1958: large federal transfers"], loc="upper left",
               bbox_to_anchor=(0.005, 0.915), ncol=3, frameon=False, fontsize=9)
    titles(fig, "Effect of TVA by census year: change relative to 1940",
           "Oaxaca-Blinder estimate and 95% CI; filled = interval excludes 0",
           "Controls and samples of Kline and Moretti (2014). Rest of US: S.E. clustered by state; "
           "proposed authorities: spatial HAC (200 miles). 1920 and 1930 coefficients partly mechanical "
           "(1920/30 levels are controls).")
    fig.tight_layout(rect=(0, 0.025, 1, 0.9))
    save_figure(fig, OUT / "event_study.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data" / "build.dta"))
    args = ap.parse_args()
    check_data(args.data)
    OUT.mkdir(exist_ok=True)

    S = make_samples(build_variables(pd.read_stata(args.data, convert_categoricals=False)))
    es = estimate(S)
    es.to_csv(OUT / "event_study.csv", index=False)
    plot(es)

    # consistency check with Table III (decadalized 1940-2000 effect x 6)
    chk = es[(es["year"] == 2000) & (es["comparison"] == "Rest of US")].set_index("outcome")["effect"] / 6
    print("2000 effect / 6 (should equal Table III, Panel A, col. 3):")
    print(chk.round(3).to_string())


if __name__ == "__main__":
    main()
