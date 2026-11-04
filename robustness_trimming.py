"""
Robustness: how much do the main estimates depend on the trimming rule?

Kline & Moretti drop the 25% of control counties with the lowest propensity score.
This script re-estimates Table III, Panel A (Oaxaca-Blinder, S.E. clustered by state)
for manufacturing and agricultural employment with no trimming and with 10%, 25%
and 50% of controls dropped.

Usage:  python robustness_trimming.py --data data/build.dta
Output: output/robustness_trimming.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from build import build_variables                                  # noqa: E402
from estimators import oaxaca                                      # noqa: E402
from replicate import OUT, arrays, change, check_data, make_samples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data" / "build.dta"))
    args = ap.parse_args()
    check_data(args.data)
    d = build_variables(pd.read_stata(args.data, convert_categoricals=False))

    recs = []
    for q in (0, 0.10, 0.25, 0.50):
        s = make_samples(d, trim_q=q)["trim"]
        a = arrays(s)
        for var in ("lnmanuf", "lnagr"):
            dy = change(s, var, "40", "2000", 6)
            b, se, n = oaxaca(dy, a["X"], a["t"], "cluster", groups=a["groups"])
            recs.append(dict(trim_share=q, outcome=var, effect=b, se=se, N=n))
    out = pd.DataFrame(recs)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "robustness_trimming.csv", index=False)
    print(out.pivot(index="trim_share", columns="outcome", values=["effect", "se", "N"]).round(3).to_string())


if __name__ == "__main__":
    main()
