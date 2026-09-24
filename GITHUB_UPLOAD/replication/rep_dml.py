"""Stage 3: Double / debiased machine learning (Chernozhukov et al., 2018).

Partially linear model at each horizon h:

    Delta_h y = theta1 * |S| + theta2 * (|S| x zCredit) + g(X) + e

g(.) is a gradient-boosted tree ensemble. Cross-fitting uses 5 folds that are
CLUSTERED BY COUNTRY, so no country appears in both the training fold and the
evaluation fold. Fixed effects (country x calendar-month, global month) are
absorbed from y, D and X before the ML step, which keeps the Neyman-orthogonal
moment linear in the FE. Inference uses Driscoll-Kraay on the orthogonalised
residuals.
"""
import io
import os

import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

import config as C
from rep_data import load_panel
from rep_lp import control_lags
from rep_toolkit import absorb, fmt, _meat_dk

BUF = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    BUF.write(line + "\n")


# Structural / macro covariates entering g(X). Irrigation is deliberately
# excluded: it is observed for only a third of the panel and would cut the
# estimation sample by more than half.
XVARS = [
    "lgdp", "agshare",
    "annual_lag1_wb_urban_population_pct",
    "annual_lag1_wb_cereal_yield",
    "annual_lag1_wb_agricultural_employment_pct",
    "annual_lag1_wb_fertilizer_per_hectare",
    "annual_lag1_wb_food_production_index",
    "annual_lag1_wb_electricity_access_pct",
    "annual_lag1_legacy_ndgain_vulnerability",
    "annual_lag1_legacy_ndgain_readiness",
    "annual_lag1_wb_arable_land_pct",
    "annual_lag1_wb_reserves_months_imports",
    "annual_lag1_wb_gdp_real_growth",
    "annual_lag1_wb_inflation_pct",
    "annual_lag1_legacy_trade_openness",
    "annual_lag1_wb_food_imports_merchandise_pct",
    "area_tot", "infL",
]


def dml_one(d, h, ycol="cfood", lagpref="dl", seed=None):
    seed = C.SEED if seed is None else seed
    xv = [v for v in XVARS if v in d.columns]
    lagx = control_lags("C") + control_lags("R") + control_lags(lagpref)

    dd = d.copy()
    g = dd.groupby("iso3", sort=False)
    dd["yy"] = g[ycol].shift(-h) - g[ycol].shift(1)
    zc = (dd["lcredit"] - dd["lcredit"].mean()) / dd["lcredit"].std()
    dd["zc"] = zc
    dd["D1"] = dd["CV"]
    dd["D2"] = dd["CV"] * dd["zc"]
    dd = dd[dd["insample"]]

    cols = list(dict.fromkeys(
        ["yy", "iso3", "t", "cm", "D1", "D2", "zc", "RV"] + xv + lagx))
    s = dd[cols].dropna().reset_index(drop=True)
    if len(s) < 500:
        return None

    fes = [pd.factorize(s["cm"])[0], pd.factorize(s["t"])[0]]
    xnames = ["zc", "RV"] + xv + lagx
    M = absorb(np.column_stack([s["yy"].to_numpy(float),
                                s["D1"].to_numpy(float),
                                s["D2"].to_numpy(float),
                                s[xnames].to_numpy(float)]), fes)
    Y, D, X = M[:, 0], M[:, 1:3], M[:, 3:]

    groups = s["iso3"].to_numpy()
    n_splits = min(C.DML_FOLDS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    Yr = np.zeros_like(Y)
    Dr = np.zeros_like(D)
    for tr, te in gkf.split(X, Y, groups=groups):
        mY = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=4,
            random_state=seed).fit(X[tr], Y[tr])
        Yr[te] = Y[te] - mY.predict(X[te])
        for j in range(D.shape[1]):
            mD = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=4,
                random_state=seed).fit(X[tr], D[tr, j])
            Dr[te, j] = D[te, j] - mD.predict(X[te])

    XX = Dr.T @ Dr
    theta = np.linalg.solve(XX, Dr.T @ Yr)
    u = Yr - Dr @ theta
    tvec = s["t"].to_numpy()
    meat = _meat_dk(Dr, u, tvec, abs(h) + 1)
    XXi = np.linalg.inv(XX)
    G = len(np.unique(tvec))
    adj = G / max(G - 1, 1) * len(Y) / max(len(Y) - D.shape[1], 1)
    V = XXi @ meat @ XXi * adj
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    p = 2 * (1 - st.norm.cdf(np.abs(theta / se)))
    return dict(theta=theta, se=se, p=p, V=V, n=len(Y),
                nc=int(s["iso3"].nunique()), nx=len(xnames))


def main():
    C.ascii_stdout()
    C.ensure_dirs()
    d = load_panel()
    ins = d["insample"]
    mu = d.loc[ins, "lcredit"].mean()
    sd = d.loc[ins, "lcredit"].std()
    q10 = d.loc[ins, "lcredit"].quantile(0.10)
    q90 = d.loc[ins, "lcredit"].quantile(0.90)
    z10, z90 = (q10 - mu) / sd, (q90 - mu) / sd

    say("=" * 112)
    say("TABLE 7. DOUBLE / DEBIASED MACHINE LEARNING  (Chernozhukov et al. 2018)")
    say("Partially linear model: Delta_h y = theta1*|S| + theta2*(|S| x zCredit) + g(X) + e")
    say("g(.) gradient-boosted trees; {}-fold cross-fitting with folds clustered by country;".format(C.DML_FOLDS))
    say("fixed effects absorbed before the ML step; Driscoll-Kraay SE on orthogonalised residuals.")
    say("=" * 112)
    say("{:>4}{:>23}{:>25}{:>23}{:>23}{:>9}{:>5}".format(
        "h", "theta1  |S|", "theta2  |S| x zCredit",
        "effect at credit p10", "effect at credit p90", "N", "Nc"))
    rows = []
    for h in [3, 6, 9, 12, 18]:
        r = dml_one(d, h)
        if r is None:
            continue
        th, se, p, V = r["theta"], r["se"], r["p"], r["V"]

        def lc(w):
            b = float(w @ th)
            s_ = float(np.sqrt(max(w @ V @ w, 0.0)))
            return b, s_, 2 * (1 - st.norm.cdf(abs(b / s_))) if s_ > 0 else np.nan

        say("{:>4}{:>23}{:>25}{:>23}{:>23}{:>9,}{:>5}".format(
            h, fmt(th[0], se[0], p[0]), fmt(th[1], se[1], p[1]),
            fmt(*lc(np.array([1.0, z10]))), fmt(*lc(np.array([1.0, z90]))),
            r["n"], r["nc"]))
        rows.append(dict(h=h, theta1=th[0], theta1_se=se[0], theta1_p=p[0],
                         theta2=th[1], theta2_se=se[1], theta2_p=p[1],
                         eff_p10=lc(np.array([1.0, z10]))[0],
                         eff_p90=lc(np.array([1.0, z90]))[0],
                         n=r["n"], nc=r["nc"], n_controls=r["nx"]))
    say("")
    say("  private credit/GDP at p10 = {:.1f}% of GDP, at p90 = {:.1f}% of GDP".format(
        float(np.exp(q10)), float(np.exp(q90))))
    say("  controls entering g(X): {} macro/structural covariates + 36 lag terms".format(
        len([v for v in XVARS if v in d.columns])))
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table7_dml.csv"), index=False)
    with open(os.path.join(C.RESULTS, "dml.txt"), "w", encoding="utf-8") as fh:
        fh.write(BUF.getvalue())


if __name__ == "__main__":
    main()
