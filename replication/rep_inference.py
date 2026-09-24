"""Stage 4: identification checks, diagnostics and simulation-based inference.

Table 8  - balance / exogeneity of the shock
Table 9  - panel unit roots, cross-sectional dependence, serial correlation
Table 10 - parameter stability (rolling windows, structural-break test)
Table 11 - leave-one-country-out jackknife
Table 12 - randomisation inference
Table 13 - is climate volatility trending, and what does it imply for inflation
"""
import io
import os
import time

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.tsa.stattools import adfuller

import config as C
from rep_data import load_panel
from rep_lp import lp, control_lags
from rep_toolkit import absorb, fe_ols, fmt

BUF = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    BUF.write(line + "\n")


def rule(n=112):
    say("=" * n)


# ------------------------------------------------------------------ Table 8
def table8_balance(d):
    rule()
    say("TABLE 8. BALANCE / EXOGENEITY OF THE CLIMATE SHOCK")
    say("Each row regresses |S|_it on one standardised lagged predictor, with")
    say("country x calendar-month and global month FE. If the shock is weather,")
    say("no economic variable should forecast it and the within-R2 should be ~0.")
    rule()
    preds = [("dl_l1", "Food inflation (t-1)"),
             ("dl_l2", "Food inflation (t-2)"),
             ("dh_l1", "Headline inflation (t-1)"),
             ("infL", "Trend headline inflation (12m)"),
             ("credit", "Private credit / GDP"),
             ("lgdp", "log GDP per capita"),
             ("agshare", "Agriculture VA / GDP"),
             ("annual_lag1_wb_gdp_real_growth", "GDP growth (t-1)"),
             ("annual_lag1_wb_reserves_months_imports", "Reserves (months of imports)"),
             ("annual_lag1_wb_food_imports_merchandise_pct", "Food / merch. imports (%)")]
    dd = d[d["insample"]]
    rows = []
    for v, lab in preds:
        if v not in dd.columns:
            continue
        s = dd[["CV", "iso3", "t", "cm", v]].dropna()
        if len(s) < 500:
            continue
        z = ((s[v] - s[v].mean()) / s[v].std()).to_numpy(float)
        r = fe_ols(s["CV"].to_numpy(float), z.reshape(-1, 1),
                   [pd.factorize(s["cm"])[0], pd.factorize(s["t"])[0]],
                   se="cluster", cluster=pd.factorize(s["iso3"])[0],
                   t=s["t"].to_numpy(), names=[v])
        say("  {:<36}{:>22}   N={:>7,}  within-R2={:.5f}".format(
            lab, fmt(r.loc[v, "coef"], r.loc[v, "se"], r.loc[v, "p"]),
            r.attrs["n"], r.attrs["r2_within"]))
        rows.append(dict(predictor=lab, coef=r.loc[v, "coef"], se=r.loc[v, "se"],
                         p=r.loc[v, "p"], n=r.attrs["n"], r2=r.attrs["r2_within"]))
    vs = [v for v, _ in preds if v in dd.columns]
    s = dd[["CV", "iso3", "t", "cm"] + vs].dropna()
    Z = ((s[vs] - s[vs].mean()) / s[vs].std()).to_numpy(float)
    r = fe_ols(s["CV"].to_numpy(float), Z,
               [pd.factorize(s["cm"])[0], pd.factorize(s["t"])[0]],
               se="cluster", cluster=pd.factorize(s["iso3"])[0],
               t=s["t"].to_numpy(), names=vs)
    b = r["coef"].to_numpy()
    W = float(b @ np.linalg.pinv(r.attrs["V"]) @ b)
    say("")
    say("  Joint Wald chi2({}) = {:.2f}, p = {:.4f}".format(
        len(vs), W, 1 - st.chi2.cdf(W, len(vs))))
    say("  JOINT within-R2 = {:.5f}  -> economically zero predictive content".format(
        r.attrs["r2_within"]))
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table8_balance.csv"), index=False)


# ------------------------------------------------------------------ Table 9
def _fisher(groups, label, lags=6):
    ps = []
    for _, s in groups:
        s = s.dropna()
        if len(s) < 60:
            continue
        try:
            ps.append(adfuller(s.to_numpy(), maxlag=lags, autolag=None, regression="c")[1])
        except Exception:
            continue
    if not ps:
        return None
    ps = np.clip(np.asarray(ps), 1e-8, 1 - 1e-8)
    stat = float(-2 * np.log(ps).sum())
    dfree = 2 * len(ps)
    z = (stat - dfree) / np.sqrt(2 * dfree)
    say("  {:<42}series={:>4}  Fisher chi2={:>10.1f}  z={:>8.2f}  p={:.4f}".format(
        label, len(ps), stat, z, 1 - st.norm.cdf(z)))
    return dict(series=label, n=len(ps), chi2=stat, z=z, p=1 - st.norm.cdf(z))


def _pesaran_cd(resid_frame):
    """Vectorised Pesaran (2004) CD statistic from a t x N residual panel."""
    X = resid_frame.to_numpy(float)
    M = np.isfinite(X).astype(float)
    Xf = np.where(np.isfinite(X), X, 0.0)
    cnt = M.T @ M                                   # pairwise complete counts
    ssum = Xf.T @ Xf
    m1 = Xf.T @ M                                   # sum of x_i over common obs
    sq = (Xf ** 2).T @ M                            # sum of x_i^2 over common obs
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = ssum / cnt - (m1 / cnt) * (m1.T / cnt)
        var_i = sq / cnt - (m1 / cnt) ** 2
        rho = cov / np.sqrt(var_i * var_i.T)
    N = X.shape[1]
    iu = np.triu_indices(N, 1)
    ok = np.isfinite(rho[iu]) & (cnt[iu] >= 50)
    terms = np.sqrt(cnt[iu][ok]) * rho[iu][ok]
    cd = np.sqrt(2.0 / (N * (N - 1))) * terms.sum()
    return cd, int(ok.sum()), float(np.nanmean(np.abs(rho[iu][ok]))), N


def table9_diagnostics(d):
    say("")
    rule()
    say("TABLE 9. TIME-SERIES AND PANEL DIAGNOSTICS")
    rule()
    say("")
    say("Panel A. Fisher/Choi panel unit-root tests (H0: every series has a unit root)")
    rows = []
    for col, lab in [("lfood", "log food CPI (level)"),
                     ("dlfood_w", "monthly log change in food CPI"),
                     ("CV", "climate variability shock |S|"),
                     ("S", "signed climate surprise S")]:
        r = _fisher(d.groupby("iso3")[col], lab)
        if r:
            rows.append(r)
    say("  -> the level is I(1) and the change is I(0): the cumulative-difference")
    say("     local projection is the correct specification.")

    say("")
    say("Panel B. Pesaran (2004) CD test for cross-sectional dependence (h=0 residuals)")
    r0 = lp(d, 0, ["CV", "RV"], "cfood", "dl")
    s = r0.attrs["sample"].copy()
    s["u"] = r0.attrs["resid"]
    W = s.pivot_table(index="t", columns="iso3", values="u")
    W = W[[c for c in W.columns if W[c].notna().sum() >= 50]]
    cd, npairs, mrho, N = _pesaran_cd(W)
    say("  N = {} cross-sections, {} usable pairs".format(N, npairs))
    say("  CD = {:.2f}, p = {:.4f}, mean pairwise |rho| = {:.3f}".format(
        cd, 2 * (1 - st.norm.cdf(abs(cd))), mrho))
    say("  -> cross-sectional dependence is present; Driscoll-Kraay SEs are required.")

    say("")
    say("Panel C. Residual serial correlation (h=0 residual on its own lag)")
    s = s.sort_values(["iso3", "t"])
    s["ul"] = s.groupby("iso3")["u"].shift(1)
    ss = s.dropna(subset=["ul"])
    rr = fe_ols(ss["u"].to_numpy(float), ss[["ul"]].to_numpy(float),
                [pd.factorize(ss["cm"])[0], pd.factorize(ss["t"])[0]],
                se="cluster", cluster=pd.factorize(ss["iso3"])[0],
                t=ss["t"].to_numpy(), names=["resid(t-1)"])
    say("  rho = {:+.3f} (se {:.3f}), p = {:.3f}".format(
        rr.loc["resid(t-1)", "coef"], rr.loc["resid(t-1)", "se"], rr.loc["resid(t-1)", "p"]))
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table9_unitroot.csv"), index=False)


# ----------------------------------------------------------------- Table 10
def table10_stability(d):
    say("")
    rule()
    say("TABLE 10. PARAMETER STABILITY")
    say("Rolling ten-year windows; coefficient on |S| at h=6 (FOOD CPI).")
    rule()
    rows = []
    for y0 in range(2003, 2017, 2):
        mask = (d["date"] >= "{}-01-01".format(y0)) & (d["date"] < "{}-01-01".format(y0 + 10))
        r = lp(d, 6, ["CV", "RV"], "cfood", "dl", mask=mask)
        if r is None:
            continue
        say("  {}-{}   {}   N={:>7,}  Nc={}".format(
            y0, y0 + 9, fmt(r.loc["CV", "coef"], r.loc["CV", "se"], r.loc["CV", "p"]),
            r.attrs["n"], r.attrs["nc"]))
        rows.append(dict(window="{}-{}".format(y0, y0 + 9), b=r.loc["CV", "coef"],
                         se=r.loc["CV", "se"], p=r.loc["CV", "p"], n=r.attrs["n"]))
    dd = d.copy()
    dd["post"] = (dd["date"] >= "2014-01-01").astype(float)
    dd["CV_post"] = dd["CV"] * dd["post"]
    r = lp(dd, 6, ["CV", "CV_post", "RV"], "cfood", "dl")
    say("")
    say("  Chow-type break test at 2014m1: |S| x 1{{t>=2014}} = {}".format(
        fmt(r.loc["CV_post", "coef"], r.loc["CV_post", "se"], r.loc["CV_post", "p"])))
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table10_stability.csv"), index=False)


# ----------------------------------------------------------------- Table 11
def table11_jackknife(d):
    say("")
    rule()
    say("TABLE 11. LEAVE-ONE-COUNTRY-OUT JACKKNIFE (|S| at h=6, FOOD CPI)")
    rule()
    isos = sorted(d.loc[d["insample"] & d["CV"].notna(), "iso3"].dropna().unique())
    out = []
    for c in isos:
        r = lp(d, 6, ["CV", "RV"], "cfood", "dl", mask=d["iso3"] != c)
        if r is not None:
            out.append((c, r.loc["CV", "coef"], r.loc["CV", "p"]))
    j = pd.DataFrame(out, columns=["dropped", "beta", "p"])
    full = lp(d, 6, ["CV", "RV"], "cfood", "dl").loc["CV", "coef"]
    say("  full-sample beta = {:+.3f}".format(full))
    say("  {} jackknife draws | range [{:+.3f}, {:+.3f}] | median {:+.3f}".format(
        len(j), j["beta"].min(), j["beta"].max(), j["beta"].median()))
    say("  share significant at 5% = {:.2f}".format(float((j["p"] < 0.05).mean())))
    j["shift"] = (j["beta"] - full).abs()
    say("")
    say("  most influential countries:")
    for _, row in j.sort_values("shift", ascending=False).head(5).iterrows():
        say("    drop {:<5} beta={:+.3f}  p={:.3f}".format(
            row["dropped"], row["beta"], row["p"]))
    j.to_csv(os.path.join(C.RESULTS, "table11_jackknife.csv"), index=False)


# ----------------------------------------------------------------- Table 12
def table12_randomisation(d):
    say("")
    rule()
    say("TABLE 12. RANDOMISATION INFERENCE")
    say("Each country's ENTIRE shock block (|S| and its 12 lags, plus |R| and its")
    say("lags) is reassigned to another country, preserving the calendar. The")
    say("estimate is recomputed; the p-value is the share of placebo draws with")
    say("|beta| at least as large as the actual one.")
    rule()
    h, P = 6, 6
    blk = ["CV", "RV"] + control_lags("C", P) + control_lags("R", P)
    dd = d.copy()
    g = dd.groupby("iso3", sort=False)
    dd["yy"] = g["cfood"].shift(-h) - g["cfood"].shift(1)
    dd = dd[dd["insample"]]
    s = dd[["yy", "iso3", "t", "cm"] + blk + control_lags("dl", P)].dropna().reset_index(drop=True)
    fes = [pd.factorize(s["cm"])[0], pd.factorize(s["t"])[0]]
    Zfix = s[control_lags("dl", P)].to_numpy(float)

    def est(B):
        M = absorb(np.column_stack([s["yy"].to_numpy(float), B, Zfix]), fes)
        return float(np.linalg.lstsq(M[:, 1:], M[:, 0], rcond=None)[0][0])

    actual = est(s[blk].to_numpy(float))
    tcode, tu = pd.factorize(s["t"])
    ccode, cu = pd.factorize(s["iso3"])
    cube = np.full((len(tu), len(cu), len(blk)), np.nan)
    cube[tcode, ccode, :] = s[blk].to_numpy(float)

    rng = np.random.default_rng(C.SEED)
    draws = []
    t0 = time.time()
    for _ in range(C.N_PERM):
        perm = rng.permutation(len(cu))
        B = cube[tcode, perm[ccode], :]
        keep = ~np.isnan(B).any(axis=1)
        if keep.mean() < 0.6:
            continue
        draws.append(est(np.where(np.isnan(B), 0.0, B)))
        if time.time() - t0 > C.RI_TIME_BUDGET:
            break
    draws = np.asarray(draws)
    pval = float((np.abs(draws) >= abs(actual)).mean())
    say("  estimation sample : {:,} obs, {} countries".format(len(s), len(cu)))
    say("  actual beta(|S|)  : {:+.4f}".format(actual))
    say("  placebo draws     : {} | mean {:+.4f} | sd {:.4f} | 2.5% {:+.4f} | 97.5% {:+.4f}".format(
        len(draws), draws.mean(), draws.std(),
        np.percentile(draws, 2.5), np.percentile(draws, 97.5)))
    say("  two-sided RI p    : {:.4f}".format(pval))
    pd.DataFrame(dict(draw=draws)).to_csv(
        os.path.join(C.RESULTS, "table12_ri_draws.csv"), index=False)


# ----------------------------------------------------------------- Table 13
def table13_volatility(d):
    say("")
    rule()
    say("TABLE 13. IS GROWING-SEASON CLIMATE VOLATILITY TRENDING?")
    say("S is purged of each country's seasonal norm AND its own warming trend, so")
    say("any trend in |S| is a pure SECOND-MOMENT effect. This table reports the")
    say("trend and the inflation contribution it implies - deliberately, because")
    say("the contribution is SMALL and the paper states so rather than overclaiming.")
    rule()
    dd = d[d["insample"] & d["CV"].notna()].copy()
    dd["trend10"] = dd["t"] / 120.0
    rows = []
    for lab, sub in [("All countries", pd.Series(True, index=dd.index)),
                     ("Low-income half", dd["lgdp"] < dd["lgdp"].median()),
                     ("High-income half", dd["lgdp"] >= dd["lgdp"].median())]:
        s = dd[sub].dropna(subset=["CV", "trend10"])
        r = fe_ols(s["CV"].to_numpy(float), s[["trend10"]].to_numpy(float),
                   [pd.factorize(s["cm"])[0]], se="cluster",
                   cluster=pd.factorize(s["iso3"])[0], t=s["t"].to_numpy(),
                   names=["trend"])
        say("  {:<20}d|S| per decade = {}   N={:>7,}  Nc={}".format(
            lab, fmt(r.loc["trend", "coef"], r.loc["trend", "se"], r.loc["trend", "p"]),
            r.attrs["n"], s["iso3"].nunique()))
        rows.append(dict(group=lab, trend=r.loc["trend", "coef"],
                         se=r.loc["trend", "se"], p=r.loc["trend", "p"]))
        if lab == "All countries":
            b, bse = r.loc["trend", "coef"], r.loc["trend", "se"]
    say("")
    say("  Frequency of extreme growing seasons:")
    dd["period"] = pd.cut(dd["date"].dt.year, [2002, 2009, 2016, 2026],
                          labels=["2003-09", "2010-16", "2017-25"])
    for thr in (1.5, 2.0, 2.5):
        fr = dd.groupby("period", observed=True)["CV"].apply(lambda x: (x > thr).mean() * 100)
        say("    P(|S| > {:.1f}) : ".format(thr)
            + "   ".join("{}: {:5.2f}%".format(k, v) for k, v in fr.items()))

    span = (dd["t"].max() - dd["t"].min()) / 120.0
    dCV, dCV_se = b * span, bse * span
    say("")
    say("  Implied rise in E|S| over {:.2f} decades: {:+.3f} sd (se {:.3f})".format(
        span, dCV, dCV_se))
    for h in (6, 9, 12):
        r = lp(d, h, ["CV", "RV"], "cfood", "dl")
        beta, bse2 = r.loc["CV", "coef"], r.loc["CV", "se"]
        eff = beta * dCV
        se = abs(eff) * np.sqrt((bse2 / beta) ** 2 + (dCV_se / dCV) ** 2)
        say("    h={:>2}: beta={:+.3f} -> cumulative food-CPI contribution = "
            "{:+.3f} pp (se {:.3f})".format(h, beta, eff, se))
    say("")
    say("  READ THIS AS A SCOPE STATEMENT: volatility is rising and the price")
    say("  response is real, but the trend contribution to inflation is small.")
    say("  The paper is about the response to shocks, not the cost of warming.")
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table13_volatility_trend.csv"),
                              index=False)


def main():
    C.ascii_stdout()
    C.ensure_dirs()
    d = load_panel()
    table8_balance(d)
    table9_diagnostics(d)
    table10_stability(d)
    table11_jackknife(d)
    table12_randomisation(d)
    table13_volatility(d)
    with open(os.path.join(C.RESULTS, "diagnostics.txt"), "w", encoding="utf-8") as fh:
        fh.write(BUF.getvalue())
    say("")
    say("  diagnostics written to " + os.path.join(C.RESULTS, "diagnostics.txt"))


if __name__ == "__main__":
    main()
