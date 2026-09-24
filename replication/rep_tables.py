"""Stage 2: Tables 1-6 (descriptives, baseline IRFs, convexity, mechanism,
heterogeneity, robustness). All output is ASCII-only and is echoed to
results/tables.txt as well as the console.
"""
import io
import os

import numpy as np
import pandas as pd
from scipy import stats as st

import config as C
from rep_data import load_panel
from rep_lp import lp, standardise
from rep_toolkit import fe_ols, fmt, lincom

BUF = io.StringIO()


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    BUF.write(line + "\n")


def rule(char="=", n=112):
    say(char * n)


# ----------------------------------------------------------------- Table 1
def table1_sample(d):
    rule()
    say("TABLE 1. SAMPLE AND DESCRIPTIVE STATISTICS")
    rule()
    sm = d[d["insample"] & d["CV"].notna()]
    say("  Panel          : {} countries, monthly, {} to {}".format(
        sm["iso3"].nunique(), sm["date"].min().date(), sm["date"].max().date()))
    say("  Country-months : {:,} with a defined climate shock".format(len(sm)))
    say("  Excluded       : maize (all columns, at read time); chronic-hyperinflation")
    say("                   economies; FAO-imputed prices; |headline yoy| >= 40%")
    cols = ["CV", "S", "RV", "dlfood_w", "dlhead_w", "relf_w", "credit", "lgdp",
            "dep_wheat", "dep_rice"]
    labs = ["|S|  climate variability shock",
            "S    signed climate surprise",
            "|R|  precipitation variability",
            "Food CPI, monthly log change (%)",
            "Headline CPI, monthly log change (%)",
            "Relative food price change (%)",
            "Private credit / GDP (%)",
            "log GDP per capita",
            "Wheat import dependence",
            "Rice import dependence"]
    t = sm[cols].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    say("")
    say("  {:<38}{:>9}{:>11}{:>11}{:>11}{:>11}{:>11}".format(
        "variable", "N", "mean", "sd", "min", "median", "max"))
    for c, lab in zip(cols, labs):
        r = t.loc[c]
        say("  {:<38}{:>9,}{:>11.3f}{:>11.3f}{:>11.3f}{:>11.3f}{:>11.3f}".format(
            lab, int(r["count"]), r["mean"], r["std"], r["min"], r["50%"], r["max"]))
    t.to_csv(os.path.join(C.RESULTS, "table1_descriptives.csv"))


# ----------------------------------------------------------------- Table 2
def table2_baseline(d):
    say("")
    rule()
    say("TABLE 2. BASELINE IMPULSE RESPONSES")
    say("Cumulative response (percentage points) to a one-standard-deviation")
    say("ABSOLUTE growing-season temperature anomaly |S|.")
    say("Country x calendar-month FE, global month FE, 12 lags, Driscoll-Kraay SE (L=h+1).")
    rule()
    say("{:>11}{:>23}{:>23}{:>23}{:>10}{:>6}".format(
        "h (months)", "FOOD CPI", "RELATIVE FOOD PRICE", "HEADLINE CPI", "N", "Nc"))
    rows = []
    for h in [-12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21, 24]:
        a = lp(d, h, ["CV", "RV"], "cfood", "dl")
        b = lp(d, h, ["CV", "RV"], "crel", "dr")
        c = lp(d, h, ["CV", "RV"], "chead", "dh")
        if a is None or b is None or c is None:
            continue
        say("{:>11}  {}  {}  {}{:>10,}{:>6}".format(
            h,
            fmt(a.loc["CV", "coef"], a.loc["CV", "se"], a.loc["CV", "p"]),
            fmt(b.loc["CV", "coef"], b.loc["CV", "se"], b.loc["CV", "p"]),
            fmt(c.loc["CV", "coef"], c.loc["CV", "se"], c.loc["CV", "p"]),
            a.attrs["n"], a.attrs["nc"]))
        rows.append(dict(
            h=h,
            food=a.loc["CV", "coef"], food_se=a.loc["CV", "se"], food_p=a.loc["CV", "p"],
            rel=b.loc["CV", "coef"], rel_se=b.loc["CV", "se"], rel_p=b.loc["CV", "p"],
            head=c.loc["CV", "coef"], head_se=c.loc["CV", "se"], head_p=c.loc["CV", "p"],
            n=a.attrs["n"], nc=a.attrs["nc"]))
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table2_baseline_irf.csv"), index=False)

    zs = []
    for h in range(-12, -1):
        r = lp(d, h, ["CV", "RV"], "cfood", "dl")
        if r is not None:
            zs.append(r.loc["CV", "coef"] / r.loc["CV", "se"])
    zs = np.asarray(zs)
    stat = float((zs ** 2).sum())
    crit = float(st.chi2.ppf(0.95, len(zs)))
    say("")
    say("  Joint pre-trend test, h = -12..-2 : sum of squared z = {:.2f}".format(stat))
    say("  chi2({}) 5% critical value       : {:.2f}   -> {}".format(
        len(zs), crit, "NOT rejected" if stat < crit else "REJECTED"))
    return rows


# ----------------------------------------------------------------- Table 3
def table3_convexity(d):
    say("")
    rule()
    say("TABLE 3. THE CONTRIBUTION - CONVEXITY AND SYMMETRY")
    say("Panel A races the signed specification used in the price literature against")
    say("the quadratic one. Panel B tests heat against cold. Panel C is nonparametric.")
    rule()
    say("")
    say("Panel A. Signed level S versus squared S^2 (outcome: FOOD CPI)")
    say("{:>5}{:>44}{:>26}{:>10}".format(
        "h", "S  (linear: the literature's regressor)", "S^2  (convex)", "N"))
    rows = []
    for h in [3, 6, 9, 12, 18]:
        r = lp(d, h, ["S", "S2", "RV"], "cfood", "dl")
        say("{:>5}{:>44}{:>26}{:>10,}".format(
            h,
            fmt(r.loc["S", "coef"], r.loc["S", "se"], r.loc["S", "p"]),
            fmt(r.loc["S2", "coef"], r.loc["S2", "se"], r.loc["S2", "p"]),
            r.attrs["n"]))
        rows.append(dict(h=h, lin=r.loc["S", "coef"], lin_se=r.loc["S", "se"],
                         lin_p=r.loc["S", "p"], quad=r.loc["S2", "coef"],
                         quad_se=r.loc["S2", "se"], quad_p=r.loc["S2", "p"]))
    pd.DataFrame(rows).to_csv(
        os.path.join(C.RESULTS, "table3a_linear_vs_quadratic.csv"), index=False)

    say("")
    say("Panel B. Heat tail versus cold tail (both entered as positive magnitudes)")
    say("{:>5}{:>23}{:>23}{:>34}{:>10}".format(
        "h", "HEAT  S+", "COLD  S-", "Wald test of equality", "N"))
    rows = []
    for h in [0, 3, 6, 9, 12, 15, 18]:
        r = lp(d, h, ["CVp", "CVn", "RV"], "cfood", "dl")
        b, se, p = lincom(r, {"CVp": 1, "CVn": -1})
        say("{:>5}  {}  {}    diff={:+.3f} (se {:.3f}) p={:.3f}{:>10,}".format(
            h,
            fmt(r.loc["CVp", "coef"], r.loc["CVp", "se"], r.loc["CVp", "p"]),
            fmt(r.loc["CVn", "coef"], r.loc["CVn", "se"], r.loc["CVn", "p"]),
            b, se, p, r.attrs["n"]))
        rows.append(dict(h=h, heat=r.loc["CVp", "coef"], heat_se=r.loc["CVp", "se"],
                         cold=r.loc["CVn", "coef"], cold_se=r.loc["CVn", "se"],
                         diff=b, diff_se=se, diff_p=p))
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table3b_symmetry.csv"), index=False)

    say("")
    say("Panel C. Nonparametric bin response (reference bin: |S| <= 0.5, i.e. a")
    say("normal growing season). Bins in standard deviations of the local norm.")
    edges = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
    labels = ["S <= -1.5  (severe cold)", "-1.5 < S <= -0.5  (mild cold)",
              "-0.5 < S <= 0.5  (normal)", "0.5 < S <= 1.5  (mild heat)",
              "S > 1.5  (severe heat)"]
    dd = d.copy()
    dd["bin"] = pd.cut(dd["S"], bins=edges, labels=False)
    dum = pd.get_dummies(dd["bin"], prefix="b").astype(float)
    ref = "b_2.0" if "b_2.0" in dum.columns else dum.columns[2]
    keep = [c for c in dum.columns if c != ref]
    dd = pd.concat([dd, dum[keep]], axis=1)
    rows = []
    for h in (6, 9, 12):
        r = lp(dd, h, keep + ["RV"], "cfood", "dl")
        if r is None:
            continue
        say("")
        say("  h = {} months        share of sample      effect vs a normal season".format(h))
        for c in keep:
            k = int(float(str(c).split("_")[1]))
            shr = float((dd.loc[dd["insample"], "bin"] == k).mean()) * 100
            say("  {:<34}{:>8.1f}%{:>28}".format(
                labels[k], shr, fmt(r.loc[c, "coef"], r.loc[c, "se"], r.loc[c, "p"])))
            rows.append(dict(h=h, bin=labels[k], share=shr, b=r.loc[c, "coef"],
                             se=r.loc[c, "se"], p=r.loc[c, "p"]))
        say("  N={:,}  countries={}".format(r.attrs["n"], r.attrs["nc"]))
    pd.DataFrame(rows).to_csv(
        os.path.join(C.RESULTS, "table3c_bin_response.csv"), index=False)


# ----------------------------------------------------------------- Table 4
def table4_mechanism(d):
    say("")
    rule()
    say("TABLE 4. MECHANISM - THE QUANTITY SIDE (independent annual data)")
    say("Annual panel; dep. var. is the annual log change (%); country and year FE;")
    say("SE clustered by country. |S| is the calendar-year mean of the monthly shock.")
    rule()
    m = d[d["CV"].notna()].copy()
    m["year"] = m["date"].dt.year
    ann = (m.groupby(["iso3", "year"])
             .agg(CV=("CV", "mean"), RV=("RV", "mean"), nobs=("CV", "size"))
             .reset_index())
    ann = ann[ann["nobs"] >= 10]

    a = pd.read_csv(C.ANNUAL_CSV, low_memory=False,
                    usecols=lambda c: "maize" not in c.lower())
    use = ["iso3", "year", "fao_wheat_production_yield_kg_per_ha",
           "fao_rice_production_yield_kg_per_ha", "wb_cereal_yield",
           "wb_crop_production_index", "wb_food_production_index"]
    a = a[[c for c in use if c in a.columns]]
    df = a.merge(ann, on=["iso3", "year"], how="inner").sort_values(["iso3", "year"])
    df["CV_l1"] = df.groupby("iso3")["CV"].shift(1)
    df["RV_l1"] = df.groupby("iso3")["RV"].shift(1)

    targets = [("fao_rice_production_yield_kg_per_ha", "Rice yield (kg/ha)"),
               ("fao_wheat_production_yield_kg_per_ha", "Wheat yield (kg/ha)"),
               ("wb_cereal_yield", "Cereal yield (WDI)"),
               ("wb_crop_production_index", "Crop production index"),
               ("wb_food_production_index", "Food production index")]
    say("{:<28}{:>22}{:>22}{:>10}{:>9}{:>6}".format(
        "outcome", "|S|_t", "|S|_t-1", "sum", "N", "Nc"))
    rows = []
    for col, lab in targets:
        if col not in df.columns:
            continue
        df["_y"] = np.log(df[col].where(df[col] > 0))
        df["_dy"] = df.groupby("iso3")["_y"].diff() * 100.0
        s = df[["_dy", "CV", "CV_l1", "RV", "RV_l1", "iso3", "year"]].dropna()
        if len(s) < 300:
            continue
        nm = ["|S|_t", "|S|_t-1", "|R|_t", "|R|_t-1"]
        r = fe_ols(s["_dy"].to_numpy(float),
                   s[["CV", "CV_l1", "RV", "RV_l1"]].to_numpy(float),
                   [pd.factorize(s["iso3"])[0], pd.factorize(s["year"])[0]],
                   se="cluster", cluster=pd.factorize(s["iso3"])[0],
                   t=s["year"].to_numpy(), names=nm)
        tot = r.loc["|S|_t", "coef"] + r.loc["|S|_t-1", "coef"]
        say("{:<28}{:>22}{:>22}{:>10.3f}{:>9,}{:>6}".format(
            lab,
            fmt(r.loc["|S|_t", "coef"], r.loc["|S|_t", "se"], r.loc["|S|_t", "p"]),
            fmt(r.loc["|S|_t-1", "coef"], r.loc["|S|_t-1", "se"], r.loc["|S|_t-1", "p"]),
            tot, r.attrs["n"], s["iso3"].nunique()))
        rows.append(dict(outcome=lab, b0=r.loc["|S|_t", "coef"], se0=r.loc["|S|_t", "se"],
                         p0=r.loc["|S|_t", "p"], b1=r.loc["|S|_t-1", "coef"],
                         se1=r.loc["|S|_t-1", "se"], p1=r.loc["|S|_t-1", "p"],
                         total=tot, n=r.attrs["n"]))
    say("")
    say("  Output falls on impact and rebounds the following year: a TRANSITORY")
    say("  quantity shock, which is what generates the hump-shaped price path.")
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table4_mechanism.csv"), index=False)


# ----------------------------------------------------------------- Table 5
def table5_heterogeneity(d):
    say("")
    rule()
    say("TABLE 5. HETEROGENEITY BY FINANCIAL DEPTH  (secondary result)")
    say("|S| interacted with standardised log private credit/GDP. The log GDP per")
    say("capita interaction is always included, so this is not an income margin.")
    rule()
    dd = d.copy()
    dd["zc"] = standardise(dd["lcredit"])
    dd["zg"] = standardise(dd["lgdp"])
    dd["i_c"] = dd["CV"] * dd["zc"]
    dd["i_g"] = dd["CV"] * dd["zg"]
    SH = ["CV", "i_c", "i_g", "zc", "zg"]
    ins = dd["insample"]
    qs = dd.loc[ins, "lcredit"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    mu = dd.loc[ins, "lcredit"].mean()
    sd = dd.loc[ins, "lcredit"].std()
    hdr = "".join("{:>21}".format(str(int(round(np.exp(qs.loc[q])))) + "%") for q in qs.index)
    say("{:>4}{:>23}{:>23}  | effect at private credit/GDP =".format(
        "h", "|S| at mean credit", "|S| x zCredit"))
    say("{:>4}{:>23}{:>23}  |{}".format("", "", "", hdr))
    rows = []
    last = None
    for h in [3, 6, 9, 12, 18]:
        r = lp(dd, h, SH, "cfood", "dl")
        last = r
        line = "{:>4}  {}  {}  |".format(
            h,
            fmt(r.loc["CV", "coef"], r.loc["CV", "se"], r.loc["CV", "p"]),
            fmt(r.loc["i_c", "coef"], r.loc["i_c", "se"], r.loc["i_c", "p"]))
        rec = dict(h=h, main=r.loc["CV", "coef"], main_se=r.loc["CV", "se"],
                   inter=r.loc["i_c", "coef"], inter_se=r.loc["i_c", "se"],
                   inter_p=r.loc["i_c", "p"], n=r.attrs["n"])
        for q in qs.index:
            zq = (qs.loc[q] - mu) / sd
            b, se, p = lincom(r, {"CV": 1, "i_c": zq})
            line += "{:>21}".format(fmt(b, se, p))
            rec["eff_p{}".format(int(q * 100))] = b
        say(line)
        rows.append(rec)
    say("  N={:,}  countries={}".format(last.attrs["n"], last.attrs["nc"]))
    say("")
    say("  CAVEAT carried into the manuscript: this interaction is NOT robust to")
    say("  excluding 2020m1-2022m12 (Table 6, row 9). It is reported as a secondary")
    say("  result; nothing in Table 2 or Table 3 depends on it.")
    pd.DataFrame(rows).to_csv(
        os.path.join(C.RESULTS, "table5_credit_heterogeneity.csv"), index=False)


# ----------------------------------------------------------------- Table 6
def robustness_specs(d):
    """The fifteen specifications used by both Table 6 and Table 15.

    Defined once so the two tables always line up; make_tables.py merges them
    on the specification label.
    """
    dates = d["date"]
    crisis = (dates.between("2007-01-01", "2009-12-31")
              | dates.between("2020-01-01", "2022-12-31"))
    areamean = d.groupby("iso3")["area_tot"].transform("mean")
    EU = ["AUT", "BEL", "BGR", "HRV", "CZE", "DNK", "EST", "FIN", "FRA", "DEU",
          "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "NLD", "POL", "PRT",
          "ROU", "SVK", "SVN", "ESP", "SWE", "GBR", "CHE", "NOR"]
    return [
        ("(1)  Baseline", {}),
        ("(2)  Driscoll-Kraay, L=24", dict(L=24)),
        ("(3)  Cluster by country", dict(se="cluster")),
        ("(4)  Two-way cluster (country, month)", dict(se="twoway")),
        ("(5)  Country FE only (no cal-month FE)", dict(fe1="iso3")),
        ("(6)  6 lags", dict(P=6)),
        ("(7)  3 lags", dict(P=3)),
        ("(8)  Excl. 2007m1-2009m12",
         dict(mask=~dates.between("2007-01-01", "2009-12-31"))),
        ("(9)  Excl. 2020m1-2022m12",
         dict(mask=~dates.between("2020-01-01", "2022-12-31"))),
        ("(10) Excl. both crisis windows", dict(mask=~crisis)),
        ("(11) 2003-2013 only", dict(mask=dates < "2014-01-01")),
        ("(12) 2014-2025 only", dict(mask=dates >= "2014-01-01")),
        ("(13) Excl. GDP p.c. above USD 20,000",
         dict(mask=d["annual_lag1_wb_gdp_pc_usd"] < 20000)),
        ("(14) Excl. OECD-Europe", dict(mask=~d["iso3"].isin(EU))),
        ("(15) Smallest growing-area tercile out",
         dict(mask=areamean > areamean.quantile(1.0 / 3.0))),
    ]


def table6_robustness(d):
    say("")
    rule()
    say("TABLE 6. ROBUSTNESS")
    say("Column 1: coefficient on |S| in the baseline at h=6 (Table 2).")
    say("Column 2: coefficient on |S| x zCredit at h=12 (Table 5).")
    rule()
    dd = d.copy()
    dd["zc"] = standardise(dd["lcredit"])
    dd["zg"] = standardise(dd["lgdp"])
    dd["i_c"] = dd["CV"] * dd["zc"]
    dd["i_g"] = dd["CV"] * dd["zg"]
    SH = ["CV", "i_c", "i_g", "zc", "zg"]
    specs = robustness_specs(dd)
    say("{:<42}{:>22}{:>24}{:>11}".format(
        "specification", "|S| at h=6", "|S| x zCredit at h=12", "N (h=6)"))
    rows = []
    for lab, kw in specs:
        ra = lp(dd, 6, ["CV", "RV"], "cfood", "dl", **kw)
        rb = lp(dd, 12, SH, "cfood", "dl", **kw)
        ca = fmt(ra.loc["CV", "coef"], ra.loc["CV", "se"], ra.loc["CV", "p"]) if ra is not None else "n.a."
        cb = fmt(rb.loc["i_c", "coef"], rb.loc["i_c", "se"], rb.loc["i_c", "p"]) if rb is not None else "n.a."
        say("{:<42}{:>22}{:>24}{:>11,}".format(
            lab, ca, cb, ra.attrs["n"] if ra is not None else 0))
        rows.append(dict(spec=lab,
                         base=ra.loc["CV", "coef"] if ra is not None else np.nan,
                         base_se=ra.loc["CV", "se"] if ra is not None else np.nan,
                         base_p=ra.loc["CV", "p"] if ra is not None else np.nan,
                         cred=rb.loc["i_c", "coef"] if rb is not None else np.nan,
                         cred_se=rb.loc["i_c", "se"] if rb is not None else np.nan,
                         cred_p=rb.loc["i_c", "p"] if rb is not None else np.nan,
                         n=ra.attrs["n"] if ra is not None else 0))

    say("")
    say("  Alternative outcomes at h=6:")
    for ycol, lagp, lab in [("crel", "dr", "relative food price"),
                            ("chead", "dh", "headline CPI")]:
        r = lp(dd, 6, ["CV", "RV"], ycol, lagp)
        say("    {:<24}{:>22}".format(
            lab, fmt(r.loc["CV", "coef"], r.loc["CV", "se"], r.loc["CV", "p"])))
    pd.DataFrame(rows).to_csv(os.path.join(C.RESULTS, "table6_robustness.csv"), index=False)


def table15_convexity_robustness(d):
    """The paper's central robustness table: does the CONVEXITY survive?

    Table 6 asks the same question of the level coefficient. This one runs the
    linear and quadratic terms through the identical fifteen specifications,
    and is what the manuscript's Table 7 and Fig. 4 report.
    """
    say("")
    rule()
    say("TABLE 15. ROBUSTNESS OF THE FUNCTIONAL-FORM RESULT")
    say("Linear S and quadratic S^2 at h=6, outcome cumulative FOOD CPI,")
    say("across the same fifteen specifications used in Table 6.")
    rule()
    say("{:<42}{:>22}{:>22}{:>11}".format(
        "specification", "linear, S", "quadratic, S^2", "N"))
    rows = []
    for lab, kw in robustness_specs(d):
        r = lp(d, 6, ["S", "S2", "RV"], "cfood", "dl", **kw)
        if r is None:
            say("{:<42}{:>22}".format(lab, "n.a."))
            continue
        say("{:<42}{:>22}{:>22}{:>11,}".format(
            lab,
            fmt(r.loc["S", "coef"], r.loc["S", "se"], r.loc["S", "p"]),
            fmt(r.loc["S2", "coef"], r.loc["S2", "se"], r.loc["S2", "p"]),
            r.attrs["n"]))
        rows.append(dict(spec=lab, lin=r.loc["S", "coef"], lin_se=r.loc["S", "se"],
                         lin_p=r.loc["S", "p"], quad=r.loc["S2", "coef"],
                         quad_se=r.loc["S2", "se"], quad_p=r.loc["S2", "p"],
                         n=r.attrs["n"]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(C.RESULTS, "table15_convexity_robustness.csv"), index=False)
    say("")
    say("  quadratic significant at 5% in {} of {}, at 10% in {}".format(
        int((t["quad_p"] < 0.05).sum()), len(t), int((t["quad_p"] < 0.10).sum())))
    say("  quadratic range {:.3f} to {:.3f}, median {:.3f}".format(
        t["quad"].min(), t["quad"].max(), t["quad"].median()))
    say("  linear significant at 5% in {} of {}".format(
        int((t["lin_p"] < 0.05).sum()), len(t)))


def table14_state_dependence(d):
    """Does a tight global food market amplify local growing-season shocks?

    Reported in the manuscript as a null: it is one of the two explanations
    tested and rejected for the period sensitivity of the level coefficient.
    """
    say("")
    rule()
    say("TABLE 14. STATE DEPENDENCE ON GLOBAL FOOD-MARKET TIGHTNESS")
    say("Tight months are the top three deciles of the cross-country median of")
    say("12-month food inflation. The level is absorbed by the time effects;")
    say("the interaction with the country-specific shock is not.")
    rule()
    dd = d.copy()
    g = dd[dd["insample"]].groupby("t")["dlfood_w"].median() * 12
    dd["gfood"] = dd["t"].map(g)
    dd["gtight"] = (dd["gfood"] > dd["gfood"].quantile(0.70)).astype(float)
    dd["CV_tight"] = dd["CV"] * dd["gtight"]
    say("{:>4}{:>24}{:>24}{:>26}{:>11}".format(
        "h", "|S| in slack market", "|S| x tight", "effect in tight market", "N"))
    rows = []
    for h in (3, 6, 9, 12):
        r = lp(dd, h, ["CV", "CV_tight", "RV"], "cfood", "dl")
        if r is None:
            continue
        tight = lincom(r, {"CV": 1, "CV_tight": 1})
        say("{:>4}{:>24}{:>24}{:>26}{:>11,}".format(
            h,
            fmt(r.loc["CV", "coef"], r.loc["CV", "se"], r.loc["CV", "p"]),
            fmt(r.loc["CV_tight", "coef"], r.loc["CV_tight", "se"], r.loc["CV_tight", "p"]),
            fmt(*tight), r.attrs["n"]))
        rows.append(dict(h=h, slack=r.loc["CV", "coef"], slack_se=r.loc["CV", "se"],
                         inter=r.loc["CV_tight", "coef"], inter_se=r.loc["CV_tight", "se"],
                         inter_p=r.loc["CV_tight", "p"], tight=tight[0],
                         tight_se=tight[1], tight_p=tight[2], n=r.attrs["n"]))
    pd.DataFrame(rows).to_csv(
        os.path.join(C.RESULTS, "table14_state_dependence.csv"), index=False)


def main():
    C.ascii_stdout()
    C.ensure_dirs()
    d = load_panel()
    table1_sample(d)
    table2_baseline(d)
    table3_convexity(d)
    table4_mechanism(d)
    table5_heterogeneity(d)
    table6_robustness(d)
    table15_convexity_robustness(d)
    table14_state_dependence(d)
    with open(os.path.join(C.RESULTS, "tables.txt"), "w", encoding="utf-8") as fh:
        fh.write(BUF.getvalue())
    say("")
    say("  tables written to " + os.path.join(C.RESULTS, "tables.txt"))


if __name__ == "__main__":
    main()
