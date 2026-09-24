"""Stage 1: build the analysis panel from the two raw CSVs.

Design notes that matter for replication:

* Maize is dropped at read time (column filter), never enters any frame.
* Leads and lags are taken on a COMPLETE monthly calendar grid per country,
  so a horizon-h lead is always exactly h calendar months ahead even where
  the estimation sample filter removes interior observations.
* The climate shock is constructed on the full panel (weather has nothing to
  do with price-data quality); the sample filter only decides which rows may
  serve as time-t observations in a regression.
"""
import numpy as np
import pandas as pd

import config as C
from rep_toolkit import residualise_on_fe_and_trend


def _read_monthly():
    head = pd.read_csv(C.MONTHLY_CSV, nrows=0).columns.tolist()
    keep = [c for c in head if "maize" not in c.lower()]
    dropped = len(head) - len(keep)
    df = pd.read_csv(C.MONTHLY_CSV, usecols=keep, low_memory=False)
    print(f"  monthly file: {len(df):,} rows, {len(keep)} columns kept, "
          f"{dropped} maize columns dropped at read time")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["iso3", "date"]).reset_index(drop=True)


def _complete_grid(df):
    """Reindex to a full iso3 x month grid so shift() is calendar-exact."""
    lo, hi = df["date"].min(), df["date"].max()
    months = pd.date_range(lo, hi, freq="MS")
    isos = df["iso3"].dropna().unique()
    grid = pd.MultiIndex.from_product([isos, months], names=["iso3", "date"])
    out = (df.set_index(["iso3", "date"])
             .reindex(grid)
             .reset_index())
    print(f"  calendar grid: {len(isos)} countries x {len(months)} months "
          f"= {len(out):,} rows ({len(out) - len(df):,} filler rows added)")
    return out


def _import_dependence(df, crop):
    P = df[f"annual_lag1_fao_{crop}_production_production_t"]
    M = df[f"annual_lag1_fao_{crop}_trade_import_quantity_t"].fillna(0)
    X = df[f"annual_lag1_fao_{crop}_trade_export_quantity_t"].fillna(0)
    cons = (P.fillna(0) + M - X).clip(lower=0)
    return (M / (cons + 1e-9)).where(cons > 0).clip(0, 1)


def build_panel(verbose=True):
    C.ensure_dirs()
    C.check_inputs()
    if verbose:
        print("[1/4] reading raw monthly file")
    df = _read_monthly()
    df = _complete_grid(df)

    df["t"] = (df["date"].dt.year - 2003) * 12 + df["date"].dt.month - 1
    df["mo"] = df["date"].dt.month
    df["cm"] = df["iso3"].astype(str) + "_" + df["mo"].astype(str)

    # ------------------------------------------------ prices: levels -> changes
    if verbose:
        print("[2/4] constructing price variables")
    df["lfood"] = np.log(df["fao_food_cpi"].where(df["fao_food_cpi"] > 0))
    df["lhead"] = np.log(df["fao_headline_cpi"].where(df["fao_headline_cpi"] > 0))
    g = df.groupby("iso3", sort=False)
    df["dlfood"] = g["lfood"].diff() * 100.0
    df["dlhead"] = g["lhead"].diff() * 100.0
    for v in ("dlfood", "dlhead"):
        lo, hi = df[v].quantile([C.WINSOR, 1 - C.WINSOR])
        df[v + "_w"] = df[v].clip(lo, hi)
    df["relf_w"] = df["dlfood_w"] - df["dlhead_w"]

    # cumulative indices rebuilt from winsorised changes (outlier-robust LP)
    g = df.groupby("iso3", sort=False)
    df["cfood"] = g["dlfood_w"].cumsum()
    df["chead"] = g["dlhead_w"].cumsum()
    df["crel"] = g["relf_w"].cumsum()

    # ------------------------------------------------------------- the shock
    if verbose:
        print("[3/4] constructing the growing-season climate shock")
    sur_t, sur_p, area = {}, {}, {}
    for crop in C.CROPS:
        ok = df[f"{crop}_domestic_weather_quality_pass"] == True   # noqa: E712
        for tag, src, store in (
            ("tmp", f"{crop}_domestic_tmp_growing_exposure_z", sur_t),
            ("pre", f"{crop}_domestic_pre_growing_exposure_z", sur_p),
        ):
            tmp = df[[src, "cm", "iso3", "t"]].copy()
            tmp[src] = tmp[src].where(ok)
            store[crop] = residualise_on_fe_and_trend(
                tmp, src, fe_col="cm", unit_col="iso3", time_col="t")
        area[crop] = df[f"{crop}_domestic_growing_area_ha"].where(ok)

    W = pd.concat([area[c] for c in C.CROPS], axis=1).fillna(0.0)
    wsum = W.sum(axis=1).replace(0.0, np.nan)
    have = pd.concat([sur_t[c] for c in C.CROPS], axis=1).notna().any(axis=1)
    df["Ssur"] = (sum(sur_t[c].fillna(0) * area[c].fillna(0) for c in C.CROPS) / wsum).where(have)
    df["Rsur"] = (sum(sur_p[c].fillna(0) * area[c].fillna(0) for c in C.CROPS) / wsum).where(have)
    df["area_tot"] = wsum

    df["CV"] = df["Ssur"].abs()          # climate-variability shock  |S|
    df["RV"] = df["Rsur"].abs()          # precipitation counterpart  |R|
    df["CVp"] = df["Ssur"].clip(lower=0)             # heat tail
    df["CVn"] = -df["Ssur"].clip(upper=0)            # cold tail
    df["S"] = df["Ssur"]
    df["S2"] = df["Ssur"] ** 2

    # -------------------------------------------------- moderators and lags
    if verbose:
        print("[4/4] moderators, lags, and the estimation-sample flag")
    df["lgdp"] = np.log(df["annual_lag1_wb_gdp_pc_usd"].where(df["annual_lag1_wb_gdp_pc_usd"] > 0))
    df["credit"] = df["annual_lag1_wb_private_credit_gdp"]
    df["lcredit"] = np.log(df["credit"].where(df["credit"] > 0))
    df["agshare"] = df["annual_lag1_wb_agriculture_va_gdp"]
    for crop in C.CROPS:
        df[f"dep_{crop}"] = _import_dependence(df, crop)

    g = df.groupby("iso3", sort=False)
    for l in range(1, C.P_LAG_MAX + 1):
        df[f"C_l{l}"] = g["CV"].shift(l)
        df[f"R_l{l}"] = g["RV"].shift(l)
        df[f"dl_l{l}"] = g["dlfood_w"].shift(l)
        df[f"dr_l{l}"] = g["relf_w"].shift(l)
        df[f"dh_l{l}"] = g["dlhead_w"].shift(l)

    df["inf12"] = g["dlhead_w"].transform(lambda s: s.rolling(12, min_periods=9).mean() * 12)
    df["infL"] = df.groupby("iso3", sort=False)["inf12"].shift(1)

    # estimation-sample filter (applies to time-t rows only; leads/lags are
    # always taken from the complete grid above)
    imputed = (df["food_flag"] == "I") | (df["food_yoy_uses_E_or_I"] == True)  # noqa: E712
    df["insample"] = (
        df["iso3"].notna()
        & ~df["iso3"].isin(C.HYPER)
        & ~imputed
        & (df["fao_headline_inflation_yoy_pct"].abs() < C.YOY_CAP)
    )

    df.to_pickle(C.PANEL_PKL)
    if verbose:
        sm = df[df["insample"] & df["CV"].notna()]
        print(f"  saved -> {C.PANEL_PKL}")
        print(f"  estimation-eligible: {len(sm):,} country-months, "
              f"{sm['iso3'].nunique()} countries, "
              f"{sm['date'].min().date()} to {sm['date'].max().date()}")
    return df


def load_panel():
    import os
    if not os.path.exists(C.PANEL_PKL):
        return build_panel()
    return pd.read_pickle(C.PANEL_PKL)


if __name__ == "__main__":
    C.ascii_stdout()
    build_panel()
