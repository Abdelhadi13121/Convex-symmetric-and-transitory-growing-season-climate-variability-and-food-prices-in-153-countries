"""The local-projection estimator used in every table.

    Delta_h y_{i,t+h} = b1*CV_it + b2*RV_it (+ interactions)
                        + sum_j ( CV_{i,t-j}, RV_{i,t-j}, dy_{i,t-j} )
                        + alpha_{i,month(t)} + delta_t + e

Fixed effects: country x calendar-month (absorbs country-specific harvest
price seasonality) and global month (absorbs world commodity/energy/dollar
cycles, so identification is cross-sectional differential weather).
Standard errors: Driscoll-Kraay with truncation L = |h|+1 by default.

IMPORTANT - pre-trend horizons.  At h < 0 the dependent variable is
    y_{t+h} - y_{t-1} = -( dy_{t+h+1} + ... + dy_{t-1} ),
which is an exact linear combination of the outcome lags dy_{t-1}..dy_{t-|h|+1}.
If those lags are in the control set the regression is degenerate (R^2 = 1,
coefficient identically zero).  We therefore SHIFT the whole control window
back by |h| at negative horizons, so controls are dated t-|h|-1 and earlier
and the placebo dependent variable is not mechanically spanned.  This is why
the panel stores lags out to config.P_LAG_MAX.
"""
import numpy as np
import pandas as pd

import config as C
from rep_toolkit import fe_ols


def control_lags(prefix, P=None, offset=0):
    P = C.P_LAG if P is None else P
    return [f"{prefix}_l{l}" for l in range(1 + offset, P + 1 + offset)]


def lp(df, h, shock, ycol="cfood", lagpref="dl", P=None, fe1="cm", fe2="t",
       se="dk", L=None, extra=(), mask=None, use_insample=True):
    """One local projection at horizon h. Returns a fe_ols result frame."""
    P = C.P_LAG if P is None else P
    offset = max(0, -h)                     # shift controls at placebo horizons
    if P + offset > C.P_LAG_MAX:
        P = max(1, C.P_LAG_MAX - offset)

    d = df
    if mask is not None:
        d = d[mask.reindex(d.index).fillna(False)]
    d = d.copy()

    # leads/lags of the outcome come from the complete calendar grid
    g = d.groupby("iso3", sort=False)
    d["yy"] = g[ycol].shift(-h) - g[ycol].shift(1)

    if use_insample:
        d = d[d["insample"]]

    names = (list(shock)
             + control_lags("C", P, offset)
             + control_lags("R", P, offset)
             + control_lags(lagpref, P, offset)
             + list(extra))
    cols = list(dict.fromkeys(["yy", "iso3", "t", fe1, fe2] + names))
    s = d[cols].dropna()
    if len(s) < 300:
        return None

    res = fe_ols(
        s["yy"].to_numpy(float),
        s[names].to_numpy(float),
        [pd.factorize(s[fe1])[0], pd.factorize(s[fe2])[0]],
        se=se,
        t=s["t"].to_numpy(),
        cluster=pd.factorize(s["iso3"])[0],
        L=(abs(h) + 1 if L is None else L),
        names=names,
    )
    res.attrs["nc"] = int(s["iso3"].nunique())
    res.attrs["sample"] = s
    return res


def standardise(series):
    return (series - series.mean()) / series.std()
