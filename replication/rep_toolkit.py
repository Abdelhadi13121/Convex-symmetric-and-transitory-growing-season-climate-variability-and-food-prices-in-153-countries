"""Panel estimation toolkit: high-dimensional fixed-effect absorption and
cross-sectionally robust variance estimators.

Deliberately dependency-light (numpy/pandas/scipy only) so the package runs
on any machine that can import pandas.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ absorption
def absorb(Y, fe_list, tol=1e-10, maxit=300):
    """Alternating-projections within transformation.

    Y        : (n,) or (n,k) array
    fe_list  : list of integer code arrays, one per fixed effect dimension
    """
    Y = np.asarray(Y, dtype=float).copy()
    if Y.ndim == 1:
        Y = Y[:, None]
    if len(fe_list) == 1:
        c = fe_list[0]
        cnt = np.bincount(c)
        for j in range(Y.shape[1]):
            m = np.bincount(c, weights=Y[:, j]) / cnt
            Y[:, j] -= m[c]
        return Y
    for _ in range(maxit):
        delta = 0.0
        for c in fe_list:
            cnt = np.bincount(c)
            for j in range(Y.shape[1]):
                m = np.bincount(c, weights=Y[:, j]) / cnt
                Y[:, j] -= m[c]
                if m.size:
                    delta = max(delta, float(np.abs(m).max()))
        if delta < tol:
            break
    return Y


# ------------------------------------------------------------------- meat mats
def _meat_cluster(X, u, g):
    k = X.shape[1]
    M = np.zeros((k, k))
    order = np.argsort(g, kind="mergesort")
    Xs, us, gs = X[order], u[order], np.asarray(g)[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    bounds = np.r_[starts, len(gs)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = Xs[a:b].T @ us[a:b]
        M += np.outer(s, s)
    return M


def _meat_dk(X, u, t, L):
    """Driscoll-Kraay: Newey-West in time on cross-sectionally summed scores."""
    k = X.shape[1]
    order = np.argsort(t, kind="mergesort")
    Xs, us, ts = X[order], u[order], np.asarray(t)[order]
    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    bounds = np.r_[starts, len(ts)]
    h = np.zeros((len(bounds) - 1, k))
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        h[i] = Xs[a:b].T @ us[a:b]
    M = h.T @ h
    for l in range(1, int(L) + 1):
        if l >= h.shape[0]:
            break
        w = 1.0 - l / (L + 1.0)
        G = h[l:].T @ h[:-l]
        M += w * (G + G.T)
    return M


# --------------------------------------------------------------------- fe_ols
def fe_ols(y, X, fe_list, se="dk", cluster=None, t=None, L=None, names=None):
    """OLS after absorbing fixed effects, with a choice of robust variance.

    se : 'dk' (Driscoll-Kraay), 'cluster' (one-way by `cluster`),
         'twoway' (Cameron-Gelbach-Miller), or 'robust'.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    Z = absorb(np.column_stack([y, X]), fe_list)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XX = Xt.T @ Xt
    XXi = np.linalg.pinv(XX)
    b = XXi @ (Xt.T @ yt)
    u = yt - Xt @ b
    n, k = Xt.shape

    if se == "dk":
        if L is None:
            L = 4
        M = _meat_dk(Xt, u, t, L)
        G = len(np.unique(t))
        adj = G / max(G - 1, 1) * n / max(n - k, 1)
    elif se == "cluster":
        M = _meat_cluster(Xt, u, cluster)
        G = len(np.unique(cluster))
        adj = G / max(G - 1, 1) * (n - 1) / max(n - k, 1)
    elif se == "twoway":
        gg = pd.factorize(
            pd.Series(np.asarray(cluster)).astype(str)
            + "_"
            + pd.Series(np.asarray(t)).astype(str)
        )[0]
        M = _meat_cluster(Xt, u, cluster) + _meat_cluster(Xt, u, t) - _meat_cluster(Xt, u, gg)
        G = min(len(np.unique(cluster)), len(np.unique(t)))
        adj = G / max(G - 1, 1) * (n - 1) / max(n - k, 1)
    else:
        M = Xt.T @ ((u ** 2)[:, None] * Xt)
        adj = n / max(n - k, 1)

    V = XXi @ M @ XXi * adj
    V = 0.5 * (V + V.T)
    seb = np.sqrt(np.maximum(np.diag(V), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = np.where(seb > 0, b / np.where(seb > 0, seb, 1.0), np.nan)
    p = 2.0 * (1.0 - stats.norm.cdf(np.abs(tv)))

    idx = list(names) if names is not None else [f"x{i}" for i in range(k)]
    res = pd.DataFrame({"coef": b, "se": seb, "t": tv, "p": p}, index=idx)
    tss = float(((yt - yt.mean()) ** 2).sum())
    res.attrs.update(
        V=V, n=int(n), resid=u, fitted=Xt @ b,
        r2_within=(1.0 - float((u ** 2).sum()) / tss) if tss > 0 else np.nan,
    )
    return res


# ------------------------------------------------------------------ utilities
def lincom(res, weights):
    """Linear combination of coefficients with its Driscoll-Kraay/cluster SE."""
    idx = list(res.index)
    V = res.attrs["V"]
    w = np.zeros(len(idx))
    for key, val in weights.items():
        w[idx.index(key)] = val
    b = float(w @ res["coef"].values)
    se = float(np.sqrt(max(w @ V @ w, 0.0)))
    if se == 0:
        return b, se, np.nan
    return b, se, float(2 * (1 - stats.norm.cdf(abs(b / se))))


def stars(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(b, se, p):
    """ASCII-only coefficient cell: '+0.199 (0.052)***'."""
    return "{:>+7.3f} ({:.3f}){:<3}".format(b, se, stars(p))


def residualise_on_fe_and_trend(frame, col, fe_col, unit_col, time_col):
    """Remove unit x calendar-month means and a unit-specific linear trend.

    Returns a Series aligned to `frame.index` (NaN where `col` is missing).
    """
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    mask = frame[col].notna()
    if mask.sum() == 0:
        return out
    sub = frame.loc[mask]
    Z = absorb(
        np.column_stack([sub[col].to_numpy(float), sub[time_col].to_numpy(float)]),
        [pd.factorize(sub[fe_col])[0]],
    )
    unit = pd.factorize(sub[unit_col])[0]
    res = np.empty(len(sub))
    for u in np.unique(unit):
        k = unit == u
        yy, xx = Z[k, 0], Z[k, 1]
        denom = float(xx @ xx)
        res[k] = yy - (float(xx @ yy) / denom) * xx if denom > 0 else yy
    out.loc[mask] = res
    return out
