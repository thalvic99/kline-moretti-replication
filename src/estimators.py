"""
Estimators used by Kline & Moretti (2014).

Python port of x_ols_JP_v11.ado:
  * OLS with standard errors clustered by state (Stata `reg, cluster()`);
  * OLS with Conley (1999) spatial HAC standard errors, 200-mile Bartlett
    window in the north-south and east-west distances (x_ols_JP);
  * Kline (2011) Oaxaca-Blinder estimator of the effect of treatment on the
    treated, with either type of standard error (bo_x).

As in Stata, regressors that are collinear with earlier ones are omitted:
their coefficient and variance are set to zero.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Fit:
    b: np.ndarray       # coefficients (0 for omitted columns)
    V: np.ndarray       # variance matrix (0 rows/cols for omitted columns)
    n: int


def _independent_columns(X: np.ndarray) -> list[int]:
    """Keep a column only if it adds rank, scanning left to right (Stata-like)."""
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    keep: list[int] = []
    for j in range(X.shape[1]):
        if not np.any(Xs[:, j]):
            continue
        cand = keep + [j]
        if np.linalg.matrix_rank(Xs[:, cand]) == len(cand):
            keep.append(j)
    return keep


def _expand(b_k, V_k, keep, k_all):
    b = np.zeros(k_all)
    V = np.zeros((k_all, k_all))
    b[keep] = b_k
    V[np.ix_(keep, keep)] = V_k
    return b, V


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray) -> Fit:
    """OLS with Stata's clustered variance: G/(G-1) * (N-1)/(N-k)."""
    keep = _independent_columns(X)
    Xk = X[:, keep]
    XtX_inv = np.linalg.inv(Xk.T @ Xk)
    b = XtX_inv @ Xk.T @ y
    u = y - Xk @ b
    n, k = Xk.shape
    meat = np.zeros((k, k))
    gids = np.unique(groups)
    for g in gids:
        s = (Xk[groups == g] * u[groups == g, None]).sum(axis=0)
        meat += np.outer(s, s)
    G = len(gids)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    return Fit(*_expand(b, V, keep, X.shape[1]), n=n)


def conley_window(lat: np.ndarray, lon: np.ndarray, cut1=200.0, cut2=200.0) -> np.ndarray:
    """Bartlett weights from distance() and window() in the ado file.

    D1 is the north-south and D2 the east-west great-circle distance in miles.
    The earth radius uses the latitude of the first observation, as in Mata.
    """
    la = np.radians(lat)
    lo = np.radians(lon)
    R = 3963 - 13 * np.sin(la[0])
    a1 = np.sin((la[:, None] - la[None, :]) / 2) ** 2
    a2 = np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2
    D1 = R * 2 * np.arcsin(np.minimum(np.sqrt(a1), 1))
    D2 = R * 2 * np.arcsin(np.minimum(np.sqrt(a2), 1))
    return (1 - D1 / cut1) * (D1 < cut1) * (1 - D2 / cut2) * (D2 < cut2)


def ols_conley(y: np.ndarray, X: np.ndarray, lat: np.ndarray, lon: np.ndarray,
               cut: float = 200.0) -> Fit:
    """OLS with Conley spatial HAC variance, as in x_ols_JP.

    V = N/(N-K) (X'X)^-1 [sum_ij w_ij e_i e_j x_i x_j'] (X'X)^-1, where K counts
    every column passed (Mata keeps omitted columns in X).
    """
    keep = _independent_columns(X)
    Xk = X[:, keep]
    XtX_inv = np.linalg.inv(Xk.T @ Xk)
    b = XtX_inv @ Xk.T @ y
    e = y - Xk @ b
    W = conley_window(lat, lon, cut, cut)
    meat = Xk.T @ (W * np.outer(e, e)) @ Xk
    n, k_all = X.shape
    V = XtX_inv @ meat @ XtX_inv * n / (n - k_all)
    return Fit(*_expand(b, V, keep, k_all), n=n)


def oaxaca(y, X, t, se="cluster", groups=None, lat=None, lon=None):
    """Kline (2011) Oaxaca-Blinder estimate of the ATT, as in bo_x.

    Regress y on [(1-T)*X, (1-T), T] without a constant; the counterfactual
    mean for the treated is xbar_T' b0, where xbar_T averages [X, 1] over all
    treated units in the sample (including those with missing y, as in the
    original code). Returns (att, se, n).
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    t = np.asarray(t, float)
    ok = ~np.isnan(y)
    c = 1 - t
    design = np.column_stack([X * c[:, None], c, t])

    if se == "cluster":
        fit = ols_cluster(y[ok], design[ok], np.asarray(groups)[ok])
    else:
        fit = ols_conley(y[ok], design[ok], np.asarray(lat)[ok], np.asarray(lon)[ok])

    k = X.shape[1] + 1
    b0, mu1 = fit.b[:k], fit.b[k]
    xbar = np.column_stack([X[t == 1], np.ones((t == 1).sum())]).mean(axis=0)
    att = mu1 - xbar @ b0
    var = fit.V[k, k] + xbar @ fit.V[:k, :k] @ xbar
    return att, np.sqrt(var), fit.n


def diff_means(y, t, se="cluster", groups=None, lat=None, lon=None):
    """Unadjusted difference: regress y on T and a constant."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    ok = ~np.isnan(y)
    X = np.column_stack([t, np.ones_like(t)])[ok]
    if se == "cluster":
        fit = ols_cluster(y[ok], X, np.asarray(groups)[ok])
    else:
        fit = ols_conley(y[ok], X, np.asarray(lat)[ok], np.asarray(lon)[ok])
    return fit.b[0], np.sqrt(fit.V[0, 0]), fit.n
