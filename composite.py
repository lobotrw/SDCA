"""Composite risk + accumulation/distribution curve model.

Re-implementation of the "Investing Campus BTC Strategic Dollar Cost Averaging Signal"
web app (https://sdca-signals-automation-production-bdd2.up.railway.app/) as a mode that
sits next to the 200-week-SMA SDCA logic in ``sdca.py``. Nothing in ``sdca.py`` changes.

The web app is a single static HTML page. Everything is computed in the browser from
data embedded in the page, so the formulas below come from its JavaScript (no server
API is involved). See README, section "Composite risk model", for the full write-up.

Two modes
---------
* ``mode="causal"`` (default): every value on day t uses only data up to day t.
  - Price valuation (Asymmetric Tail Curvature): the tied-curvature quadratic quantile
    regression is refitted walk-forward (expanding window, every ``refit_every`` days)
    instead of using the published full-sample coefficients.
  - Sharpe: expanding percentile rank instead of the app's full-history rank.
  - On-chain risk: a free Coin Metrics community proxy (MVRV, NUPL, MVRV-Z, Puell),
    each ranked with expanding percentiles, or the app's own series (``onchain_source``).
* ``mode="replica"``: faithful copy of the app. Published full-sample ATC coefficients
  (hindsight) and the app's own indicator series as they are embedded in its page.

Indicator z-scores are in [-3, +3]; positive = cheap (buy side), negative = expensive.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SITE_URL = "https://sdca-signals-automation-production-bdd2.up.railway.app/"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"

# ---------------------------------------------------------------- ATC (price valuation)
# Rearranged Asymmetric Quadratic Quantile Regression, as used by the app:
#   Q_tau(log10 P) = c + a*x + b*x^2,  x = ln(days since 2009-01-01) - MU
# with the curvature b tied within the groups (Q1, Q10, Q25), (Q50), (Q75, Q95, Q99).
# The quantile prices are then sorted ascending (rearrangement, so bands never cross).
ATC_GENESIS = pd.Timestamp("2009-01-01")
ATC_MU = 7.9914
ATC_TAUS = (0.01, 0.10, 0.25, 0.50, 0.75, 0.95, 0.99)
ATC_GROUPS = ((0.01, 0.10, 0.25), (0.50,), (0.75, 0.95, 0.99))
ATC_RISK_MARKS = (1, 10, 25, 50, 75, 95, 99)
# Full-sample coefficients published in the app (it cites "Cowen 2026, Table 3"). They are
# fitted on 2010-2026 data, so using them on any earlier date is look-ahead.
ATC_PUBLISHED = {
    0.01: (2.837, 2.578, -0.0241),
    0.10: (2.933, 2.552, -0.0241),
    0.25: (3.004, 2.554, -0.0241),
    0.50: (3.214, 2.482, -0.1126),
    0.75: (3.562, 2.283, -0.3259),
    0.95: (3.897, 1.964, -0.3259),
    0.99: (4.028, 1.904, -0.3259),
}
EQM_BANDS = ((10, "1-10% Fire sale"), (25, "10-25% Accumulate"), (50, "25-50% Value"),
             (75, "50-75% Above mid"), (95, "75-95% Hot"), (101, "95-99% Bubble"))

# ---------------------------------------------------------------- composite + curve
INDICATORS = ("price", "sharpe", "cbpl", "onchain")
INDICATOR_LABELS = {
    "price": "Asymmetric Tail Curvature (price valuation)",
    "sharpe": "Sharpe (realized P/L, 14d EMA)",
    "cbpl": "Cost Basis Profit/Loss Ratio (7d EMA)",
    "onchain": "Onchain Risk Composite (7d EMA)",
}
DEFAULT_ENABLED = ("price", "cbpl", "onchain")   # app default: Sharpe toggle is off
COMPOSITE_BANDS = ((25, "Buy"), (50, "Accumulate"), (75, "Trim"), (100.0001, "Sell"))
CURVE_RISK_NODES = tuple(range(0, 101, 5))
CURVE_DEFAULT_VALUES = (10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -2, -4, -10)
MODES = ("causal", "replica")


# ================================================================ helpers
def _days_since_genesis(index):
    return np.maximum(1, np.round((pd.DatetimeIndex(index) - ATC_GENESIS).days.values))


def _atc_x(index):
    return np.log(_days_since_genesis(index)) - ATC_MU


def _bands_from_coefs(x, coefs):
    """coefs: {tau: (c, a, b)} -> (n, 7) array of ascending band prices."""
    raw = np.column_stack([10 ** (coefs[t][0] + coefs[t][1] * x + coefs[t][2] * x * x) for t in ATC_TAUS])
    return np.sort(raw, axis=1)


def expanding_pct_rank(values, min_periods=365):
    """Share of values up to and including day t that are <= the day-t value (causal)."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    seen = []
    import bisect
    for i, x in enumerate(v):
        if not np.isfinite(x):
            continue
        bisect.insort(seen, x)
        if len(seen) >= min_periods:
            out[i] = bisect.bisect_right(seen, x) / len(seen)
    return out


def expanding_robust_z(values, min_periods=365):
    """(v_t - median(v_<=t)) / (1.4826 * MAD(v_<=t)): causal robust z-score."""
    v = np.asarray(values, dtype=float)
    out = np.full(len(v), np.nan)
    hist = []
    for i, x in enumerate(v):
        if not np.isfinite(x):
            continue
        hist.append(x)
        if len(hist) >= min_periods:
            h = np.asarray(hist)
            m = np.median(h)
            mad = np.median(np.abs(h - m)) or 1.0
            out[i] = (x - m) / (1.4826 * mad)
    return out


# ================================================================ ATC model
def fit_tied_quantiles(x, y, groups=ATC_GROUPS):
    """Quantile regression of y on [1, x, x^2], curvature tied within each group.

    Solved exactly as a linear program (sum of check losses). Returns {tau: (c, a, b)}.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    coefs = {}
    for taus in groups:
        k = len(taus)
        p = 2 * k + 1
        par = np.zeros((k * n, p))
        for j in range(k):
            par[j * n:(j + 1) * n, j] = 1.0
            par[j * n:(j + 1) * n, k + j] = x
            par[j * n:(j + 1) * n, 2 * k] = x * x
        eye = sparse.identity(k * n, format="csr")
        a_eq = sparse.hstack([sparse.csr_matrix(par), eye, -eye], format="csr")
        cost = np.concatenate([np.zeros(p)] + [np.full(n, t) for t in taus] + [np.full(n, 1 - t) for t in taus])
        bounds = [(None, None)] * p + [(0, None)] * (2 * k * n)
        res = linprog(cost, A_eq=a_eq, b_eq=np.tile(y, k), bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"quantile fit failed: {res.message}")
        for j, t in enumerate(taus):
            coefs[t] = (float(res.x[j]), float(res.x[k + j]), float(res.x[2 * k]))
    return coefs


def atc_bands(close, mode="causal", refit_every=30, min_history=1460, fit_stride=7, progress=None):
    """DataFrame of the seven ATC quantile prices (columns q01..q99) for each day.

    replica: published full-sample coefficients (look-ahead).
    causal:  walk-forward. On every ``refit_every``-th day T (from ``min_history`` days of
             data on) the model is refitted on closes up to and including T, sampled every
             ``fit_stride`` days counting back from T, and those coefficients are used for
             T .. T+refit_every-1. Days before the first fit are NaN.
    """
    close = pd.Series(close, dtype=float).dropna()
    close = close[close > 0]
    idx = close.index
    x = _atc_x(idx)
    cols = [f"q{int(round(t * 100)):02d}" for t in ATC_TAUS]
    if mode == "replica":
        return pd.DataFrame(_bands_from_coefs(x, ATC_PUBLISHED), index=idx, columns=cols)
    if mode != "causal":
        raise ValueError(f"mode must be one of {MODES}")
    key = (len(close), close.index[0], close.index[-1], float(close.sum()), refit_every, min_history, fit_stride)
    if key in _ATC_CACHE:
        return _ATC_CACHE[key].copy()
    y = np.log10(close.values)
    out = np.full((len(idx), len(ATC_TAUS)), np.nan)
    fit_days = list(range(min_history - 1, len(idx), refit_every))
    coef_log = {}
    for n_fit, T in enumerate(fit_days):
        sel = np.arange(T, -1, -fit_stride)[::-1]
        coefs = fit_tied_quantiles(x[sel], y[sel])
        coef_log[idx[T]] = coefs
        end = min(len(idx), T + refit_every)
        out[T:end] = _bands_from_coefs(x[T:end], coefs)
        if progress:
            progress(n_fit + 1, len(fit_days))
    df = pd.DataFrame(out, index=idx, columns=cols)
    df.attrs["coefficients"] = coef_log
    _ATC_CACHE[key] = df
    return df.copy()


_ONCHAIN_CACHE = {}
_ATC_CACHE = {}       # walk-forward fits are deterministic; reuse them within one process


def atc_z(close, bands):
    """App's 'reversed EQM' z: +3 at Q1, 0 at Q50, -3 at Q99, linear in log price."""
    lp = np.log(np.asarray(close, float))
    ll, lm, lh = (np.log(bands[c].values) for c in ("q01", "q50", "q99"))
    below = np.clip(3 * (lm - lp) / np.where(lm - ll == 0, 1e-12, lm - ll), 0, 3)
    above = np.clip(-3 * (lp - lm) / np.where(lh - lm == 0, 1e-12, lh - lm), -3, 0)
    z = np.where(lp <= lm, below, above)
    ok = np.isfinite(ll) & np.isfinite(lm) & np.isfinite(lh) & (ll < lm) & (lm < lh)
    return pd.Series(np.where(ok, z, np.nan), index=bands.index)


def eqm_risk(close, bands):
    """App's side-panel 'EQM risk': log-interpolation of price between the 7 quantiles."""
    p = np.asarray(close, float)
    lv = np.log(bands[[f"q{int(round(t * 100)):02d}" for t in ATC_TAUS]].values)
    out = np.full(len(p), np.nan)
    for i in range(len(p)):
        row = lv[i]
        if not np.all(np.isfinite(row)) or not np.isfinite(p[i]):
            continue
        out[i] = np.clip(np.interp(np.log(p[i]), row, ATC_RISK_MARKS), 1, 99)
    return pd.Series(out, index=bands.index)


def eqm_band(risk):
    for hi, name in EQM_BANDS:
        if risk < hi:
            return name
    return EQM_BANDS[-1][1]


# ================================================================ data sources
def _get(url, params=None, timeout=60):
    import requests
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "sdca-composite/1.0"})
    r.raise_for_status()
    return r


def _cached(name, max_age_hours, fetch):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return json.loads(path.read_text())
    try:
        data = fetch()
    except Exception:
        if path.exists():           # stale cache beats nothing
            return json.loads(path.read_text())
        raise
    path.write_text(json.dumps(data))
    return data


def parse_site_html(html):
    """Pull the embedded indicator series (and BTC closes) out of the app's page."""
    out = {}
    for name in ("SHARPE_Z", "CBPL_Z", "ONCHAIN_Z"):
        m = re.search(rf"const {name}\s*=\s*(\{{.*?\}});", html, re.S)
        out[name] = json.loads(m.group(1)) if m else {"start": 0, "values": []}
    m = re.search(r"const EMBEDDED_BTC_DATA\s*=\s*(\[\[.*?\]\]);", html, re.S)
    out["BTC"] = json.loads(m.group(1)) if m else []
    return out


def load_site_data(url=SITE_URL, max_age_hours=12):
    """The app's embedded series (it is rebuilt daily). Cached under data/cache/ (git-ignored;
    the series are derived from ChartInspect data, so they are not redistributed in the repo)."""
    return _cached("railway_site.json", max_age_hours, lambda: parse_site_html(_get(url).text))


def site_series(site, name):
    block = site.get(name) or {"start": 0, "values": []}
    vals = [np.nan if v is None else float(v) for v in block["values"]]
    idx = pd.to_datetime(block["start"], unit="ms") + pd.to_timedelta(np.arange(len(vals)), unit="D")
    return pd.Series(vals, index=idx, dtype=float, name=name)


def load_coinmetrics(metrics=("PriceUSD", "CapMVRVCur", "CapMrktCurUSD", "IssTotUSD"),
                     start="2010-07-18", max_age_hours=12):
    """Free Coin Metrics community API (no key; data licensed CC BY-NC 4.0)."""
    def fetch():
        rows, url = [], COINMETRICS_URL
        params = {"assets": "btc", "metrics": ",".join(metrics), "frequency": "1d",
                  "start_time": start, "page_size": 10000}
        while url:
            j = _get(url, params=params).json()
            rows.extend(j.get("data", []))
            url, params = j.get("next_page_url"), None
        return rows
    rows = _cached("coinmetrics_btc.json", max_age_hours, fetch)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"].str[:10])
    df = df.set_index("time").drop(columns=["asset"], errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


# ================================================================ indicators
def sharpe_z(site_sharpe, mode="causal", min_periods=365):
    """The app stores z = 3 - 6 * rank, where rank is the FULL-history percentile rank of the
    14d-EMA realized-P/L Sharpe (look-ahead). A rank is monotone in the underlying value, so
    an expanding rank of the stored values equals an expanding rank of the raw Sharpe; that
    gives the exact causal version without the raw data."""
    s = site_sharpe.astype(float)
    if mode == "replica":
        return s
    rank = expanding_pct_rank(-s.values, min_periods)     # stored z is reversed: -z ~ Sharpe
    return pd.Series(np.clip(3 - 6 * rank, -3, 3), index=s.index)


def cbpl_z(site_cbpl, mode="causal"):
    """The app's cost-basis P/L ratio series is already an expanding (causal) robust z of the
    7d EMA. Only its 121 zero-loss days were filled with the all-history max (minor look-ahead
    we cannot undo without ChartInspect's raw data). Used as-is in both modes."""
    return site_cbpl.astype(float)


def onchain_proxy_z(cm, min_periods=365, ema_span=7):
    """Free proxy for ChartInspect's 'Onchain Risk Composite' from Coin Metrics community data.

    ChartInspect averages 8 sub-scores (active MVRV, NUPL, LTH PVR, supply P/L, P/L ratio x2,
    MVRV, MVRV z, Puell). Only MVRV, NUPL, MVRV z and Puell can be built from free data:
      MVRV   = CapMVRVCur
      NUPL   = 1 - 1/MVRV
      MVRV z = (market cap - realized cap) / expanding std(market cap)
      Puell  = daily issuance USD / 365d mean issuance USD
    Each is turned into an expanding percentile (0 cheap .. 1 expensive), averaged, smoothed
    with a 7d EMA, expanding robust z, sign flipped (+ = cheap), clipped to [-3, 3]."""
    mvrv = cm["CapMVRVCur"]
    mcap = cm["CapMrktCurUSD"]
    rcap = mcap / mvrv
    subs = {
        "mvrv": mvrv,
        "nupl": 1 - 1 / mvrv,
        "mvrv_z": (mcap - rcap) / mcap.expanding(min_periods=min_periods).std(),
        "puell": cm["IssTotUSD"] / cm["IssTotUSD"].rolling(365, min_periods=365).mean(),
    }
    ranks = pd.DataFrame({k: expanding_pct_rank(v.values, min_periods) for k, v in subs.items()}, index=cm.index)
    comp = ranks.mean(axis=1, skipna=False)
    ema = comp.ewm(span=ema_span, adjust=False, ignore_na=True).mean().where(comp.notna())
    z = -expanding_robust_z(ema.values, min_periods)
    return pd.Series(np.clip(z, -3, 3), index=cm.index), ranks


# ================================================================ composite
def composite(ind_z, enabled=DEFAULT_ENABLED, weights=None):
    """Weighted mean of the enabled indicator z's. A day is NaN unless EVERY enabled
    indicator has a value (the app never blends a partial set). Clipped to [-3, 3]."""
    enabled = [k for k in enabled if k in ind_z.columns]
    if not enabled:
        raise ValueError("enable at least one indicator")
    w = pd.Series({k: float(np.clip((weights or {}).get(k, 1.0), 0, 5)) for k in enabled})
    sub = ind_z[enabled]
    z = (sub * w).sum(axis=1, min_count=len(enabled)) / w.sum()
    z[sub.isna().any(axis=1)] = np.nan
    return z.clip(-3, 3)


def risk_from_z(z):
    """Linear map used by the app: risk = 50 - 50 * z / 3, clipped to 0..100."""
    return np.clip(50 - np.asarray(z, float) / 3 * 50, 0, 100)


def composite_band(risk):
    if not np.isfinite(risk):
        return "n/a"
    for hi, name in COMPOSITE_BANDS:
        if risk < hi:
            return name
    return "Sell"


def curve_rate(risk, nodes=CURVE_DEFAULT_VALUES, xs=CURVE_RISK_NODES):
    """% of cash bought (>0) or % of BTC sold (<0) per day, piecewise linear in risk."""
    r = np.clip(np.asarray(risk, float), 0, 100)
    out = np.interp(r, xs, nodes)
    return np.where(np.isfinite(np.asarray(risk, float)), out, 0.0)


def parse_curve(text):
    """'10,10,10,...' (21 values for risk 0,5,...,100) -> tuple of floats."""
    vals = tuple(float(v) for v in str(text).replace(";", ",").split(",") if v.strip())
    if len(vals) != len(CURVE_RISK_NODES):
        raise ValueError(f"curve needs {len(CURVE_RISK_NODES)} values (risk 0,5,...,100), got {len(vals)}")
    return tuple(float(np.clip(v, -10, 10)) for v in vals)


@dataclass
class Model:
    mode: str
    frame: pd.DataFrame                 # close, bands, indicator z's, composite z/risk
    enabled: tuple
    notes: list = field(default_factory=list)
    coefficients: dict = field(default_factory=dict)   # walk-forward fits {fit date: {tau: (c, a, b)}}

    def last_index(self):
        valid = self.frame["comp_z"].dropna()
        return valid.index[-1] if len(valid) else None


def build(close, mode="causal", enabled=DEFAULT_ENABLED, weights=None, site=None, cm=None,
          onchain_source=None, refit_every=30, progress=None):
    """Assemble everything into one DataFrame indexed by day."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    onchain_source = onchain_source or ("coinmetrics" if mode == "causal" else "site")
    notes = []
    close = pd.Series(close, dtype=float).dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    bands = atc_bands(close, mode=mode, refit_every=refit_every, progress=progress)
    coefficients = bands.attrs.get("coefficients", {})
    f = bands.copy()
    f.attrs = {}                       # keep attrs small: pandas deep-copies them on every op
    f.insert(0, "close", close)
    f["atc_z"] = atc_z(close, bands)
    f["eqm_risk"] = eqm_risk(close, bands)
    ind = pd.DataFrame(index=f.index)
    ind["price"] = f["atc_z"]
    need_site = any(k in enabled for k in ("sharpe", "cbpl")) or onchain_source == "site"
    if need_site and site is None:
        try:
            site = load_site_data()
        except Exception as e:  # pragma: no cover - network
            notes.append(f"app data unavailable ({e}); sharpe/cbpl/site-onchain skipped")
    for key, fn in (("sharpe", lambda s: sharpe_z(s, mode)), ("cbpl", lambda s: cbpl_z(s, mode))):
        name = {"sharpe": "SHARPE_Z", "cbpl": "CBPL_Z"}[key]
        ind[key] = fn(site_series(site, name)).reindex(f.index) if site else np.nan
    if onchain_source == "site":
        ind["onchain"] = site_series(site, "ONCHAIN_Z").reindex(f.index) if site else np.nan
    else:
        if cm is None:
            try:
                cm = load_coinmetrics()
            except Exception as e:  # pragma: no cover - network
                notes.append(f"Coin Metrics unavailable ({e}); onchain skipped")
        if cm is not None:
            ckey = (id(cm), len(cm))
            if ckey not in _ONCHAIN_CACHE:
                _ONCHAIN_CACHE[ckey] = onchain_proxy_z(cm)[0]
            z = _ONCHAIN_CACHE[ckey]
            ind["onchain"] = z.reindex(f.index)
        else:
            ind["onchain"] = np.nan
    usable = []
    for k in enabled:
        if ind[k].notna().any():
            usable.append(k)
        else:
            notes.append(f"indicator '{k}' has no data and was skipped")
    for k in INDICATORS:
        f[f"z_{k}"] = ind[k]
    f["comp_z"] = composite(ind, usable, weights)
    f["comp_risk"] = risk_from_z(f["comp_z"])
    m = Model(mode=mode, frame=f, enabled=tuple(usable), notes=notes, coefficients=coefficients)
    m.frame.attrs["onchain_source"] = onchain_source
    return m


# ================================================================ today + backtest
def today_action(model, cash, curve=CURVE_DEFAULT_VALUES):
    """Instruction for the last day the full composite is defined (on-chain feeds lag ~1d)."""
    d = model.last_index()
    if d is None:
        return {"date": None, "action": "Awaiting signal data"}
    row = model.frame.loc[d]
    risk = float(row["comp_risk"])
    rate = float(curve_rate(risk, curve))
    out = {"date": d.date().isoformat(), "price": float(row["close"]), "composite_z": float(row["comp_z"]),
           "composite_risk": risk, "band": composite_band(risk), "curve_rate_pct_per_day": rate,
           "cash": float(cash), "median": float(row["q50"]), "q01_rail": float(row["q01"]),
           "q99_rail": float(row["q99"]), "atc_z": float(row["atc_z"]), "eqm_risk": float(row["eqm_risk"]),
           "eqm_band": eqm_band(float(row["eqm_risk"]))}
    if rate > 1e-4:
        spend = cash * rate / 100
        out.update(action=f"BUY {rate:.2f}% of cash", spend_usd=spend, buy_btc=spend / row["close"])
    elif rate < -1e-4:
        out.update(action=f"SELL {-rate:.2f}% of BTC", sell_fraction_of_btc=-rate / 100)
    else:
        out.update(action="HOLD - no trade today")
    return out


def backtest(model, start="2015-01-01", capital=10000.0, curve=CURVE_DEFAULT_VALUES, end=None):
    """The app's curve backtest. Each day: rate = curve(risk_t); buy rate% of current cash
    (rate>0) or sell |rate|% of current BTC (rate<0) at that day's close; no fees. NaN risk =
    no trade. Benchmark: lump sum of the whole capital on the first day. Max drawdown is
    peak-to-trough of daily portfolio value."""
    f = model.frame.loc[pd.Timestamp(start):]
    if end is not None:
        f = f.loc[:pd.Timestamp(end)]
    f = f[f["close"] > 0]
    if f.empty:
        raise ValueError("no data in backtest window")
    cash, btc = float(capital), 0.0
    recs = []
    buy_days = sell_days = 0
    gross_bought = btc_bought = 0.0
    rates = curve_rate(f["comp_risk"].values, curve)
    for d, risk, price, rate in zip(f.index, f["comp_risk"].values, f["close"].values, rates):
        trade = 0.0
        if rate > 1e-4:
            usd = min(cash, cash * rate / 100)
            if usd > 0.005:
                cash -= usd
                btc += usd / price
                gross_bought += usd
                btc_bought += usd / price
                trade = usd
                buy_days += 1
        elif rate < -1e-4:
            sell = btc * (-rate / 100)
            if sell > 1e-12:
                cash += sell * price
                btc -= sell
                trade = -sell * price
                sell_days += 1
        recs.append((d, price, risk, rate, trade, cash, btc, cash + btc * price))
    ledger = pd.DataFrame(recs, columns=["date", "price", "risk", "rate", "trade_usd", "cash", "btc", "portfolio"]).set_index("date")
    lump_btc = capital / ledger["price"].iloc[0]
    ledger["lump_sum"] = lump_btc * ledger["price"]

    def mdd(s):
        return float(((s.cummax() - s) / s.cummax()).max() * 100)

    last = ledger.iloc[-1]
    valid = ledger["risk"].notna()
    summary = {
        "start": ledger.index[0].date().isoformat(), "end": ledger.index[-1].date().isoformat(),
        "days": len(ledger), "buy_days": buy_days, "sell_days": sell_days,
        "net_btc": float(last["btc"]), "cash_reserve": float(last["cash"]),
        "portfolio_value": float(last["portfolio"]), "lump_sum_value": float(last["lump_sum"]),
        "vs_lump_sum_usd": float(last["portfolio"] - last["lump_sum"]),
        "vs_lump_sum_pct": float((last["portfolio"] / last["lump_sum"] - 1) * 100),
        "avg_daily_rate_pct": float(ledger.loc[valid, "rate"].mean()) if valid.any() else float("nan"),
        "avg_buy_price": gross_bought / btc_bought if btc_bought else float("nan"),
        "max_drawdown_pct": mdd(ledger["portfolio"]), "max_drawdown_lump_pct": mdd(ledger["lump_sum"]),
    }
    return summary, ledger


# ================================================================ price loader
def load_close(drop_open_candle=True):
    """Same price history as the SDCA app (Yahoo BTC-USD spliced onto the bundled Coin
    Metrics pre-2014 closes). Falls back to Coin Metrics PriceUSD if Yahoo fails.
    Today's still-open UTC candle is dropped by default, so 'today' is the last closed day."""
    import sdca
    try:
        import yfinance as yf
        d = yf.download("BTC-USD", period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
        if isinstance(d, pd.DataFrame):
            d = d.iloc[:, 0]
        d = d.dropna()
        if d.empty:
            raise RuntimeError("empty Yahoo response")
        close = sdca.splice_history(sdca.load_pre2014_history(), d)
        source = "Yahoo BTC-USD + Coin Metrics pre-2014"
    except Exception:
        close = load_coinmetrics()["PriceUSD"].dropna()
        source = "Coin Metrics PriceUSD (Yahoo unavailable)"
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    if drop_open_candle:
        close = close[close.index < pd.Timestamp.now("UTC").tz_localize(None).normalize()]
    close.attrs["source"] = source
    return close
