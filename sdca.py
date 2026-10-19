"""Core calculations for the Crypto SDCA System.

Kept free of Streamlit so it can be unit-tested and reused.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SMA_WINDOW = 200 * 7            # 200 weeks expressed in daily bars
Z_MIN_PERIODS = 365             # need one year of deviation history before a Z-score is shown
DEFAULT_IN_THRESHOLD = -1.0     # recalibrated for the log-deviation, no-look-ahead Z (see README)
DEFAULT_OUT_THRESHOLD = 1.5

# Cycle-relative sell mode (see cycle_percentile)
CYCLE_WINDOW = 1460             # ~one halving epoch: 210,000 blocks x 10 min = 1,458 days
CYCLE_MIN_PERIODS = 730         # need two years of deviation history before ranking
DEFAULT_CYCLE_OUT_THRESHOLD = 0.85
SELL_MODES = ("cycle", "zscore")

DATA_DIR = Path(__file__).resolve().parent / "data"
PRE2014_CSV = DATA_DIR / "btc_pre2014.csv"


def load_pre2014_history(path=PRE2014_CSV):
    """Bundled daily BTC/USD closes 2010-07-18 .. 2014-09-16 (Coin Metrics community data, PriceUSD)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["Close"].astype(float).sort_index()


def splice_history(older, newer):
    """Join two daily close series; ``newer`` wins wherever the dates overlap."""
    older = older.copy()
    newer = newer.copy()
    older.index = pd.to_datetime(older.index).tz_localize(None).normalize()
    newer.index = pd.to_datetime(newer.index).tz_localize(None).normalize()
    older = older[older.index < newer.index.min()]
    out = pd.concat([older, newer]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.name = "BTC-USD"
    return out


def compute_sdca(close, sma_window=SMA_WINDOW, z_min_periods=Z_MIN_PERIODS):
    """Return a DataFrame with the price, its 200-week SMA, the log deviation and its Z-score.

    * ``LogDev`` = ln(close / SMA). A relative (log) gap, so a 2013 deviation and a
      2025 deviation are comparable even though the dollar prices differ by 1000x.
    * ``Z_Diff`` is computed WITHOUT look-ahead: each day's Z uses only the mean and
      standard deviation of ``LogDev`` from the first SMA day up to and including that
      day (an expanding window). We prefer expanding over a fixed rolling window because
      a 200-week SMA cycle is ~4 years long; a rolling window short enough to be useful
      would only ever contain one or two cycles and would re-centre itself inside each
      bull/bear phase, flattening the very extremes we want to detect. The expanding
      window uses every cycle seen so far, and past values never change when new data
      arrives. ``z_min_periods`` (one year) avoids meaningless values from a tiny sample.
    """
    close = pd.Series(close, dtype=float).sort_index()
    df = pd.DataFrame({"BTC-USD": close})
    df["SMA_BTC"] = df["BTC-USD"].rolling(window=sma_window, min_periods=sma_window).mean()
    df["LogDev"] = np.log(df["BTC-USD"] / df["SMA_BTC"])
    exp = df["LogDev"].expanding(min_periods=z_min_periods)
    df["Z_Diff"] = (df["LogDev"] - exp.mean()) / exp.std()
    df["Cycle_Pct"] = cycle_percentile(df["LogDev"])
    return df


def _share_below_last(window):
    last = window[-1]
    if not np.isfinite(last):
        return np.nan
    past = window[:-1]
    past = past[np.isfinite(past)]
    return float((past < last).mean()) if len(past) else np.nan


def cycle_percentile(logdev, window=CYCLE_WINDOW, min_periods=CYCLE_MIN_PERIODS):
    """Cycle-relative valuation: share of the previous ~4 years (one halving epoch) of
    ln(price / SMA) values that lie below today's value. 0 = cheapest, 1 = richest.

    Why: each cycle has peaked lower relative to the 200-week SMA (about 16x in 2017,
    about 6x in 2021, about 2.5x in 2024/25), so a Z-score against ALL history can no longer
    reach a sell level. Ranking against one cycle is non-parametric (nothing is fitted,
    no distribution assumed), is unaffected by the decay in amplitude, and is causal: day t
    only sees days t-window+1 .. t. Only two free choices: the window (fixed in advance at
    one halving epoch) and the sell threshold.
    """
    s = pd.Series(logdev, dtype=float)
    return s.rolling(window, min_periods=min_periods).apply(_share_below_last, raw=True)


def signals(df, in_threshold=DEFAULT_IN_THRESHOLD, out_threshold=None, sell_mode="zscore"):
    """Boolean Series for DCA-IN and DCA-OUT.

    Buys always use the expanding Z-score (Z below in_threshold).
    Sells use either the expanding Z-score (``sell_mode="zscore"``, Z above out_threshold) or
    the 4-year cycle percentile (``sell_mode="cycle"``, Cycle_Pct above out_threshold, 0..1).
    ``out_threshold=None`` uses the default for the chosen mode.
    """
    if sell_mode not in SELL_MODES:
        raise ValueError(f"sell_mode must be one of {SELL_MODES}")
    if out_threshold is None:
        out_threshold = DEFAULT_CYCLE_OUT_THRESHOLD if sell_mode == "cycle" else DEFAULT_OUT_THRESHOLD
    sell_score = df["Cycle_Pct"] if sell_mode == "cycle" else df["Z_Diff"]
    return df["Z_Diff"] < in_threshold, sell_score > out_threshold


def z_change(df):
    """Latest Z value and its day-over-day change (today minus yesterday)."""
    z = df["Z_Diff"].dropna()
    return float(z.iloc[-1]), float(z.iloc[-1] - z.iloc[-2])
